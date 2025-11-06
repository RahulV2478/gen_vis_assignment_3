import os
import glob
import pickle
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

from visualization import visualize_motion_to_video


def recover_global_motion(local_motion, trans_vel_xz, rot_vel_y, frame_rate=30.0):
    """
    local_motion: [B,T,J,3] (torch)
    trans_vel_xz: [B,T,2]   (torch)  velocities in m/s
    rot_vel_y:    [B,T]     (torch)  rad/s
    """
    B, T, J, _ = local_motion.shape
    global_motion = local_motion.clone()

    for b in range(B):
        gtrans = torch.zeros((T, 2), device=local_motion.device, dtype=local_motion.dtype)
        grot = torch.zeros(T, device=local_motion.device, dtype=local_motion.dtype)
        for t in range(1, T):
            gtrans[t] = gtrans[t - 1] + trans_vel_xz[b, t - 1] / frame_rate
            grot[t] = grot[t - 1] + rot_vel_y[b, t - 1] / frame_rate

        for t in range(T):
            ang = grot[t]
            c, s = torch.cos(ang), torch.sin(ang)
            x = global_motion[b, t, :, 0].clone()
            z = global_motion[b, t, :, 2].clone()
            global_motion[b, t, :, 0] = c * x + s * z
            global_motion[b, t, :, 2] = -s * x + c * z
            global_motion[b, t, :, 0] += gtrans[t, 0]
            global_motion[b, t, :, 2] += gtrans[t, 1]

    return global_motion


class CMUMotionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cache_dir: str = None,
        frame_rate: int = 60,
        window_size: int = 240,
        overlap: float = 0.5,
        joint_selection: Optional[List[str]] = None,
        include_velocity: bool = True,
        include_foot_contact: bool = True,
        force_recompute: bool = False
    ):
        self.data_root = data_dir
        self.data_dir = os.path.join(data_dir, "data")
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.overlap = overlap
        self.joint_selection = joint_selection
        self.include_velocity = include_velocity
        self.include_foot_contact = include_foot_contact

        os.makedirs(self.cache_dir, exist_ok=True)

        self.file_list = glob.glob(os.path.join(self.data_dir, "**/*.bvh"), recursive=True)
        self.file_list.sort()
        print(f"Found {len(self.file_list)} BVH files in {self.data_dir}")

        self.cache_id = self._create_cache_id()
        self._load_or_compute_dataset(force_recompute)

    def _create_cache_id(self) -> str:
        s = f"fr{self.frame_rate}_ws{self.window_size}_ol{self.overlap}"
        s += f"_vel{int(self.include_velocity)}_fc{int(self.include_foot_contact)}"
        if self.joint_selection:
            s += f"_js{'_'.join(self.joint_selection)}"
        return s

    def _compute_statistics(self, motion_data: Dict) -> None:
        first_key = next(iter(motion_data))
        J3_shape = motion_data[first_key]["local_positions"].shape[1:]

        sum_x = np.zeros(J3_shape, dtype=np.float64)
        sum_x2 = np.zeros(J3_shape, dtype=np.float64)
        N = 0

        for data in motion_data.values():
            X = data["local_positions"]  # [T,J,3] numpy
            sum_x += X.sum(axis=0)
            sum_x2 += (X * X).sum(axis=0)
            N += X.shape[0]

        mean = (sum_x / max(N, 1)).astype(np.float32)
        var = (sum_x2 / max(N, 1)) - (mean.astype(np.float64) ** 2)
        std = np.sqrt(np.clip(var, 1e-8, None)).astype(np.float32)

        self.mean_pose = mean
        self.std = std

        stats_path = os.path.join(self.cache_dir, f"stats_{self.cache_id}.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump({"mean_pose": self.mean_pose, "std": self.std}, f)

    def _load_or_compute_dataset(self, force_recompute: bool = False) -> None:
        cache_file = os.path.join(self.cache_dir, f"dataset_{self.cache_id}.pkl")

        if os.path.exists(cache_file) and not force_recompute:
            print(f"Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            self.windows = cache_data["windows"]
            self.motion_data = cache_data["motion_data"]
            self.joint_names = cache_data["joint_names"]
            self.joint_parents = cache_data["joint_parents"]
            print(f"Loaded {len(self.windows)} windows")
        else:
            print("Computing dataset and caching results...")
            self._preprocess_dataset()
            cache_data = {
                "windows": self.windows,
                "motion_data": self.motion_data,
                "joint_names": self.joint_names,
                "joint_parents": self.joint_parents,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(self.windows)} windows to cache: {cache_file}")

        stats_file = os.path.join(self.cache_dir, f"stats_{self.cache_id}.pkl")
        if os.path.exists(stats_file):
            with open(stats_file, "rb") as f:
                stats = pickle.load(f)
            self.mean_pose = stats["mean_pose"]
            self.std = stats["std"]
        else:
            self._compute_statistics(self.motion_data)

    def _preprocess_dataset(self) -> None:
        self.motion_data = {}
        all_local_positions = []

        for file_path in tqdm.tqdm(self.file_list, desc="Processing BVH files"):
            try:
                from read_bvh import BVHReader
                reader = BVHReader(file_path)
                reader.read()

                joint_positions = reader.get_joint_positions_batch()  # torch [T,J,3] or numpy-like
                num_frames = joint_positions.shape[0]
                if num_frames < self.window_size:
                    continue

                root_translation = reader.root_translation.numpy()
                root_rotation = reader.root_rotation.numpy()
                joint_names = reader.joint_names
                joint_parents = reader.joint_parents.tolist()
                joint_offsets = reader.joint_offsets.numpy()

                if abs(1.0 / reader.frame_time - self.frame_rate) > 1e-5:
                    original_fps = 1.0 / reader.frame_time
                    joint_positions = self._resample_positions(joint_positions.numpy(), original_fps)
                    root_rotation = self._resample_positions(root_rotation, original_fps)
                    root_translation = self._resample_positions(root_translation, original_fps)
                else:
                    joint_positions = joint_positions.numpy()

                root_rotation = np.deg2rad(root_rotation)
                root_rotation_y = root_rotation[..., 1]
                num_frames = joint_positions.shape[0]

                foot_heights = self._get_foot_heights(joint_positions, joint_names)
                if np.mean(foot_heights) > 1.0:
                    print(f"Warning: Unusual foot heights detected in {file_path}.")

                local_positions = self._remove_global_transforms(
                    joint_positions, root_translation, root_rotation_y
                )

                trans_vel_xz = np.zeros((num_frames, 2), dtype=np.float32)
                rot_vel_y = np.zeros(num_frames, dtype=np.float32)
                if num_frames > 1:
                    trans_vel_xz[1:, 0] = root_translation[1:, 0] - root_translation[:-1, 0]
                    trans_vel_xz[1:, 1] = root_translation[1:, 2] - root_translation[:-1, 2]
                    ang_diff = root_rotation_y[1:] - root_rotation_y[:-1]
                    ang_diff = np.arctan2(np.sin(ang_diff), np.cos(ang_diff))
                    rot_vel_y[1:] = ang_diff
                    trans_vel_xz *= self.frame_rate
                    rot_vel_y *= self.frame_rate

                bvh_data = {
                    "positions": joint_positions.astype(np.float32),
                    "local_positions": local_positions.astype(np.float32),
                    "joint_names": joint_names,
                    "joint_parents": joint_parents,
                    "joint_offsets": joint_offsets.astype(np.float32),
                    "num_frames": num_frames,
                    "root_translation": root_translation.astype(np.float32),
                    "root_rotation": root_rotation.astype(np.float32),
                    "trans_vel_xz": trans_vel_xz,
                    "rot_vel_y": rot_vel_y,
                }

                if self.include_velocity:
                    velocities = self._calculate_velocities(joint_positions)
                    bvh_data["velocities"] = velocities.astype(np.float32)

                if self.include_foot_contact:
                    foot_contacts = self._detect_foot_contacts(joint_positions, joint_names)
                    bvh_data["foot_contacts"] = foot_contacts.astype(np.float32)

                self.motion_data[file_path] = bvh_data
                all_local_positions.append(local_positions)

                if not hasattr(self, "joint_names"):
                    self.joint_names = joint_names
                    self.joint_parents = joint_parents

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        if not all_local_positions:
            raise ValueError("No valid motion files found. Check your data directory.")

        self.windows = self._create_windows()
        print(f"Processed {len(self.motion_data)} files with {len(self.windows)} valid windows")

    def _resample_positions(self, positions, original_fps):
        T = positions.shape[0]
        duration = T / original_fps
        new_T = int(duration * self.frame_rate)
        new_positions = np.zeros((new_T, *positions.shape[1:]), dtype=np.float32)
        for i in range(new_T):
            src = i * original_fps / self.frame_rate
            f1 = int(np.floor(src))
            f2 = min(f1 + 1, T - 1)
            a = src - f1
            new_positions[i] = positions[f1] * (1 - a) + positions[f2] * a
        return new_positions

    def _create_windows(self) -> List[Tuple[str, int]]:
        windows = []
        stride = max(1, int(self.window_size * (1 - self.overlap)))
        for file_path, md in self.motion_data.items():
            T = md["positions"].shape[0]
            if T < self.window_size:
                continue
            for s in range(0, T - self.window_size + 1, stride):
                windows.append((file_path, s))
        return windows

    def _calculate_velocities(self, positions):
        T = positions.shape[0]
        vel = np.zeros_like(positions, dtype=np.float32)
        if T <= 1:
            return vel
        vel[1:] = (positions[1:] - positions[:-1]) * self.frame_rate
        vel[0] = vel[1]
        return vel

    def _get_foot_heights(self, positions, joint_names):
        left_foot_idx = 4
        right_foot_idx = 10
        heights = []
        if left_foot_idx is not None:
            heights.append(np.mean(positions[:, left_foot_idx, 1]))
        if right_foot_idx is not None:
            heights.append(np.mean(positions[:, right_foot_idx, 1]))
        return np.mean(heights) if heights else 0.0

    def _detect_foot_contacts(self, positions, joint_names):
        T = positions.shape[0]
        left_foot_idx, right_foot_idx = 4, 10
        left_toe_idx, right_toe_idx = 5, 11
        contacts = np.zeros((T, 4), dtype=np.float32)

        ground_level = min(
            np.min(positions[:, left_foot_idx, 1]),
            np.min(positions[:, right_foot_idx, 1]),
            np.min(positions[:, left_toe_idx, 1]),
            np.min(positions[:, right_toe_idx, 1]),
        )
        hth = ground_level + 0.05
        vth = 0.15

        lfh = positions[:, left_foot_idx, 1]
        rfh = positions[:, right_foot_idx, 1]
        lth = positions[:, left_toe_idx, 1]
        rth = positions[:, right_toe_idx, 1]

        lfv = np.zeros_like(lfh); lfv[1:] = (lfh[1:] - lfh[:-1]) * self.frame_rate
        rfv = np.zeros_like(rfh); rfv[1:] = (rfh[1:] - rfh[:-1]) * self.frame_rate
        ltv = np.zeros_like(lth); ltv[1:] = (lth[1:] - lth[:-1]) * self.frame_rate
        rtv = np.zeros_like(rth); rtv[1:] = (rth[1:] - rth[:-1]) * self.frame_rate

        contacts[:, 0] = (lfh < hth) & (np.abs(lfv) < vth)
        contacts[:, 1] = (lth < hth) & (np.abs(ltv) < vth)
        contacts[:, 2] = (rfh < hth) & (np.abs(rfv) < vth)
        contacts[:, 3] = (rth < hth) & (np.abs(rtv) < vth)

        contacts = contacts * 2 - 1
        return contacts

    def _remove_global_transforms(self, positions, root_translation, root_rotation_y):
        T, J, _ = positions.shape
        local = positions.copy()
        local[..., 0] -= root_translation[:, None, 0]
        local[..., 2] -= root_translation[:, None, 2]
        for f in range(T):
            ang = root_rotation_y[f]
            c, s = np.cos(ang), np.sin(ang)
            x = local[f, :, 0].copy()
            z = local[f, :, 2].copy()
            local[f, :, 0] = c * x - s * z
            local[f, :, 2] = s * x + c * z
        return local

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_path, s = self.windows[idx]
        md = self.motion_data[file_path]
        e = s + self.window_size

        local = md["local_positions"][s:e].copy()  # [T,J,3]
        orig = md["positions"][s:e].copy()
        orig[:, :, 0] -= orig[:1, :1, 0]
        orig[:, :, 2] -= orig[:1, :1, 2]

        trans_vel_xz = md["trans_vel_xz"][s:e].copy()
        rot_vel_y = md["rot_vel_y"][s:e].copy()

        eps = 1e-8
        positions_normalized = (local - self.mean_pose[None, :, :]) / (self.std[None, :, :] + eps)

        T, J, _ = local.shape
        positions_flat = local.reshape(T, J * 3)
        positions_normalized_flat = positions_normalized.reshape(T, J * 3)

        result = {
            "positions": torch.tensor(local, dtype=torch.float32),
            "positions_normalized": torch.tensor(positions_normalized, dtype=torch.float32),
            "positions_flat": torch.tensor(positions_flat, dtype=torch.float32),
            "positions_normalized_flat": torch.tensor(positions_normalized_flat, dtype=torch.float32),
            "trans_vel_xz": torch.tensor(trans_vel_xz, dtype=torch.float32),
            "rot_vel_y": torch.tensor(rot_vel_y, dtype=torch.float32),
            "root_positions": torch.tensor(md["positions"][s:e, 0, :], dtype=torch.float32),
            "original_positions": torch.tensor(orig, dtype=torch.float32),
        }

        if self.include_foot_contact and "foot_contacts" in md:
            result["foot_contacts"] = torch.tensor(md["foot_contacts"][s:e], dtype=torch.float32)
        if self.include_velocity and "velocities" in md:
            result["velocities"] = torch.tensor(md["velocities"][s:e], dtype=torch.float32)

        return result

    def get_mean_pose(self):
        return self.mean_pose

    def get_std(self):
        return self.std

    def get_joint_names(self):
        return self.joint_names

    def get_joint_parents(self):
        return self.joint_parents


# ---------- Video export helpers ----------

def export_sample_videos(dataset: CMUMotionDataset, out_dir: str, idx: int, fps: int = 30):
    os.makedirs(out_dir, exist_ok=True)
    sample = dataset[idx]
    jp = dataset.get_joint_parents()

    raw_path = os.path.join(out_dir, f"sample_{idx}_raw.mp4")
    visualize_motion_to_video(sample["positions"], jp, raw_path, fps=fps)

    norm_path = os.path.join(out_dir, f"sample_{idx}_normalized.mp4")
    visualize_motion_to_video(sample["positions_normalized"], jp, norm_path, fps=fps)

    if ("trans_vel_xz" in sample) and ("rot_vel_y" in sample):
        rec = recover_global_motion(
            sample["positions"].unsqueeze(0),
            sample["trans_vel_xz"].unsqueeze(0),
            sample["rot_vel_y"].unsqueeze(0),
            frame_rate=fps
        )[0]
        rec_path = os.path.join(out_dir, f"sample_{idx}_recovered.mp4")
        visualize_motion_to_video(rec, jp, rec_path, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./cmu-mocap")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--out_dir", default="./cmu-mocap/visualizations")
    parser.add_argument("--window_size", type=int, default=160)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = CMUMotionDataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        frame_rate=args.fps,
        window_size=args.window_size,
        overlap=args.overlap,
        include_velocity=True,
        include_foot_contact=True,
        force_recompute=args.force_recompute
    )

    print(f"Dataset size: {len(dataset)} windows")
    print(f"Writing {args.num_examples} samples to: {args.out_dir}")

    end = min(len(dataset), args.start_index + args.num_examples)
    for i in range(args.start_index, end):
        print(f"Exporting sample {i}...")
        export_sample_videos(dataset, args.out_dir, i, fps=args.fps)

    print("Done.")