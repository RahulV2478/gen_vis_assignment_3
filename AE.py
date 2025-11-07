# AE.py
import os, json, time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import CMUMotionDataset, recover_global_motion
from visualization import *

# ------------------------------
# Model
# ------------------------------
import torch.nn.functional as F

class MotionAutoencoder(nn.Module):
    """
    Temporal Conv1D Autoencoder for motion windows.
    Input x: [B, T, F]  (F = J*3 (+ 3 if root vel/rot appended))
    """
    def __init__(self, input_dim=63, bottleneck_ch=64):
        super().__init__()
        self.input_dim = input_dim

        # Encoder operates on [B, F, T]
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, bottleneck_ch, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )
        # Decoder mirrors encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(bottleneck_ch, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, self.input_dim, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x_btf):
        x_bft = x_btf.transpose(1, 2)        # [B,T,F] -> [B,F,T]
        z = self.encoder(x_bft)              # [B,C',T']
        return z

    def decode(self, z):
        xhat_bft = self.decoder(z)           # [B,F,T]
        xhat_btf = xhat_bft.transpose(1, 2)  # [B,T,F]
        return xhat_btf

    def forward(self, x, corrupt_input: bool=False, corruption_prob: float=0.1):
        T_in = x.size(1)
        if corrupt_input and self.training:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - corruption_prob))
            x = x * mask
        z = self.encode(x)
        xhat = self.decode(z)
        # Make time dim match input (robust to stride/pad drift)
        if xhat.size(1) != T_in:
            if xhat.size(1) > T_in:
                xhat = xhat[:, :T_in, :]
            else:
                pad = T_in - xhat.size(1)
                xhat = F.pad(xhat, (0, 0, 0, pad))  # pad time
        return xhat, z


# ------------------------------
# Trainer
# ------------------------------
class MotionManifoldTrainer:
    """Trainer for the Motion Manifold Conv1D Autoencoder."""
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 25,
        fine_tune_epochs: int = 25,
        learning_rate: float = 3e-4,      # saner defaults
        fine_tune_lr: float = 1e-4,
        sparsity_weight: float = 1e-2,
        window_size: int = 160,
        val_split: float = 0.1,
        device: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.sparsity_weight = sparsity_weight
        self.window_size = window_size
        self.val_split = val_split

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self._load_dataset()
        self._init_model()

    def _build_input(self, batch):
        # positions (normalized, flat) [B,T,F_pos]
        x = batch["positions_normalized_flat"].to(self.device).float()
        # append root velocities if present/used
        if getattr(self, "use_vel", False):
            tv = batch["trans_vel_xz"].to(self.device).float()            # [B,T,2]
            ry = batch["rot_vel_y"].to(self.device).float().unsqueeze(-1) # [B,T,1]
            x = torch.cat([x, tv, ry], dim=-1)
        return x

    def _load_dataset(self):
        self.dataset = CMUMotionDataset(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            frame_rate=30,
            window_size=self.window_size,
            overlap=0.5,
            include_velocity=True,
            include_foot_contact=True
        )

        val_size = max(1, int(self.val_split * len(self.dataset)))
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=0, pin_memory=True)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False,
                                       num_workers=0, pin_memory=True)

        print(f"Dataset loaded with {len(self.dataset)} windows from {len(self.dataset.motion_data)} files")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")

        self.mean_pose = torch.tensor(self.dataset.get_mean_pose(), device=self.device, dtype=torch.float32)
        self.std       = torch.tensor(self.dataset.get_std(), device=self.device, dtype=torch.float32)
        self.joint_names   = self.dataset.get_joint_names()
        self.joint_parents = self.dataset.get_joint_parents()

    def _init_model(self):
        sample = self.dataset[0]
        if "positions_flat" in sample:
            posF = sample["positions_flat"].shape[1]
            if "trans_vel_xz" in sample and "rot_vel_y" in sample:
                self.use_vel = True
                input_dim = posF + sample["trans_vel_xz"].shape[1] + 1  # +2 tv +1 ry
                print(f"Input includes positions ({posF}) + velocities (3) -> {input_dim}")
            else:
                self.use_vel = False
                input_dim = posF
                print(f"Input only positions ({input_dim})")
        else:
            positions = sample["positions"]  # [T,J,3]
            self.use_vel = False
            input_dim = positions.shape[1] * positions.shape[2]
            print(f"Fallback input_dim: {input_dim}")

        self.model = MotionAutoencoder(input_dim=input_dim, bottleneck_ch=64).to(self.device)
        print(f"Created model with input dimension: {input_dim}")

    def train(self):
        s1 = self._train_phase(self.epochs, self.learning_rate, 0.1, self.sparsity_weight, "phase1")
        s2 = self._train_phase(self.fine_tune_epochs, self.fine_tune_lr, 0.0, self.sparsity_weight * 0.5, "phase2")
        all_stats = {"phase1": s1, "phase2": s2}

        with open(os.path.join(self.output_dir, "training_stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)

        self._save_model()
        self._save_normalization_params()
        self._plot_training_curves(all_stats)
        return all_stats

    def _train_phase(self, epochs, learning_rate, corruption_prob, sparsity_weight, phase_name):
        print(f"\n===== {phase_name.capitalize()} Training Phase =====")
        opt = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        crit = nn.MSELoss()
        vel_w = 0.5

        stats = {"train_total": [], "train_rec": [], "train_vel": [],
                 "val_total": [], "val_rec": [], "val_vel": []}
        best_val = float("inf")

        for epoch in range(epochs):
            # ---- Train ----
            self.model.train()
            tr_rec = tr_vel = tr_tot = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                x = self._build_input(batch)                 # [B,T,F]
                xhat, z = self.model(x, corrupt_input=True, corruption_prob=corruption_prob)

                posF = batch["positions_normalized_flat"].shape[2]
                x_pos, xhat_pos = x[..., :posF], xhat[..., :posF]
                loss_rec = crit(xhat_pos, x_pos)
                loss_vel = crit(xhat_pos[:, 1:] - xhat_pos[:, :-1],
                                x_pos[:, 1:]    - x_pos[:, :-1])
                loss_spr = torch.mean(torch.abs(z))
                loss = loss_rec + vel_w * loss_vel + sparsity_weight * loss_spr

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                tr_rec += loss_rec.item(); tr_vel += loss_vel.item(); tr_tot += loss.item()

            stats["train_rec"].append(tr_rec / max(1, len(self.train_loader)))
            stats["train_vel"].append(tr_vel / max(1, len(self.train_loader)))
            stats["train_total"].append(tr_tot / max(1, len(self.train_loader)))

            # ---- Val ----
            self.model.eval()
            va_rec = va_vel = va_tot = 0.0
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    x = self._build_input(batch)
                    xhat, z = self.model(x, corrupt_input=False)

                    posF = batch["positions_normalized_flat"].shape[2]
                    x_pos, xhat_pos = x[..., :posF], xhat[..., :posF]
                    loss_rec = crit(xhat_pos, x_pos)
                    # FIXED: symmetric slicing
                    loss_vel = crit(xhat_pos[:, 1:] - xhat_pos[:, :-1],
                                    x_pos[:, 1:]    - x_pos[:, :-1])
                    loss_spr = torch.mean(torch.abs(z))
                    loss = loss_rec + vel_w * loss_vel + sparsity_weight * loss_spr

                    va_rec += loss_rec.item(); va_vel += loss_vel.item(); va_tot += loss.item()

            val_loss = va_tot / max(1, len(self.val_loader))
            stats["val_rec"].append(va_rec / max(1, len(self.val_loader)))
            stats["val_vel"].append(va_vel / max(1, len(self.val_loader)))
            stats["val_total"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(epoch, end=f"_valloss_{val_loss:.6f}", phase_name=phase_name)
                print(f"  Saved checkpoint with val_loss: {val_loss:.6f}")

            sched.step()

        return stats

    def _save_checkpoint(self, epoch, end, phase_name):
        path = os.path.join(self.output_dir, "checkpoints", f"{phase_name}_epoch_{epoch+1}{end}.pt")
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()}, path)

    def _save_model(self):
        path = os.path.join(self.output_dir, "models", "motion_autoencoder.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def _save_normalization_params(self):
        np.savez(os.path.join(self.output_dir, "normalization.npz"),
                 mean_pose=self.mean_pose.cpu().numpy(),
                 std=self.std.cpu().numpy(),
                 joint_names=np.array(self.joint_names, dtype=object),
                 joint_parents=np.array(self.joint_parents, dtype=object))
        print(f"Normalization parameters saved to {self.output_dir}/normalization.npz")

    def _plot_training_curves(self, stats):
        if not isinstance(stats[list(stats.keys())[0]], dict):
            stats = {"train": stats}
        n_p = len(stats)
        plt.figure(figsize=(12, 4 * n_p))
        for i, (phase_name, phase_stats) in enumerate(stats.items()):
            plt.subplot(n_p, 1, i+1)
            for key, values in phase_stats.items():
                plt.plot(values, label=key)
            plt.title(f"{phase_name.capitalize()} Training Phase")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "training_curves.png"))
        plt.close()
        print(f"Training curves saved to {self.output_dir}/plots/training_curves.png")


# ------------------------------
# Synthesizer (Interpolation / Repair)
# ------------------------------
class MotionManifoldSynthesizer:
    """Generate, fix, and interpolate motions using the learned manifold."""
    def __init__(self, model_path: str, dataset: CMUMotionDataset, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self._load_normalization(dataset)
        self._load_model(model_path)

    def _load_normalization(self, dataset: CMUMotionDataset):
        self.mean_pose = torch.tensor(dataset.mean_pose, device=self.device, dtype=torch.float32)  # [J,3]
        self.std       = torch.tensor(dataset.std, device=self.device, dtype=torch.float32)        # [J,3]
        self.joint_names   = dataset.joint_names
        self.joint_parents = dataset.joint_parents
        self.posF = int(self.mean_pose.numel())  # J*3

    def _norm_flat(self, x_btj3: torch.Tensor):
        x = (x_btj3 - self.mean_pose[None, None, :]) / (self.std[None, None, :] + 1e-8)
        B, T, J, _ = x.shape
        return x.view(B, T, J * 3)

    def _denorm_flat(self, x_btf: torch.Tensor):
        B, T, F = x_btf.shape
        J = self.mean_pose.shape[0]
        x = x_btf.view(B, T, J, 3)
        x = x * (self.std[None, None, :] + 1e-8) + self.mean_pose[None, None, :]
        return x

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        state = torch.load(model_path, map_location=self.device)
        # Accept checkpoints: {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Infer input_dim from first conv weight if available
        first_w = None
        for k in state.keys():
            if k.endswith("encoder.0.weight") or k.endswith("enc.0.weight") or ("encoder.0.weight" in k):
                first_w = state[k]; break
        if first_w is not None:
            input_dim = int(first_w.shape[1])
        else:
            input_dim = self.posF  # fallback
        self.input_dim = input_dim

        self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        print(f"Model loaded from {model_path} (input_dim={self.input_dim})")

    def _build_x(self, motion: Dict[str, torch.Tensor]):
        # Build inference input consistent with training channels
        xpos = self._norm_flat(motion["positions"].to(self.device).float())     # [B,T,posF]
        if self.input_dim == self.posF + 3:
            tv = motion["trans_vel_xz"].to(self.device).float()                 # [B,T,2]
            ry = motion["rot_vel_y"].to(self.device).float().unsqueeze(-1)     # [B,T,1]
            return torch.cat([xpos, tv, ry], dim=-1)                            # [B,T,posF+3]
        elif self.input_dim == self.posF:
            return xpos
        else:
            raise ValueError(f"Model expects {self.input_dim} channels; posF={self.posF} (posF+3={self.posF+3}).")

    def fix_corrupted_motion(self, motion, corruption_type='zero', corruption_params=None):
        positions = motion['positions'].to(self.device).float()  # [B,T,J,3]
        if corruption_params is not None:
            corrupted = self._apply_corruption(positions, corruption_type, corruption_params)
        else:
            corrupted = positions.clone()

        x_in = self._build_x({"positions": corrupted,
                              "trans_vel_xz": motion.get("trans_vel_xz", None),
                              "rot_vel_y": motion.get("rot_vel_y", None)})
        with torch.no_grad():
            xhat, _ = self.model(x_in, corrupt_input=False)
        fixed_local = self._denorm_flat(xhat[..., :self.posF])

        if ('trans_vel_xz' in motion) and ('rot_vel_y' in motion):
            fixed_global = recover_global_motion(
                fixed_local,
                motion['trans_vel_xz'].to(self.device).float(),
                motion['rot_vel_y'].to(self.device).float()
            )
            return corrupted, fixed_global
        return corrupted, fixed_local

    def _apply_corruption(self, motion, corruption_type, params):
        corrupted = motion.clone()
        if corruption_type == 'zero':
            prob = params.get('prob', 0.5)
            mask = torch.bernoulli(torch.ones_like(corrupted) * (1 - prob))
            corrupted = corrupted * mask
        elif corruption_type == 'noise':
            scale = params.get('scale', 0.1)
            corrupted = corrupted + torch.randn_like(corrupted) * scale
        elif corruption_type == 'missing':
            j = params.get('joint_idx', 0)
            corrupted[:, :, j, :] = 0.0
        return corrupted

    def interpolate_motions(self, motion1, motion2, t: float):
        """
        Latent linear interpolation with input channels matching training.
        Returns local joints [B,T,J,3]; caller can recover globals with root vel/rot.
        """
        x1 = self._build_x(motion1)
        x2 = self._build_x(motion2)
        with torch.no_grad():
            _, z1 = self.model(x1, corrupt_input=False)
            _, z2 = self.model(x2, corrupt_input=False)
            zt = (1.0 - t) * z1 + t * z2
            xhat = self.model.decode(zt)                 # [B,T,F]
            xhat_pos = xhat[..., :self.posF]             # keep positions only
        return self._denorm_flat(xhat_pos)


# ------------------------------
# Example entry
# ------------------------------
def main():
    """Example: train quickly (adjust paths as needed)."""
    data_dir = "./cmu-mocap"
    output_dir = "./runs/ae_quick"

    trainer = MotionManifoldTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=32,
        epochs=10,
        fine_tune_epochs=10,
        learning_rate=1e-3,
        fine_tune_lr=5e-4,
        sparsity_weight=1e-2,
        window_size=160,
        val_split=0.1
    )
    trainer.train()

if __name__ == "__main__":
    main()