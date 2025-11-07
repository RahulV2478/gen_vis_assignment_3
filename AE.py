import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import CMUMotionDataset
from visualization import *

class MotionAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Motion Data as described in the paper
    "Learning Motion Manifolds with Convolutional Autoencoders"
    """
    def __init__(self, input_dim=63, bottleneck_ch=64):
        super(MotionAutoencoder, self).__init__()
        self.input_dim = input_dim

        # Encoder/decoder operate over time with features as channels: [B,T,F] -> [B,F,T]
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, bottleneck_ch, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(bottleneck_ch, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, input_dim, kernel_size=4, stride=2, padding=1)
        )
    
    def encode(self, x):
        x = x.transpose(1, 2)          # [B,T,F] -> [B,F,T]
        z = self.encoder(x)            # [B,k,T']
        return z
    
    def decode(self, z):
        xhat = self.decoder(z)         # [B,F,T]
        return xhat.transpose(1, 2)    # [B,T,F]
    
    def forward(self, x, corrupt_input=False, corruption_prob=0.1):
        if corrupt_input and self.training:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - corruption_prob))
            x = x * mask
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


class MotionManifoldTrainer:
    """Trainer for the Motion Manifold Convolutional Autoencoder"""
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 25,
        fine_tune_epochs: int = 25,
        learning_rate: float = 0.5,
        fine_tune_lr: float = 0.01,
        sparsity_weight: float = 0.01,
        window_size: int = 160,
        val_split: float = 0.1,
        device: str = None
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
        x = batch["positions_normalized_flat"].to(self.device).float()  # [B,T,F_pos]
        if getattr(self, "use_vel", False):
            tv = batch["trans_vel_xz"].to(self.device).float()          # [B,T,2]
            ry = batch["rot_vel_y"].to(self.device).float().unsqueeze(-1)  # [B,T,1]
            x = torch.cat([x, tv, ry], dim=-1)                           # concat on feature dim
        return x

    def _load_dataset(self):
        """Load the CMU Motion dataset and create training/validation splits"""
        # Use the same dataloader module as imported at top
        self.dataset = CMUMotionDataset(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            frame_rate=30,
            window_size=self.window_size,
            overlap=0.5,
            include_velocity=True,
            include_foot_contact=True
        )
        
        val_size = int(self.val_split * len(self.dataset))
        val_size = max(1, val_size)
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Colab-safe workers
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"Dataset loaded with {len(self.dataset)} windows from {len(self.dataset.motion_data)} files")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        self.mean_pose = torch.tensor(self.dataset.get_mean_pose(), device=self.device, dtype=torch.float32)
        self.std = torch.tensor(self.dataset.get_std(), device=self.device, dtype=torch.float32)
        self.joint_names = self.dataset.get_joint_names()
        self.joint_parents = self.dataset.get_joint_parents()
        
    def _init_model(self):
        """Initialize the motion autoencoder model"""
        sample = self.dataset[0]
        if "positions_flat" in sample:
            posF = sample["positions_flat"].shape[1]
            if "trans_vel_xz" in sample and "rot_vel_y" in sample:
                self.use_vel = True
                input_dim = posF + sample["trans_vel_xz"].shape[1] + 1
                print(f"Input includes positions ({posF}) + velocities (3) -> {input_dim}")
            else:
                self.use_vel = False
                input_dim = posF
                print(f"Input only positions ({input_dim})")
        else:
            positions = sample["positions"]
            self.use_vel = False
            input_dim = positions.shape[1] * positions.shape[2]
            print(f"Fallback input_dim: {input_dim}")
        
        self.model = MotionAutoencoder(input_dim=input_dim, bottleneck_ch=64).to(self.device)
        print(f"Created model with input dimension: {input_dim}")
        
    def train(self):
        s1 = self._train_phase(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            corruption_prob=0.1,
            sparsity_weight=self.sparsity_weight,
            phase_name="phase1"
        )
        s2 = self._train_phase(
            epochs=self.fine_tune_epochs,
            learning_rate=self.fine_tune_lr,
            corruption_prob=0.0,
            sparsity_weight=self.sparsity_weight * 0.5,
            phase_name="phase2"
        )
        all_stats = {"phase1": s1, "phase2": s2}
        
        with open(os.path.join(self.output_dir, "training_stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)
        self._save_model()
        self._save_normalization_params()
        self._plot_training_curves(all_stats)
        return all_stats

    def _train_phase(self, epochs, learning_rate, corruption_prob, sparsity_weight, phase_name):
        """Train the model for a specific phase (initial training or fine-tuning)"""
        print(f"\n===== {phase_name.capitalize()} Training Phase =====")
        
        opt = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        crit = nn.MSELoss()
        vel_w = 0.5

        stats = {
            "train_total": [], "train_rec": [], "train_vel": [],
            "val_total": [], "val_rec": [], "val_vel": []
        }
        best_val_loss = float("inf")
        
        for epoch in range(epochs):
            self.model.train()
            tr_rec = tr_vel = tr_tot = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in progress_bar:
                x = self._build_input(batch)                       # [B,T,F]
                xhat, z = self.model(x, corrupt_input=True, corruption_prob=corruption_prob)
                # velocity loss on position features only (if velocities are appended)
                posF = batch["positions_normalized_flat"].shape[2]
                x_pos = x[..., :posF]
                xhat_pos = xhat[..., :posF]
                loss_rec = crit(xhat_pos, x_pos)
                loss_vel = crit(
                    xhat_pos[:,1:] - xhat_pos[:,:-1],
                    x_pos[:,1:]  - x_pos[:,:-1]
                )
                loss_spr = torch.mean(torch.abs(z))
                loss = loss_rec + vel_w * loss_vel + sparsity_weight * loss_spr
                opt.zero_grad(); loss.backward(); opt.step()
                tr_rec += loss_rec.item(); tr_vel += loss_vel.item(); tr_tot += loss.item()
                progress_bar.set_postfix({"rec": f"{loss_rec.item():.4f}", "vel": f"{loss_vel.item():.4f}"})
            ntr = len(self.train_loader)
            stats["train_rec"].append(tr_rec/ntr); stats["train_vel"].append(tr_vel/ntr); stats["train_total"].append(tr_tot/ntr)
            
            self.model.eval()
            va_rec = va_vel = va_tot = 0.0
            with torch.no_grad():
                progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in progress_bar:
                    x = self._build_input(batch)
                    xhat, z = self.model(x, corrupt_input=False)
                    posF = batch["positions_normalized_flat"].shape[2]
                    x_pos = x[..., :posF]
                    xhat_pos = xhat[..., :posF]
                    loss_rec = crit(xhat_pos, x_pos)
                    loss_vel = crit(xhat_pos[:,1:] - xhat_pos[:,:-1], x_pos[:,1:] - x_pos[:,-(x_pos.size(1)-1):])  # same as above but explicit
                    loss_spr = torch.mean(torch.abs(z))
                    loss = loss_rec + vel_w * loss_vel + sparsity_weight * loss_spr
                    va_rec += loss_rec.item(); va_vel += loss_vel.item(); va_tot += loss.item()
                    progress_bar.set_postfix({"rec": f"{loss_rec.item():.4f}", "vel": f"{loss_vel.item():.4f}"})
            nva = len(self.val_loader)
            val_loss = va_tot/nva
            stats["val_rec"].append(va_rec/nva); stats["val_vel"].append(va_vel/nva); stats["val_total"].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, end=f'_valloss_{val_loss:.6f}', phase_name=phase_name)
                print(f"  Saved checkpoint with val_loss: {val_loss:.6f}")
        
        return stats
    
    def _save_checkpoint(self, epoch, end, phase_name):
        """Save a model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{phase_name}_epoch_{epoch+1}{end}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }, checkpoint_path)
    
    def _save_model(self):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, "models", "motion_autoencoder.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def _save_normalization_params(self):
        """Save normalization parameters for inference if needed"""
        norm_data = {
            "mean_pose": self.mean_pose.cpu().numpy(),
            "std": self.std.cpu().numpy(),
            "joint_names": self.joint_names,
            "joint_parents": self.joint_parents
        }
        np.save(os.path.join(self.output_dir, "normalization.npy"), norm_data)
        print(f"Normalization parameters saved to {self.output_dir}/normalization.npy")
    
    def _plot_training_curves(self, stats):
        if not isinstance(stats[list(stats.keys())[0]], dict):
            stats = {"train": stats}
        n_p = len(list(stats.keys()))
        plt.figure(figsize=(12, 4 * n_p))
        for i, (phase_name, phase_stats) in enumerate(stats.items()):
            plt.subplot(n_p, 1, i+1)
            for key, values in phase_stats.items():
                plt.plot(values, label=key)
            plt.title(f"{phase_name.capitalize()} Training Phase")
            plt.xlabel("Epoch")
            plt.ylabel("Statistics")
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "training_curves.png"))
        plt.close()
        print(f"Training curves saved to {self.output_dir}/plots/training_curves.png")


class MotionManifoldSynthesizer:
    """Synthesizer for generating, fixing, and analyzing motion using the learned manifold"""
    def __init__(
        self,
        model_path: str,
        dataset: CMUMotionDataset,
        device: str = None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self._load_normalization(dataset)
        self._load_model(model_path)
    
    def _load_normalization(self, dataset: CMUMotionDataset):
        self.mean_pose = torch.tensor(dataset.mean_pose, device=self.device, dtype=torch.float32)
        self.std = torch.tensor(dataset.std, device=self.device, dtype=torch.float32)
        self.joint_names = dataset.joint_names
        self.joint_parents = dataset.joint_parents

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
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            first_layer_weight = None
            for key in state.keys():
                if 'encoder.0.weight' in key or 'enc.0.weight' in key:
                    first_layer_weight = state[key]
                    break
            if first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]
                print(f"Inferred input dimension {input_dim} from model weights")
            else:
                input_dim = self.mean_pose.shape[0] * self.mean_pose.shape[1]
                print(f"Using fallback input dimension: {input_dim}")
            self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    def fix_corrupted_motion(self, motion, corruption_type='zero', corruption_params=None):
        positions = motion['positions'].to(self.device).float()  # [B,T,J,3]
        if corruption_params is not None:
            corrupted = self._apply_corruption(positions, corruption_type, corruption_params)
        else:
            corrupted = positions.clone()
        x_in = self._norm_flat(corrupted)
        with torch.no_grad():
            xhat_n, _ = self.model(x_in, corrupt_input=False)
        fixed_local = self._denorm_flat(xhat_n)
        if ('trans_vel_xz' in motion) and ('rot_vel_y' in motion):
            from dataloader import recover_global_motion
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
    
    def interpolate_motions(self, motion1, motion2, t):
        x1 = self._norm_flat(motion1['positions'].to(self.device).float())
        x2 = self._norm_flat(motion2['positions'].to(self.device).float())
        with torch.no_grad():
            _, z1 = self.model(x1, corrupt_input=False)
            _, z2 = self.model(x2, corrupt_input=False)
            zt = (1 - t) * z1 + t * z2
            xhat_n = self.model.decode(zt)
        return self._denorm_flat(xhat_n)
    
def main():
    """Example usage of the motion manifold training"""
    data_dir = "path/to/cmu-mocap"
    output_dir = "./output/ae"
    
    trainer = MotionManifoldTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=32,
        epochs=25,
        fine_tune_epochs=25,
        learning_rate=0.001,
        fine_tune_lr=0.0005,
        sparsity_weight=0.01,
        window_size=160,
        val_split=0.1
    )
    trainer.train()

if __name__ == "__main__":
    main()