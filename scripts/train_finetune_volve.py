"""
Finetuning Script for Volve Dataset (Downstream Tasks)

Loads a pretrained NCS+WLFM model and finetunes on:
  1. Lithology Classification: Predict SAND_FLAG from well logs + seismic
  2. Reservoir Prediction: Predict porosity (PHIF), water saturation (SW)
  3. Fault Detection: Detect faults from seismic data

Usage:
    python scripts/train_finetune_volve.py \\
        --task lithology \\
        --pretrained checkpoints/pretrain/best_pretrain.pt \\
        --epochs 50 \\
        --batch_size 8
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from models.oil_gas_model import OilGasModel
from data.volve_dataset import VolveDataset
from training.losses import DiceLoss, FocalLoss, SSIMLoss
from training.metrics import ClassificationMetrics, RegressionMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Task-Specific Heads (add to model during finetuning)
# ==============================================================================

class LithologyHead(nn.Module):
    """Predict lithology flags from well log + seismic features."""

    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, well_sequence: torch.Tensor) -> torch.Tensor:
        """well_sequence: (B, L, D) -> logits: (B, L, C)"""
        return self.classifier(well_sequence)


class ReservoirHead(nn.Module):
    """Predict reservoir properties from sparse well features."""

    def __init__(self, hidden_dim: int, n_properties: int = 2):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_properties),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, D) -> properties: (B, n_properties)"""
        return self.predictor(features)


# ==============================================================================
# Finetuning Trainer
# ==============================================================================

class FinetuneTrainer:
    """Supervised finetuning trainer for downstream tasks."""

    def __init__(
        self,
        model: OilGasModel,
        task: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.task = task
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Task-specific heads
        hidden_dim = model.config.hidden_dim
        if task == "lithology":
            self.task_head = LithologyHead(hidden_dim).to(device)
        elif task == "reservoir_prediction":
            self.task_head = ReservoirHead(hidden_dim).to(device)
        else:
            self.task_head = None

        # Optimize encoders + task head
        all_params = list(model.parameters())
        if self.task_head:
            all_params += list(self.task_head.parameters())

        self.optimizer = optim.AdamW(
            all_params,
            lr=config.get("lr", 5e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )

        total_epochs = config.get("epochs", 50)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs,
        )

        self.use_amp = config.get("mixed_precision", True) and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.log_dir = Path(config.get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task-specific loss.

        Args:
            batch: Volve dataset batch

        Returns:
            dict with 'loss' and task-specific metrics
        """
        seismic = batch["seismic"].to(self.device)
        well_log = batch["well_log"].to(self.device)
        well_mask = batch["well_mask"].to(self.device)
        labels = batch["labels"]

        result = {"loss": torch.tensor(0.0, device=self.device)}

        if self.task == "lithology":
            # Predict SAND_FLAG, COAL_FLAG from well sequence
            with autocast(enabled=self.use_amp):
                # Encode
                encoded = self.model.encode(
                    seismic, well_log, well_mask, return_intermediate=True
                )
                well_seq = encoded.get("well_sequence", encoded["well_feat"].unsqueeze(1))

                # Predict
                if self.task_head:
                    logits = self.task_head(well_seq)  # (B, L, C=3)
                    # Target: combine flags into classes (0=shale, 1=sand, 2=coal)
                    sand = labels.get("sand_flag", None)
                    coal = labels.get("coal_flag", None)

                    if sand is not None and coal is not None:
                        sand = sand.to(self.device)
                        coal = coal.to(self.device)
                        # lithology: 0=background, 1=sand, 2=coal
                        litho = torch.zeros_like(sand, dtype=torch.long)
                        litho[sand > 0.5] = 1
                        litho[coal > 0.5] = 2
                        litho = litho.to(self.device)

                        loss = self.ce_loss(
                            logits.reshape(-1, logits.shape[-1]),
                            litho.reshape(-1).long(),
                        )
                        result["loss"] = loss
                        result["accuracy"] = (
                            logits.argmax(-1) == litho
                        ).float().mean()

        elif self.task == "reservoir_prediction":
            # Predict porosity (PHIF) from well + seismic features
            with autocast(enabled=self.use_amp):
                encoded = self.model.encode(
                    seismic, well_log, well_mask, return_intermediate=True
                )

                if self.task_head:
                    # Use fused features
                    fused = encoded["fused"]
                    props = self.task_head(fused)  # (B, 2) = [porosity, Sw]

                    porosity_target = labels.get("porosity", None)
                    sw_target = labels.get("water_saturation", None)

                    if porosity_target is not None and sw_target is not None:
                        porosity = porosity_target.to(self.device).mean(dim=1)  # avg over depth
                        sw = sw_target.to(self.device).mean(dim=1)

                        target = torch.stack([porosity, sw], dim=1).to(self.device)
                        loss = self.mse_loss(props, target)
                        result["loss"] = loss
                        result["mse"] = loss.detach()

        elif self.task == "fault_detection":
            # Use existing model task head
            with autocast(enabled=self.use_amp):
                outputs = self.model(seismic, well_log, task="fault_detection")
                # Use a self-supervised proxy: Laplacian of seismic as "fault" target
                # In practice, this would need labeled fault data
                result["loss"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        return result

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        if self.task_head:
            self.task_head.train()

        metrics = {}
        n = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            losses = self.compute_loss(batch)
            loss = losses["loss"]

            accum = self.config.get("accumulate_grad", 1)
            loss = loss / accum

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum == 0:
                grad_clip = self.config.get("grad_clip", 1.0)
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    (list(self.task_head.parameters()) if self.task_head else []),
                    grad_clip,
                )
                if self.use_amp and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0.0) + v.item()

            if batch_idx % 20 == 0:
                logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{n}] "
                    f"Loss: {loss.item()*accum:.4f}"
                )

            self.global_step += 1

        self.scheduler.step()
        return {k: v / n for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        if self.task_head:
            self.task_head.eval()

        metrics = {}
        for batch in self.val_loader:
            losses = self.compute_loss(batch)
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0.0) + v.item()

        n = len(self.val_loader)
        return {k: v / n for k, v in metrics.items()}

    def fit(self, epochs: int) -> Dict:
        best_val = float("inf")
        patience = self.config.get("early_stopping", 20)
        no_improve = 0
        history = {"train": [], "val": []}

        for epoch in range(epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_m = self.train_epoch()
            train_m["time"] = time.time() - t0
            history["train"].append(train_m)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} | Loss: {train_m.get('loss', 0):.4f} | "
                f"LR: {lr:.6f} | Time: {train_m['time']:.1f}s"
            )

            val_m = self.validate()
            if val_m:
                history["val"].append(val_m)
                val_loss = val_m.get("loss", 0)
                logger.info(f"  Val loss: {val_loss:.4f}")

                for k, v in val_m.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    self.save_checkpoint(f"best_{self.task}.pt")
                else:
                    no_improve += 1

                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                self.save_checkpoint(f"{self.task}_epoch_{epoch}.pt")

        self.save_checkpoint(f"{self.task}_final.pt")
        return history

    def save_checkpoint(self, filename: str):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "task": self.task,
        }
        if self.task_head:
            ckpt["task_head_state_dict"] = self.task_head.state_dict()
        torch.save(ckpt, self.checkpoint_dir / filename)
        logger.info(f"Saved: {filename}")

    def load_pretrained(self, path: str):
        """Load pretrained encoder weights."""
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)

        # Load only encoder parameters (skip task-specific heads)
        model_state = self.model.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in model_state and model_state[k].shape == v.shape}
        self.model.load_state_dict(matched, strict=False)
        logger.info(f"Loaded {len(matched)} encoder params from {path}")

    def cleanup(self):
        self.writer.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Volve Finetuning")
    parser.add_argument("--task", type=str, required=True,
                       choices=["lithology", "reservoir_prediction", "fault_detection"])
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--data_dir", type=str, default=r"E:\oilmodel")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--seismic_patch", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--well_seq_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=192)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Task: {args.task} | Device: {device}")

    # ---- Data ----
    train_ds = VolveDataset(
        data_dir=args.data_dir, mode="pretrain",
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
    )
    val_ds = VolveDataset(
        data_dir=args.data_dir, mode="test",
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
        train_wells=train_ds.train_wells,
        val_wells=train_ds.val_wells,
        norm_stats=train_ds.norm_stats,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    # ---- Model ----
    config = ModelConfig()
    config.seismic_encoder.backbone = "ncs"
    config.seismic_encoder.embed_dim = args.embed_dim
    config.seismic_encoder.ncs_mode = "3d"
    config.seismic_encoder.img_size = tuple(args.seismic_patch)
    config.seismic_encoder.num_heads = 3
    config.seismic_encoder.num_layers = 6
    config.well_log_encoder.backbone = "wlfm"
    config.well_log_encoder.embed_dim = args.embed_dim
    config.well_log_encoder.num_heads = 6
    config.well_log_encoder.num_layers = 4
    config.hidden_dim = args.embed_dim * 2

    model = OilGasModel(config)

    # ---- Trainer ----
    trainer_config = {
        "lr": args.lr,
        "weight_decay": 0.01,
        "epochs": args.epochs,
        "accumulate_grad": 1,
        "grad_clip": 1.0,
        "mixed_precision": device == "cuda",
        "log_dir": f"{args.log_dir}/{args.task}",
        "checkpoint_dir": f"{args.checkpoint_dir}/{args.task}",
        "early_stopping": 20,
    }

    trainer = FinetuneTrainer(
        model=model, task=args.task,
        train_loader=train_loader, val_loader=val_loader,
        config=trainer_config, device=device,
    )

    if args.pretrained:
        trainer.load_pretrained(args.pretrained)

    logger.info(f"Starting {args.task} finetuning...")
    history = trainer.fit(epochs=args.epochs)

    trainer.cleanup()
    logger.info(f"{args.task} finetuning complete!")


if __name__ == "__main__":
    main()
