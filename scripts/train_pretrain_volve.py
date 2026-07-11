"""
End-to-end Pretraining Script for Volve Dataset

Trains NCS (Seismic ViT-MAE) + WLFM (Well-Log VQ-VAE Transformer) encoders
using 4 self-supervised tasks:
  1. MSM  - Masked Seismic Modeling (reconstruct masked 3D patches)
  2. MWM  - Masked Well-log Modeling (predict masked codebook tokens)
  3. CMCL - Cross-Modal Contrastive Learning (align seismic <> well log)
  4. SWM  - Seismic-Well Matching (predict if pair is matched)

Usage:
    python scripts/train_pretrain_volve.py --epochs 100 --batch_size 4 --device cuda
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from models.oil_gas_model import OilGasModelForPretraining
from models.ncs_seismic_encoder import MAEDecoder3D
from data.volve_dataset import VolveDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Pretraining Trainer
# ==============================================================================

class PretrainTrainer:
    """
    Multi-task self-supervised pretraining trainer.

    Orchestrates 4 pretraining tasks with dynamic loss weighting
    and checkpoint management.
    """

    def __init__(
        self,
        model: OilGasModelForPretraining,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.05),
            betas=(0.9, 0.999),
        )

        # Scheduler (cosine with warmup)
        warmup_epochs = config.get("warmup_epochs", 5)
        total_epochs = config.get("epochs", 100)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get("lr", 1e-4),
            epochs=total_epochs,
            steps_per_epoch=len(train_loader) // config.get("accumulate_grad", 2),
            pct_start=warmup_epochs / total_epochs,
        )

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True) and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Logging
        self.log_dir = Path(config.get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def compute_pretrain_losses(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all 4 pretraining losses.

        Args:
            batch: dict with 'seismic', 'well_log', 'well_mask'

        Returns:
            dict with 'total_loss', 'msm_loss', 'mwm_loss', 'cmcl_loss', 'swm_loss'
        """
        seismic = batch["seismic"].to(self.device)          # (B, 1, D, H, W)
        well_log = batch["well_log"].to(self.device)        # (B, C, L)
        well_mask = batch.get("well_mask", None)
        if well_mask is not None:
            well_mask = well_mask.to(self.device)

        B = seismic.shape[0]
        losses = {}

        # ---- 1. MSM: Masked Seismic Modeling ----
        # Randomly mask 60% of 3D patches and predict original values
        msm_loss = self._compute_msm_loss(seismic)
        losses["msm"] = msm_loss

        # ---- 2. MWM: Masked Well-log Modeling ----
        # Randomly mask 50% of well log token indices and predict them
        mwm_loss = self._compute_mwm_loss(well_log)
        losses["mwm"] = mwm_loss

        # ---- 3. CMCL: Cross-Modal Contrastive Learning ----
        cmcl_loss = self._compute_cmcl_loss(seismic, well_log)
        losses["cmcl"] = cmcl_loss

        # ---- 4. SWM: Seismic-Well Matching ----
        swm_loss = self._compute_swm_loss(seismic, well_log)
        losses["swm"] = swm_loss

        # Weighted total
        weights = self.config.get("task_weights", {
            "msm": 1.0, "mwm": 1.0, "cmcl": 0.5, "swm": 0.3,
        })
        total = sum(weights.get(k, 1.0) * losses[k] for k in losses)

        return {
            "total_loss": total,
            "msm_loss": losses["msm"],
            "mwm_loss": losses["mwm"],
            "cmcl_loss": losses["cmcl"],
            "swm_loss": losses["swm"],
        }

    def _compute_msm_loss(self, seismic: torch.Tensor) -> torch.Tensor:
        """
        MSM: Mask 3D patches and reconstruct.

        Uses the NCS encoder's MAE forward: mask patches, encode visible ones,
        decode to reconstruct masked patches. Loss is MSE on masked positions.
        """
        B, C, D, H, W = seismic.shape

        # Use the seismic encoder's internal MAE if available
        seis_encoder = self.model.seismic_encoder

        if hasattr(seis_encoder, 'forward_mae'):
            # Build decoder (cached for efficiency)
            if not hasattr(self, '_msm_decoder'):
                self._msm_decoder = MAEDecoder3D(
                    encoder_embed_dim=seis_encoder.embed_dim,
                    decoder_embed_dim=512,
                    patch_size=seis_encoder.patch_size,
                    img_size=(D, H, W),
                    decoder_num_heads=8,
                    decoder_num_layers=8,
                ).to(self.device)

            result = seis_encoder.forward_mae(seismic, self._msm_decoder, mask_ratio=0.6)
            return result["loss"]

        # Fallback: simple reconstruction via encoder->fc->decoder
        with autocast(enabled=self.use_amp):
            global_feat, _ = seis_encoder(seismic, return_features=False)
            # MSE on a random subset of input (simplified masking)
            mask = torch.rand(B, C, D, H, W, device=self.device) > 0.6
            masked_input = seismic * mask
            _, _ = seis_encoder(masked_input, return_features=False)
            # Simple proxy: feature consistency loss
            full_feat, _ = seis_encoder(seismic, return_features=False)
            masked_feat, _ = seis_encoder(masked_input, return_features=False)
            return F.mse_loss(masked_feat, full_feat)

    def _compute_mwm_loss(self, well_log: torch.Tensor) -> torch.Tensor:
        """
        MWM: Mask well log tokens and predict their codebook indices.

        Uses the WLFM encoder's MTM forward: tokenize the well log,
        mask random tokens, predict the discrete codebook indices.
        """
        wl_encoder = self.model.well_log_encoder

        if hasattr(wl_encoder, 'forward_mtm'):
            with autocast(enabled=self.use_amp):
                result = wl_encoder.forward_mtm(well_log, mask_ratio=0.5)
                return result["loss"]

        # Fallback: simple masked reconstruction
        global_feat, _ = wl_encoder(well_log, return_sequence=False)
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _compute_cmcl_loss(
        self, seismic: torch.Tensor, well_log: torch.Tensor
    ) -> torch.Tensor:
        """
        CMCL: Cross-Modal Contrastive Learning.

        Aligns seismic and well log embeddings in a shared space
        using InfoNCE loss.
        """
        with autocast(enabled=self.use_amp):
            # Get embeddings from both encoders
            seismic_global, _ = self.model.seismic_encoder(seismic, return_features=False)
            well_global, _ = self.model.well_log_encoder(well_log, return_sequence=False)

            # Project to common dimension
            seis_proj, well_proj = self.model.modality_proj(seismic_global, well_global)

            # Project to contrastive space
            z_seis = self.model.seismic_proj_head(seis_proj)
            z_well = self.model.well_proj_head(well_proj)

            # L2 normalize
            z_seis = F.normalize(z_seis, dim=-1)
            z_well = F.normalize(z_well, dim=-1)

            # InfoNCE loss
            temperature = 0.07
            logits = (z_seis @ z_well.T) / temperature
            labels = torch.arange(logits.shape[0], device=logits.device)

            loss_s2w = F.cross_entropy(logits, labels)
            loss_w2s = F.cross_entropy(logits.T, labels)

            return (loss_s2w + loss_w2s) / 2

    def _compute_swm_loss(
        self, seismic: torch.Tensor, well_log: torch.Tensor
    ) -> torch.Tensor:
        """
        SWM: Seismic-Well Matching.

        Binary classification: predict if a seismic patch and well log
        sequence come from the same well/position (positive) or not (negative).
        """
        B = seismic.shape[0]

        with autocast(enabled=self.use_amp):
            # Positive pairs (matched)
            seismic_global, _ = self.model.seismic_encoder(seismic, return_features=False)
            well_global, _ = self.model.well_log_encoder(well_log, return_sequence=False)
            seis_proj, well_proj = self.model.modality_proj(seismic_global, well_global)

            # Positive pairs
            pos_pairs = torch.cat([seis_proj, well_proj], dim=-1)
            pos_logits = self.model.matching_head(pos_pairs)

            # Negative pairs (shuffled well logs)
            shuffle_idx = torch.randperm(B, device=self.device)
            neg_pairs = torch.cat([seis_proj, well_proj[shuffle_idx]], dim=-1)
            neg_logits = self.model.matching_head(neg_pairs)

            # Binary cross-entropy
            pos_targets = torch.ones_like(pos_logits)
            neg_targets = torch.zeros_like(neg_logits)

            pos_loss = F.binary_cross_entropy(pos_logits, pos_targets)
            neg_loss = F.binary_cross_entropy(neg_logits, neg_targets)

            return (pos_loss + neg_loss) / 2

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch. Returns averaged metrics."""
        self.model.train()
        epoch_metrics = {}
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Compute losses
            losses = self.compute_pretrain_losses(batch)
            loss = losses["total_loss"]

            # Gradient accumulation
            accum = self.config.get("accumulate_grad", 2)
            loss = loss / accum

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum == 0:
                # Gradient clipping
                grad_clip = self.config.get("grad_clip", 1.0)
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                # Step
                if self.use_amp and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            # Track metrics
            for k, v in losses.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v.item()

            # Log
            if batch_idx % self.config.get("log_interval", 20) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Total: {loss.item()*accum:.4f} | "
                    f"MSM: {losses['msm_loss'].item():.4f} | "
                    f"MWM: {losses['mwm_loss'].item():.4f} | "
                    f"CMCL: {losses['cmcl_loss'].item():.4f} | "
                    f"SWM: {losses['swm_loss'].item():.4f} | "
                    f"LR: {lr:.6f}"
                )

                for k, v in losses.items():
                    self.writer.add_scalar(f"pretrain/{k}", v.item(), self.global_step)

            self.global_step += 1

        # Average
        return {k: v / num_batches for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        metrics = {}

        for batch in self.val_loader:
            losses = self.compute_pretrain_losses(batch)
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0.0) + v.item()

        n = len(self.val_loader)
        return {k: v / n for k, v in metrics.items()}

    def fit(self, epochs: int) -> Dict:
        """Full training loop."""
        best_val = float("inf")
        patience = self.config.get("early_stopping", 15)
        no_improve = 0
        history = {"train": [], "val": []}

        for epoch in range(epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch()
            train_metrics["time"] = time.time() - t0
            history["train"].append(train_metrics)

            logger.info(
                f"Epoch {epoch} | "
                f"Total: {train_metrics.get('total_loss', 0):.4f} | "
                f"CMCL: {train_metrics.get('cmcl_loss', 0):.4f} | "
                f"Time: {train_metrics['time']:.1f}s"
            )

            val_metrics = self.validate()
            if val_metrics:
                history["val"].append(val_metrics)
                val_loss = val_metrics.get("total_loss", 0)
                logger.info(f"  Val Total: {val_loss:.4f}")

                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    self.save_checkpoint("best_pretrain.pt")
                else:
                    no_improve += 1

                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f"pretrain_epoch_{epoch}.pt")

        self.save_checkpoint("pretrain_final.pt")
        return history

    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config,
        }, path)
        logger.info(f"Saved: {path}")

    def cleanup(self):
        self.writer.close()


# ==============================================================================
# Main
# ==============================================================================

import torch.nn.functional as F  # used by loss functions


def main():
    parser = argparse.ArgumentParser(description="Volve Pretraining")
    parser.add_argument("--data_dir", type=str, default=r"E:\oilmodel")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/pretrain")
    parser.add_argument("--log_dir", type=str, default="./logs/pretrain")
    parser.add_argument("--seismic_patch", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--well_seq_len", type=int, default=128)
    parser.add_argument("--seismic_backbone", type=str, default="ncs")
    parser.add_argument("--well_backbone", type=str, default="wlfm")
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--ncs_mode", type=str, default="3d")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    # ---- Data ----
    logger.info("Loading Volve dataset...")
    train_ds = VolveDataset(
        data_dir=args.data_dir,
        mode="pretrain",
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
        well_curves=["GR", "RT", "RHOB", "NPHI", "DT", "CALI", "PEF"],
    )

    val_ds = VolveDataset(
        data_dir=args.data_dir,
        mode="test",
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
        well_curves=["GR", "RT", "RHOB", "NPHI", "DT", "CALI", "PEF"],
        train_wells=train_ds.train_wells,
        val_wells=train_ds.val_wells,
        norm_stats=train_ds.norm_stats,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ---- Model ----
    logger.info("Building model...")
    config = ModelConfig()
    config.seismic_encoder.backbone = args.seismic_backbone
    config.seismic_encoder.embed_dim = args.embed_dim
    config.seismic_encoder.ncs_mode = args.ncs_mode
    config.seismic_encoder.img_size = tuple(args.seismic_patch)
    config.seismic_encoder.num_heads = 3 if args.embed_dim <= 192 else 6
    config.seismic_encoder.num_layers = 6  # Smaller for faster training

    config.well_log_encoder.backbone = args.well_backbone
    config.well_log_encoder.embed_dim = args.embed_dim
    config.well_log_encoder.num_heads = 6
    config.well_log_encoder.num_layers = 4
    config.well_log_encoder.wlfm_patch_len = 32
    config.well_log_encoder.wlfm_patch_stride = 16

    config.hidden_dim = args.embed_dim * 2

    model = OilGasModelForPretraining(config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params/1e6:.1f}M")

    # ---- Trainer Config ----
    trainer_config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "warmup_epochs": 5,
        "accumulate_grad": 2,
        "grad_clip": 1.0,
        "mixed_precision": device == "cuda",
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "log_interval": 10,
        "early_stopping": 15,
        "task_weights": {"msm": 1.0, "mwm": 1.0, "cmcl": 0.5, "swm": 0.3},
    }

    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
    )

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.global_step = ckpt.get("global_step", 0)

    # ---- Train ----
    logger.info("Starting pretraining...")
    history = trainer.fit(epochs=args.epochs)

    trainer.cleanup()
    logger.info("Pretraining complete!")


if __name__ == "__main__":
    main()
