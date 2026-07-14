"""
Staged Pretraining Script for Volve Dataset

Stage 1 - Single-modality (train encoders):
  MSM  Masked Seismic Modeling
  MWM  Masked Well-log Modeling (curve_mask aware)

Stage 2 - Cross-modal fusion (freeze encoders, train fusion stack):
  CMCL Cross-Modal Contrastive Learning on fused features
  SWM  Seismic-Well Matching on fused features

Usage:
    python scripts/train_pretrain_volve.py --stage1_epochs 50 --stage2_epochs 50
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

WELL_CURVES = ["GR", "SP", "CAL", "RD", "MLL", "MSFL", "NPHI", "RHOB", "DT"]


class PretrainTrainer:
    """Two-stage pretraining: encoders (stage 1) then fusion (stage 2)."""

    def __init__(
        self,
        model: OilGasModelForPretraining,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
        seismic_patch_size: tuple = (32, 32, 32),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.seismic_patch_size = seismic_patch_size

        self.use_amp = config.get("mixed_precision", True) and device == "cuda"
        self.scaler: Optional[GradScaler] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None

        self.log_dir = Path(config.get("log_dir", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.current_epoch = 0
        self.training_stage = 1
        self._msm_decoder: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Stage setup
    # ------------------------------------------------------------------

    def _ensure_msm_decoder(self) -> None:
        seis_encoder = self.model.seismic_encoder
        if not hasattr(seis_encoder, "forward_mae"):
            return
        if self._msm_decoder is not None:
            return
        d, h, w = self.seismic_patch_size
        self._msm_decoder = MAEDecoder3D(
            encoder_embed_dim=seis_encoder.embed_dim,
            decoder_embed_dim=512,
            patch_size=seis_encoder.patch_size,
            img_size=(d, h, w),
            decoder_num_heads=8,
            decoder_num_layers=8,
        ).to(self.device)

    def _stage1_parameters(self) -> List[nn.Parameter]:
        self._ensure_msm_decoder()
        params = (
            list(self.model.seismic_encoder.parameters())
            + list(self.model.well_log_encoder.parameters())
        )
        if self._msm_decoder is not None:
            params += list(self._msm_decoder.parameters())
        return params

    def _stage2_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.model.modality_proj,
            self.model.fusion_module,
            self.model.seismic_proj_head,
            self.model.well_proj_head,
            self.model.fusion_proj_head,
            self.model.matching_head,
        ]
        return [p for m in modules for p in m.parameters() if p.requires_grad]

    def _freeze_encoders(self) -> None:
        for module in (self.model.seismic_encoder, self.model.well_log_encoder):
            for param in module.parameters():
                param.requires_grad = False
        logger.info("Froze seismic_encoder and well_log_encoder for stage 2")

    def _unfreeze_encoders(self) -> None:
        for module in (self.model.seismic_encoder, self.model.well_log_encoder):
            for param in module.parameters():
                param.requires_grad = True

    def setup_stage(self, stage: int, epochs: int) -> None:
        self.training_stage = stage
        self.current_epoch = 0

        if stage == 1:
            self._unfreeze_encoders()
            trainable = self._stage1_parameters()
            stage_name = "Stage1-MSM+MWM"
        else:
            self._freeze_encoders()
            trainable = self._stage2_parameters()
            stage_name = "Stage2-Fusion+CMCL+SWM"

        n_trainable = sum(p.numel() for p in trainable)
        logger.info(f"{stage_name}: {n_trainable/1e6:.2f}M trainable params")

        self.optimizer = optim.AdamW(
            trainable,
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.05),
            betas=(0.9, 0.999),
        )

        accum = self.config.get("accumulate_grad", 2)
        steps_per_epoch = max(1, len(self.train_loader) // accum)
        warmup_epochs = self.config.get("warmup_epochs", 5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.get("lr", 1e-4),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=min(1.0, warmup_epochs / max(epochs, 1)),
        )
        self.scaler = GradScaler() if self.use_amp else None

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _parse_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        out = {
            "seismic": batch["seismic"].to(self.device),
            "well_log": batch["well_log"].to(self.device),
        }
        if "well_mask" in batch:
            out["well_mask"] = batch["well_mask"].to(self.device)
        if "curve_mask" in batch:
            out["curve_mask"] = batch["curve_mask"].to(self.device)
        if "well_value_mask" in batch:
            out["well_value_mask"] = batch["well_value_mask"].to(self.device)
        return out

    def compute_pretrain_losses(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if self.training_stage == 1:
            return self._compute_stage1_losses(batch)
        return self._compute_stage2_losses(batch)

    def _compute_stage1_losses(self, batch: Dict) -> Dict[str, torch.Tensor]:
        tensors = self._parse_batch(batch)
        seismic = tensors["seismic"]
        well_log = tensors["well_log"]
        well_mask = tensors.get("well_mask")
        curve_mask = tensors.get("curve_mask")
        value_mask = tensors.get("well_value_mask")

        msm_loss = self._compute_msm_loss(seismic)
        mwm_loss = self._compute_mwm_loss(
            well_log, curve_mask, well_mask, value_mask
        )

        weights = self.config.get("stage1_weights", {"msm": 1.0, "mwm": 1.0})
        total = weights["msm"] * msm_loss + weights["mwm"] * mwm_loss

        return {
            "total_loss": total,
            "msm_loss": msm_loss,
            "mwm_loss": mwm_loss,
            "cmcl_loss": torch.tensor(0.0, device=self.device),
            "swm_loss": torch.tensor(0.0, device=self.device),
        }

    def _compute_stage2_losses(self, batch: Dict) -> Dict[str, torch.Tensor]:
        tensors = self._parse_batch(batch)
        seismic = tensors["seismic"]
        well_log = tensors["well_log"]
        well_mask = tensors.get("well_mask")
        curve_mask = tensors.get("curve_mask")
        value_mask = tensors.get("well_value_mask")

        cmcl_loss = self._compute_fusion_cmcl_loss(
            seismic, well_log, curve_mask, well_mask, value_mask
        )
        swm_loss = self._compute_fusion_swm_loss(
            seismic, well_log, curve_mask, well_mask, value_mask
        )

        weights = self.config.get("stage2_weights", {"cmcl": 0.5, "swm": 0.3})
        total = weights["cmcl"] * cmcl_loss + weights["swm"] * swm_loss

        return {
            "total_loss": total,
            "msm_loss": torch.tensor(0.0, device=self.device),
            "mwm_loss": torch.tensor(0.0, device=self.device),
            "cmcl_loss": cmcl_loss,
            "swm_loss": swm_loss,
        }

    def _compute_msm_loss(self, seismic: torch.Tensor) -> torch.Tensor:
        seis_encoder = self.model.seismic_encoder
        if hasattr(seis_encoder, "forward_mae"):
            self._ensure_msm_decoder()
            result = seis_encoder.forward_mae(
                seismic, self._msm_decoder, mask_ratio=0.6
            )
            return result["loss"]

        b = seismic.shape[0]
        with autocast(enabled=self.use_amp):
            full_feat, _ = seis_encoder(seismic, return_features=False)
            masked_input = seismic * (torch.rand_like(seismic) > 0.6)
            masked_feat, _ = seis_encoder(masked_input, return_features=False)
        return F.mse_loss(masked_feat, full_feat)

    def _compute_mwm_loss(
        self,
        well_log: torch.Tensor,
        curve_mask: Optional[torch.Tensor],
        well_mask: Optional[torch.Tensor],
        value_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        wl_encoder = self.model.well_log_encoder
        if hasattr(wl_encoder, "forward_mtm"):
            with autocast(enabled=self.use_amp):
                result = wl_encoder.forward_mtm(
                    well_log,
                    mask_ratio=0.5,
                    curve_mask=curve_mask,
                    depth_mask=well_mask,
                    value_mask=value_mask,
                )
                return result["loss"]
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _encode_frozen(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
        curve_mask: Optional[torch.Tensor],
        well_mask: Optional[torch.Tensor],
        value_mask: Optional[torch.Tensor],
    ):
        """Run frozen encoders without gradients."""
        with torch.no_grad():
            seismic_global, _ = self.model.seismic_encoder(
                seismic, return_features=False
            )
            well_kwargs = {"return_sequence": False}
            if curve_mask is not None:
                well_kwargs["curve_mask"] = curve_mask
            if well_mask is not None:
                well_kwargs["mask"] = well_mask
            if value_mask is not None:
                well_kwargs["value_mask"] = value_mask
            well_global, _ = self.model.well_log_encoder(well_log, **well_kwargs)
        return seismic_global, well_global

    def _fusion_forward(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
        curve_mask: Optional[torch.Tensor],
        well_mask: Optional[torch.Tensor],
        value_mask: Optional[torch.Tensor],
    ):
        seis_global, well_global = self._encode_frozen(
            seismic, well_log, curve_mask, well_mask, value_mask
        )
        seis_proj, well_proj = self.model.modality_proj(seis_global, well_global)
        fused = self.model.fusion_module(seis_proj, well_proj)
        return seis_proj, well_proj, fused

    @staticmethod
    def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07):
        logits = (z_a @ z_b.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def _compute_fusion_cmcl_loss(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
        curve_mask: Optional[torch.Tensor],
        well_mask: Optional[torch.Tensor],
        value_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with autocast(enabled=self.use_amp):
            seis_proj, well_proj, fused = self._fusion_forward(
                seismic, well_log, curve_mask, well_mask, value_mask
            )
            z_fused = F.normalize(self.model.fusion_proj_head(fused), dim=-1)
            z_seis = F.normalize(self.model.seismic_proj_head(seis_proj), dim=-1)
            z_well = F.normalize(self.model.well_proj_head(well_proj), dim=-1)

            loss_fw = self._info_nce(z_fused, z_well)
            loss_fs = self._info_nce(z_fused, z_seis)
            return (loss_fw + loss_fs) / 2

    def _compute_fusion_swm_loss(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
        curve_mask: Optional[torch.Tensor],
        well_mask: Optional[torch.Tensor],
        value_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b = seismic.shape[0]
        with autocast(enabled=self.use_amp):
            _, well_proj, fused = self._fusion_forward(
                seismic, well_log, curve_mask, well_mask, value_mask
            )
            pos_pairs = torch.cat([fused, well_proj], dim=-1)
            shuffle_idx = torch.randperm(b, device=self.device)
            neg_pairs = torch.cat([fused, well_proj[shuffle_idx]], dim=-1)
            pos_logits = self.model.matching_head(pos_pairs)
            neg_logits = self.model.matching_head(neg_pairs)

        # Use logits + BCEWithLogits: AMP fp16 Sigmoid outputs can leave [0, 1]
        # and crash F.binary_cross_entropy with a device-side assert.
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits.float(),
            torch.ones(pos_logits.shape, device=pos_logits.device, dtype=torch.float32),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits.float(),
            torch.zeros(neg_logits.shape, device=neg_logits.device, dtype=torch.float32),
        )
        return (pos_loss + neg_loss) / 2

    # ------------------------------------------------------------------
    # Train / validate loops
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        if self.training_stage == 2:
            self.model.seismic_encoder.eval()
            self.model.well_log_encoder.eval()

        epoch_metrics: Dict[str, float] = {}
        num_batches = len(self.train_loader)
        accum = self.config.get("accumulate_grad", 2)

        for batch_idx, batch in enumerate(self.train_loader):
            losses = self.compute_pretrain_losses(batch)
            loss = losses["total_loss"] / accum

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum == 0:
                grad_clip = self.config.get("grad_clip", 1.0)
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._stage1_parameters() if self.training_stage == 1 else self._stage2_parameters(),
                    grad_clip,
                )
                if self.use_amp and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            for k, v in losses.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v.item()

            if batch_idx % self.config.get("log_interval", 10) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                if self.training_stage == 1:
                    logger.info(
                        f"S{self.training_stage} Ep{self.current_epoch} "
                        f"Batch {batch_idx}/{num_batches} | "
                        f"Total: {loss.item()*accum:.4f} | "
                        f"MSM: {losses['msm_loss'].item():.4f} | "
                        f"MWM: {losses['mwm_loss'].item():.4f} | LR: {lr:.6f}"
                    )
                else:
                    logger.info(
                        f"S{self.training_stage} Ep{self.current_epoch} "
                        f"Batch {batch_idx}/{num_batches} | "
                        f"Total: {loss.item()*accum:.4f} | "
                        f"CMCL: {losses['cmcl_loss'].item():.4f} | "
                        f"SWM: {losses['swm_loss'].item():.4f} | LR: {lr:.6f}"
                    )
                tag = f"stage{self.training_stage}"
                for k, v in losses.items():
                    self.writer.add_scalar(f"{tag}/{k}", v.item(), self.global_step)

            self.global_step += 1

        return {k: v / num_batches for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        metrics: Dict[str, float] = {}
        for batch in self.val_loader:
            losses = self.compute_pretrain_losses(batch)
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0.0) + v.item()
        n = len(self.val_loader)
        return {k: v / n for k, v in metrics.items()}

    def fit_stage(self, epochs: int, stage: int) -> Dict:
        self.setup_stage(stage, epochs)
        best_val = float("inf")
        patience = self.config.get("early_stopping", 15)
        no_improve = 0
        history: Dict = {"train": [], "val": []}

        for epoch in range(epochs):
            self.current_epoch = epoch
            t0 = time.time()
            train_metrics = self.train_epoch()
            train_metrics["time"] = time.time() - t0
            history["train"].append(train_metrics)

            logger.info(
                f"Stage {stage} Epoch {epoch} | "
                f"Total: {train_metrics.get('total_loss', 0):.4f} | "
                f"Time: {train_metrics['time']:.1f}s"
            )

            val_metrics = self.validate()
            if val_metrics:
                history["val"].append(val_metrics)
                val_loss = val_metrics.get("total_loss", 0)
                if stage == 1:
                    logger.info(
                        f"  Val Total: {val_loss:.4f} | "
                        f"MSM: {val_metrics.get('msm_loss', 0):.4f} | "
                        f"MWM: {val_metrics.get('mwm_loss', 0):.4f}"
                    )
                else:
                    logger.info(
                        f"  Val Total: {val_loss:.4f} | "
                        f"CMCL: {val_metrics.get('cmcl_loss', 0):.4f} | "
                        f"SWM: {val_metrics.get('swm_loss', 0):.4f}"
                    )
                for k, v in val_metrics.items():
                    self.writer.add_scalar(
                        f"stage{stage}_val/{k}", v, self.global_step
                    )
                ckpt_name = "best_stage1.pt" if stage == 1 else "best_stage2.pt"
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    self.save_checkpoint(ckpt_name)
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping stage {stage} at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                self.save_checkpoint(f"stage{stage}_epoch_{epoch}.pt")

        final_name = "stage1_final.pt" if stage == 1 else "stage2_final.pt"
        self.save_checkpoint(final_name)
        return history

    def fit_staged(self, stage1_epochs: int, stage2_epochs: int) -> Dict:
        logger.info("=" * 60)
        logger.info(f"Stage 1: MSM + MWM ({stage1_epochs} epochs)")
        logger.info("=" * 60)
        hist1 = self.fit_stage(stage1_epochs, stage=1)

        logger.info("=" * 60)
        logger.info(f"Stage 2: Fusion + CMCL + SWM ({stage2_epochs} epochs, encoders frozen)")
        logger.info("=" * 60)
        hist2 = self.fit_stage(stage2_epochs, stage=2)

        self.save_checkpoint("pretrain_final.pt")
        return {"stage1": hist1, "stage2": hist2}

    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        state = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "training_stage": self.training_stage,
            "config": self.config,
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self._msm_decoder is not None:
            state["msm_decoder_state_dict"] = self._msm_decoder.state_dict()
        torch.save(state, path)
        logger.info(f"Saved: {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.training_stage = ckpt.get("training_stage", 1)
        if self._msm_decoder is None and "msm_decoder_state_dict" in ckpt:
            self._ensure_msm_decoder()
            if self._msm_decoder is not None:
                self._msm_decoder.load_state_dict(ckpt["msm_decoder_state_dict"])
        if load_optimizer and self.optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint: {path}")

    def cleanup(self):
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Volve Staged Pretraining")
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).parent.parent))
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_epochs", type=int, default=50)
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (model weights)")
    parser.add_argument("--stage2_from", type=str, default=None,
                        help="Load stage1 checkpoint and run stage2 only")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Load NCS/WLFM pretrained weights (requires network access)",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    train_ds = VolveDataset(
        data_dir=args.data_dir,
        mode="pretrain",
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
        well_curves=WELL_CURVES,
    )
    val_ds = VolveDataset(
        data_dir=args.data_dir,
        mode="test",
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
        well_curves=WELL_CURVES,
        train_wells=train_ds.train_wells,
        val_wells=train_ds.val_wells,
        norm_stats=train_ds.norm_stats,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    config = ModelConfig()
    config.seismic_encoder.backbone = args.seismic_backbone
    config.seismic_encoder.embed_dim = args.embed_dim
    config.seismic_encoder.ncs_mode = args.ncs_mode
    config.seismic_encoder.img_size = tuple(args.seismic_patch)
    config.seismic_encoder.num_heads = 3 if args.embed_dim <= 192 else 6
    config.seismic_encoder.num_layers = 6
    config.seismic_encoder.use_pretrained = args.use_pretrained
    config.well_log_encoder.backbone = args.well_backbone
    config.well_log_encoder.num_curves = 9
    config.well_log_encoder.embed_dim = args.embed_dim
    config.well_log_encoder.num_heads = 6
    config.well_log_encoder.num_layers = 4
    config.well_log_encoder.wlfm_patch_len = 32
    config.well_log_encoder.wlfm_patch_stride = 16
    config.hidden_dim = args.embed_dim * 2

    model = OilGasModelForPretraining(config)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    trainer_config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": 5,
        "accumulate_grad": 2,
        "grad_clip": 1.0,
        "mixed_precision": device == "cuda",
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "log_interval": 10,
        "early_stopping": 15,
        "stage1_weights": {"msm": 1.0, "mwm": 1.0},
        "stage2_weights": {"cmcl": 0.5, "swm": 0.3},
    }

    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        seismic_patch_size=tuple(args.seismic_patch),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.stage2_from:
        trainer.load_checkpoint(args.stage2_from)

    if args.use_pretrained:
        logger.info("Using external pretrained NCS/WLFM weights where available")
    else:
        logger.info("Training from scratch (no external pretrained weights)")

    if args.stage2_from:
        logger.info(f"Running stage 2 only from {args.stage2_from}")
        trainer.fit_stage(args.stage2_epochs, stage=2)
    else:
        trainer.fit_staged(args.stage1_epochs, args.stage2_epochs)

    trainer.cleanup()
    logger.info("Staged pretraining complete!")


if __name__ == "__main__":
    main()
