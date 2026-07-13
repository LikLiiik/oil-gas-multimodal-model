"""
Staged pretraining on Volve + RMOTC + Penobscot (verified geometry only).

Stage 1: MSM + MWM (encoder pretraining)
Stage 2: CMCL + SWM (fusion pretraining)

Usage:
    python scripts/train_pretrain_multi.py --stage1_epochs 50 --stage2_epochs 50
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from models.oil_gas_model import OilGasModelForPretraining
from data.multimodal_dataset import CombinedMultimodalDataset, DEFAULT_FIELDS
from scripts.train_pretrain_volve import PretrainTrainer, WELL_CURVES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multi-field Staged Pretraining")
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).parent.parent),
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help="Fields to include: volve rmotc penobscot",
    )
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/pretrain_multi",
    )
    parser.add_argument("--log_dir", type=str, default="./logs/pretrain_multi")
    parser.add_argument("--seismic_patch", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--well_seq_len", type=int, default=128)
    parser.add_argument("--seismic_backbone", type=str, default="ncs")
    parser.add_argument("--well_backbone", type=str, default="wlfm")
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--ncs_mode", type=str, default="3d")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stage2_from", type=str, default=None)
    parser.add_argument("--use-pretrained", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Fields: {args.fields}")

    train_ds, val_ds = CombinedMultimodalDataset.build_train_val(
        project_root=args.project_root,
        fields=args.fields,
        seismic_patch_size=tuple(args.seismic_patch),
        well_seq_len=args.well_seq_len,
        well_curves=WELL_CURVES,
        require_verified_geometry=True,
    )
    logger.info(train_ds.summary())
    logger.info(val_ds.summary())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False
    )

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
    logger.info("Multi-field staged pretraining complete!")


if __name__ == "__main__":
    main()
