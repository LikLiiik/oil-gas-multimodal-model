#!/usr/bin/env python3
"""
Self-Supervised Pretraining Script

Usage:
    python train_pretrain.py --config config/config.yaml --epochs 100

Three-stage pretraining:
    1. Masked Seismic Modeling (MSM)
    2. Masked Well-log Modeling (MWM)
    3. Cross-Modal Contrastive Learning (CMCL) + Seismic-Well Matching (SWM)
"""

import sys
import os
import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import (
    ModelConfig, PretrainingConfig, DataConfig, TrainingConfig
)
from data.dataset import PretrainDataset, create_dataloaders
from data.transforms import SeismicAugmentation, WellLogAugmentation
from models.oil_gas_model import OilGasModelForPretraining
from training.trainer import Trainer
from training.losses import PretrainingLoss
from utils.helpers import set_seed, count_parameters, get_device, create_experiment_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Oil & Gas Multi-modal Model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/pretrain",
                        help="Checkpoint directory")
    parser.add_argument("--use_synthetic", action="store_true", default=True,
                        help="Use synthetic data for testing")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to real data (HDF5)")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable WandB logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    set_seed(42)
    device = get_device(args.device)
    exp_dir = create_experiment_dir(args.checkpoint_dir)

    print(f"Device: {device}")
    print(f"Experiment dir: {exp_dir}")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        model_config = ModelConfig.from_yaml(str(config_path))
    else:
        model_config = ModelConfig()

    # Data augmentations
    seismic_aug = SeismicAugmentation()
    well_aug = WellLogAugmentation()

    # Datasets
    train_dataset = PretrainDataset(
        data_path=args.data_path,
        num_samples=1000,
        seismic_shape=model_config.seismic_encoder.patch_size,
        use_synthetic=args.use_synthetic,
        seismic_aug=seismic_aug,
        well_aug=well_aug,
        seed=42,
    )
    val_dataset = PretrainDataset(
        data_path=args.data_path,
        num_samples=200,
        use_synthetic=args.use_synthetic,
        seed=123,
    )

    train_loader = create_dataloaders(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
    )
    val_loader = create_dataloaders(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = OilGasModelForPretraining(model_config)
    n_params = count_parameters(model, trainable_only=True)
    print(f"Trainable parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Loss
    pretrain_loss = PretrainingLoss(
        msm_weight=1.0,
        mwm_weight=1.0,
        cmcl_weight=0.5,
        swm_weight=0.3,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        log_interval=50,
        checkpoint_dir=str(exp_dir),
        use_wandb=args.wandb,
    )

    # Stage 1: MSM + MWM (modality-specific reconstruction)
    print("\n" + "=" * 60)
    print("Stage 1: Masked Seismic & Well-log Modeling")
    print("=" * 60)

    # Train MSM and MWM separately or jointly
    history_stage1 = trainer.fit(
        num_epochs=args.epochs // 3,
        task="pretrain",
        early_stopping_patience=15,
    )

    # Stage 2: Cross-modal contrastive + matching
    print("\n" + "=" * 60)
    print("Stage 2: Cross-Modal Contrastive Learning & Matching")
    print("=" * 60)

    history_stage2 = trainer.fit(
        num_epochs=args.epochs // 3,
        task="pretrain",
        early_stopping_patience=15,
    )

    # Stage 3: Joint pretraining
    print("\n" + "=" * 60)
    print("Stage 3: Joint Multi-task Pretraining")
    print("=" * 60)

    history_stage3 = trainer.fit(
        num_epochs=args.epochs - 2 * (args.epochs // 3),
        task="pretrain",
        early_stopping_patience=20,
    )

    # Save final model
    final_path = exp_dir / "pretrained_model_final.pt"
    trainer.save_checkpoint(str(final_path))
    print(f"\nPretrained model saved to {final_path}")

    trainer.cleanup()


if __name__ == "__main__":
    main()
