#!/usr/bin/env python3
"""
Downstream Task Finetuning Script

Usage:
    python train_finetune.py --task fault_detection --pretrained checkpoints/pretrain/pretrained_model_final.pt
    python train_finetune.py --task reservoir_prediction --pretrained checkpoints/pretrain/pretrained_model_final.pt
    python train_finetune.py --task lithology --pretrained checkpoints/pretrain/pretrained_model_final.pt
"""

import sys
import os
import argparse
import torch
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, FinetuningConfig
from data.dataset import FinetuneDataset, create_dataloaders
from data.transforms import SeismicAugmentation, WellLogAugmentation
from models.oil_gas_model import OilGasModel
from training.trainer import Trainer
from training.losses import MultiTaskLoss, DiceLoss, FocalLoss
from training.metrics import SegmentationMetrics, ClassificationMetrics
from utils.helpers import set_seed, count_parameters, get_device, create_experiment_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Oil & Gas Model")
    parser.add_argument("--task", type=str, required=True,
                        choices=["fault_detection", "reservoir_prediction", "lithology"],
                        help="Downstream task to finetune on")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of finetuning epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (smaller for 3D data)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=5,
                        help="Epochs with frozen encoder")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/finetune",
                        help="Checkpoint directory")
    parser.add_argument("--use_synthetic", action="store_true", default=True,
                        help="Use synthetic data")
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
    print(f"Task: {args.task}, Device: {device}")

    # Model
    config_path = Path(args.config)
    if config_path.exists():
        model_config = ModelConfig.from_yaml(str(config_path))
    else:
        model_config = ModelConfig()

    model = OilGasModel(model_config)

    # Load pretrained weights
    if os.path.exists(args.pretrained):
        checkpoint = torch.load(args.pretrained, map_location=device)
        # Load only encoder weights (partial loading)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and "encoder" in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
        print(f"  Matched {len(pretrained_dict)} encoder parameters")
    else:
        print(f"Warning: Pretrained model not found at {args.pretrained}")

    # Data
    seismic_aug = SeismicAugmentation()
    well_aug = WellLogAugmentation()

    train_dataset = FinetuneDataset(
        data_path=args.data_path,
        num_samples=500,
        use_synthetic=args.use_synthetic,
        task=args.task,
        seismic_aug=seismic_aug,
        well_aug=well_aug,
        seed=42,
    )
    val_dataset = FinetuneDataset(
        data_path=args.data_path,
        num_samples=100,
        use_synthetic=args.use_synthetic,
        task=args.task,
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

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer with layer-wise LR decay
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "encoder" in name or "fusion" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
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
        checkpoint_dir=str(exp_dir),
        use_wandb=args.wandb,
    )

    # Stage 1: Warm-up with frozen encoder
    if args.freeze_encoder_epochs > 0:
        print(f"\n--- Freezing encoder for {args.freeze_encoder_epochs} epochs ---")
        model.freeze_encoders()
        history_warmup = trainer.fit(
            num_epochs=args.freeze_encoder_epochs,
            task=args.task,
            early_stopping_patience=5,
        )

    # Stage 2: Full finetuning
    print(f"\n--- Full finetuning for remaining epochs ---")
    model.unfreeze_encoders()
    history_finetune = trainer.fit(
        num_epochs=args.epochs - args.freeze_encoder_epochs,
        task=args.task,
        early_stopping_patience=10,
    )

    # Save final model
    final_path = exp_dir / f"finetuned_{args.task}.pt"
    trainer.save_checkpoint(str(final_path))
    print(f"\nFinetuned model saved to {final_path}")

    trainer.cleanup()


if __name__ == "__main__":
    main()
