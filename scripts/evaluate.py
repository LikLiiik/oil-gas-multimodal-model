#!/usr/bin/env python3
"""
Model Evaluation Script

Usage:
    python evaluate.py --checkpoint checkpoints/finetune/best_model_fault_detection.pt \
                       --task fault_detection --data_path data/test.h5
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from models.oil_gas_model import OilGasModel
from data.dataset import FinetuneDataset, create_dataloaders
from training.metrics import SegmentationMetrics, ClassificationMetrics, RegressionMetrics
from utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Oil & Gas Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--task", type=str, required=True,
                        choices=["fault_detection", "reservoir_prediction", "lithology",
                                 "cross_modal_retrieval"],
                        help="Task to evaluate")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to test data (HDF5)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of synthetic test samples")
    parser.add_argument("--use_synthetic", action="store_true", default=True,
                        help="Use synthetic data")
    return parser.parse_args()


@torch.no_grad()
def evaluate(args):
    set_seed(42)
    device = get_device(args.device)

    # Load model
    config_path = Path(args.config)
    if config_path.exists():
        model_config = ModelConfig.from_yaml(str(config_path))
    else:
        model_config = ModelConfig()

    model = OilGasModel(model_config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Test dataset
    test_dataset = FinetuneDataset(
        data_path=args.data_path,
        num_samples=args.num_samples,
        use_synthetic=args.use_synthetic,
        task=args.task if args.task != "cross_modal_retrieval" else "fault_detection",
        seed=999,
    )
    test_loader = create_dataloaders(
        test_dataset, batch_size=args.batch_size,
        num_workers=2, shuffle=False,
    )

    if args.task == "fault_detection":
        metrics = SegmentationMetrics(threshold=0.5)
        for batch in test_loader:
            seismic = batch["seismic"].to(device)
            target = batch["fault_mask"].to(device)

            outputs = model(seismic, task="fault_detection")
            metrics.update(outputs["fault_prob"], target)

        results = metrics.compute()
        print("\n--- Fault Detection Results ---")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    elif args.task == "reservoir_prediction":
        seg_metrics = SegmentationMetrics(threshold=0.5)
        reg_metrics = RegressionMetrics()

        for batch in test_loader:
            seismic = batch["seismic"].to(device)
            well_log = batch["well_log"].to(device)
            target_mask = batch["reservoir_mask"].to(device)

            outputs = model(seismic, well_log=well_log, task="reservoir_prediction")
            seg_metrics.update(outputs["reservoir_prob"], target_mask)

        results = seg_metrics.compute()
        print("\n--- Reservoir Prediction Results ---")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    elif args.task == "lithology":
        metrics = ClassificationMetrics(num_classes=4)
        for batch in test_loader:
            well_log = batch["well_log"].to(device)
            target = batch["lithology"].to(device)

            outputs = model(well_log=well_log, task="lithology")
            metrics.update(outputs["logits"], target)

        results = metrics.compute()
        print("\n--- Lithology Classification Results ---")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    elif args.task == "cross_modal_retrieval":
        from training.metrics import RetrievalMetrics

        seismic_embeds_list = []
        well_embeds_list = []

        for batch in test_loader:
            seismic = batch["seismic"].to(device)
            well_log = batch["well_log"].to(device)

            features = model(seismic, well_log=well_log, return_features=True)
            seismic_embeds_list.append(features["seismic_feat"].cpu())
            well_embeds_list.append(features["well_feat"].cpu())

        seismic_embeds = torch.cat(seismic_embeds_list, dim=0)
        well_embeds = torch.cat(well_embeds_list, dim=0)

        # Normalize
        seismic_embeds = torch.nn.functional.normalize(seismic_embeds, dim=-1)
        well_embeds = torch.nn.functional.normalize(well_embeds, dim=-1)

        results = RetrievalMetrics.compute(seismic_embeds, well_embeds)
        print("\n--- Cross-Modal Retrieval Results ---")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    return results


def main():
    args = parse_args()
    results = evaluate(args)
    return results


if __name__ == "__main__":
    main()
