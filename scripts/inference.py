#!/usr/bin/env python3
"""
Model Inference Script

Usage:
    python inference.py --checkpoint checkpoints/finetune/best_model_fault_detection.pt \
                        --task fault_detection --input data/sample.h5 --output results/
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from models.oil_gas_model import OilGasModel
from data.synthetic_data import SyntheticDataGenerator
from utils.helpers import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Oil & Gas Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--task", type=str, required=True,
                        choices=["fault_detection", "reservoir_prediction", "lithology"],
                        help="Task to run")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input data (HDF5 or use 'synthetic')")
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> OilGasModel:
    """Load model from checkpoint."""
    config_path = Path(config_path)
    if config_path.exists():
        model_config = ModelConfig.from_yaml(str(config_path))
    else:
        model_config = ModelConfig()

    model = OilGasModel(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def run_fault_detection(
    model: OilGasModel,
    seismic: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run fault detection inference."""
    seismic = seismic.to(device)

    # Sliding window for large volumes
    if seismic.shape[-1] > 256:
        # Split into overlapping windows
        outputs = []
        window_size = 256
        overlap = 32
        for w_start in range(0, seismic.shape[-1], window_size - overlap):
            w_end = min(w_start + window_size, seismic.shape[-1])
            patch = seismic[..., w_start:w_end]
            out = model(patch, task="fault_detection")
            outputs.append(out["fault_prob"].cpu().numpy())

        # Merge with overlap weighting
        result = np.concatenate(outputs, axis=-1)
    else:
        out = model(seismic, task="fault_detection")
        result = out["fault_prob"].cpu().numpy()

    return result


@torch.no_grad()
def run_reservoir_prediction(
    model: OilGasModel,
    seismic: torch.Tensor,
    well_log: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run reservoir prediction inference."""
    seismic = seismic.to(device)
    if well_log is not None:
        well_log = well_log.to(device)

    out = model(seismic, well_log=well_log, task="reservoir_prediction")

    results = {}
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            results[key] = val.cpu().numpy()

    return results


@torch.no_grad()
def run_lithology_classification(
    model: OilGasModel,
    well_log: torch.Tensor,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run lithology classification inference."""
    well_log = well_log.to(device)

    out = model(well_log, task="lithology")

    results = {}
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            results[key] = val.cpu().numpy()

    return results


def main():
    args = parse_args()
    set_seed(42)
    device = get_device(args.device)

    # Load model
    model = load_model(args.checkpoint, args.config, device)
    print(f"Model loaded from {args.checkpoint}")

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate input data
    if args.input == "synthetic":
        generator = SyntheticDataGenerator(seed=42)
        sample = generator.generate_well_seismic_pair()

        seismic = torch.from_numpy(sample["seismic"]).float().unsqueeze(0)  # (1, 1, D, H, W)
        well_log = torch.from_numpy(sample["well_log"]).float().unsqueeze(0)  # (1, L, C)

        print(f"Using synthetic data: seismic {list(seismic.shape)}, well_log {list(well_log.shape)}")
    else:
        import h5py
        with h5py.File(args.input, "r") as f:
            seismic = torch.from_numpy(f["seismic"][:]).float().unsqueeze(0)
            well_log = torch.from_numpy(f["well_log"][:]).float().unsqueeze(0) if "well_log" in f else None

    # Run inference
    print(f"Running inference for task: {args.task}")

    if args.task == "fault_detection":
        result = run_fault_detection(model, seismic, device)
        np.save(output_dir / "fault_prob.npy", result)
        print(f"Fault detection results saved to {output_dir / 'fault_prob.npy'}")

    elif args.task == "reservoir_prediction":
        results = run_reservoir_prediction(model, seismic, well_log, device)
        for key, val in results.items():
            np.save(output_dir / f"{key}.npy", val)
            print(f"Saved {key}: {val.shape}")

    elif args.task == "lithology":
        results = run_lithology_classification(model, well_log, device)
        for key, val in results.items():
            np.save(output_dir / f"{key}.npy", val)
            print(f"Saved {key}: {val.shape}")

    print("Inference complete!")


if __name__ == "__main__":
    main()
