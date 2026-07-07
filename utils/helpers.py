"""
Utility Helper Functions

Common utilities for model training, evaluation, and debugging.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device_str: Optional device string ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        torch.device
    """
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def create_experiment_dir(base_dir: str = "./experiments") -> Path:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory for experiments

    Returns:
        Path to the new experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_model_summary(
    model: nn.Module,
    filepath: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
):
    """
    Save a model architecture summary to a file.

    Args:
        model: PyTorch model
        filepath: Output file path
        input_shapes: Optional dict of input shapes for parameter counting
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Model Architecture Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total Parameters: {count_parameters(model, trainable_only=False):,}\n")
        f.write(f"Trainable Parameters: {count_parameters(model, trainable_only=True):,}\n\n")

        f.write("=" * 60 + "\n")
        f.write("Module Structure\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(model))
        f.write("\n")

        if input_shapes:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Expected Input Shapes\n")
            f.write("=" * 60 + "\n\n")
            for name, shape in input_shapes.items():
                f.write(f"  {name}: {shape}\n")


class AverageMeter:
    """Keeps track of average values over time."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping handler.

    Monitors a metric and stops training when it stops improving.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

    def __call__(self, score: float) -> bool:
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    Helps stabilize training and improve final model quality.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].data

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
