"""
Training Module

Flexible trainer class supporting:
- Self-supervised pretraining (4 tasks)
- Supervised finetuning (3 downstream tasks)
- Mixed precision training (AMP)
- Gradient accumulation
- Model checkpointing
- TensorBoard / WandB logging
- Early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Callable
import time
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified trainer for pretraining and finetuning.

    Supports:
    - Multi-GPU training (DataParallel / DDP)
    - Mixed precision (AMP)
    - Gradient accumulation
    - Logging (TensorBoard, WandB)
    - Checkpointing and resumption
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        log_interval: int = 50,
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        seed: int = 42,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.seed = seed

        # Setup
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = self.model.to(device)
        self.scaler = GradScaler() if mixed_precision else None
        self.global_step = 0
        self.current_epoch = 0

        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Logging
        if use_wandb:
            import wandb
            self.wandb = wandb
        else:
            self.wandb = None

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "logs"))
        else:
            self.writer = None

    def train_step(
        self, batch: Dict[str, torch.Tensor], task: str = "pretrain"
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            batch: Input batch dict
            task: Task name

        Returns:
            loss dict with at least 'total_loss' key
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast(enabled=self.mixed_precision):
            if task == "pretrain":
                outputs = self.model(
                    seismic=batch["seismic"],
                    well_log=batch.get("well_log"),
                    task="fusion",
                )
                loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            elif task == "fault_detection":
                outputs = self.model(
                    seismic=batch["seismic"],
                    well_log=batch.get("well_log"),
                    task="fault_detection",
                )
                loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            elif task == "reservoir_prediction":
                outputs = self.model(
                    seismic=batch["seismic"],
                    well_log=batch.get("well_log"),
                    task="reservoir_prediction",
                )
                loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            elif task == "lithology":
                outputs = self.model(
                    seismic=batch["seismic"],
                    well_log=batch.get("well_log"),
                    task="lithology",
                )
                loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            else:
                outputs = self.model(**batch)
                loss = outputs.get("loss", outputs)

        if isinstance(loss, dict):
            loss_dict = loss
            loss = loss_dict.get("total_loss", sum(loss_dict.values()))
        else:
            loss_dict = {"total_loss": loss}

        return {"loss": loss, "outputs": outputs, **loss_dict}

    def train_epoch(self, task: str = "pretrain") -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            step_result = self.train_step(batch, task)

            loss = step_result["loss"]

            # Gradient accumulation
            loss = loss / self.accumulate_grad_batches

            if self.mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    if self.mixed_precision and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                # Optimizer step
                if self.mixed_precision and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            for key, val in step_result.items():
                if isinstance(val, torch.Tensor):
                    epoch_metrics[key] += val.item()

            if batch_idx % self.log_interval == 0:
                logger.info(
                    f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(self.train_loader)} "
                    f"| Loss: {loss.item() * self.accumulate_grad_batches:.4f}"
                )

                if self.writer:
                    self.writer.add_scalar(
                        f"train/{task}_loss_step",
                        loss.item() * self.accumulate_grad_batches,
                        self.global_step,
                    )
                if self.wandb:
                    self.wandb.log({
                        f"train/{task}_loss_step": loss.item() * self.accumulate_grad_batches,
                        "step": self.global_step,
                    })

            self.global_step += 1

        # Average metrics
        num_batches = len(self.train_loader)
        avg_metrics = {
            key: val / num_batches for key, val in epoch_metrics.items()
        }
        avg_metrics["time"] = time.time() - start_time

        return avg_metrics

    @torch.no_grad()
    def validate(self, task: str = "pretrain") -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_metrics = defaultdict(float)

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if task == "pretrain":
                outputs = self.model(
                    seismic=batch["seismic"],
                    well_log=batch.get("well_log"),
                    task="fusion",
                )
            else:
                outputs = self.model(
                    seismic=batch["seismic"],
                    well_log=batch.get("well_log"),
                    task=task,
                )

            for key, val in outputs.items():
                if isinstance(val, torch.Tensor):
                    val_metrics[key] += val.item()

        num_batches = len(self.val_loader)
        return {key: val / num_batches for key, val in val_metrics.items()}

    def fit(
        self,
        num_epochs: int,
        task: str = "pretrain",
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            task: Task name
            early_stopping_patience: Patience for early stopping
            save_best: Save best model checkpoint

        Returns:
            training history dict
        """
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train": [], "val": []}

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(task)
            history["train"].append(train_metrics)

            logger.info(
                f"Epoch {epoch} | Train Loss: {train_metrics.get('total_loss', 0):.4f}"
            )

            # Validate
            val_metrics = self.validate(task)
            if val_metrics:
                history["val"].append(val_metrics)
                val_loss = val_metrics.get("total_loss", val_metrics.get("loss", 0))
                logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

                # Log
                if self.writer:
                    for key, val in val_metrics.items():
                        self.writer.add_scalar(f"val/{key}", val, epoch)
                if self.wandb:
                    self.wandb.log({f"val/{key}": val for key, val in val_metrics.items()})

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    if save_best:
                        self.save_checkpoint(f"best_model_{task}.pt")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_metrics else train_metrics.get("total_loss", 0))
                else:
                    self.scheduler.step()

            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Checkpoint loaded from {path}")

    def cleanup(self):
        """Clean up logging resources."""
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()
