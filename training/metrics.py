"""
Evaluation Metrics

Comprehensive metrics for oil & gas geophysical tasks:
- Segmentation metrics: IoU, Dice, Precision, Recall, F1
- Regression metrics: MSE, MAE, R², Pearson correlation
- Classification metrics: Accuracy, Balanced Accuracy, F1-macro
- Cross-modal metrics: Retrieval accuracy, alignment score
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix,
)


class SegmentationMetrics:
    """Metrics for 3D segmentation tasks (fault detection, reservoir prediction)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.dice_sum = 0.0
        self.count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics with batch predictions.

        Args:
            pred: (B, 1, D, H, W) predicted probabilities
            target: (B, 1, D, H, W) binary ground truth
            mask: Optional valid region mask
        """
        pred_binary = (pred > self.threshold).float()

        if mask is not None:
            pred_binary = pred_binary * mask
            target = target * mask

        # Flatten
        pred_flat = pred_binary.flatten().long()
        target_flat = target.flatten().long()

        self.tp += (pred_flat * target_flat).sum().item()
        self.fp += (pred_flat * (1 - target_flat)).sum().item()
        self.fn += ((1 - pred_flat) * target_flat).sum().item()
        self.tn += ((1 - pred_flat) * (1 - target_flat)).sum().item()

        # Per-sample Dice
        batch_size = pred.shape[0]
        for b in range(batch_size):
            p = pred[b].flatten()
            t = target[b].flatten()
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            if union > 0:
                self.dice_sum += (2 * intersection / (union + 1e-8)).item()
            else:
                self.dice_sum += 1.0
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        eps = 1e-8

        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return {
            "iou": iou,
            "dice": dice,
            "dice_avg": self.dice_sum / max(self.count, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class ClassificationMetrics:
    """Metrics for classification tasks (lithology classification)."""

    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds_list = []
        self.targets_list = []
        self.confidences_list = []

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            logits: (B, L, num_classes) or (B, num_classes)
            target: (B, L) or (B,)
            confidence: Optional (B, L, 1)
            mask: Optional valid mask
        """
        if logits.dim() == 3:
            # Sequence classification
            B, L, C = logits.shape
            preds = logits.argmax(dim=-1)  # (B, L)

            if mask is not None:
                valid = mask.bool()
                preds = preds[valid]
                target = target[valid]
                if confidence is not None:
                    confidence = confidence[valid]
            else:
                preds = preds.flatten()
                target = target.flatten()

        else:
            preds = logits.argmax(dim=-1)

        self.preds_list.append(preds.cpu().numpy())
        self.targets_list.append(target.cpu().numpy())

        if confidence is not None:
            self.confidences_list.append(confidence.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute classification metrics."""
        if not self.preds_list:
            return {}

        preds = np.concatenate(self.preds_list)
        targets = np.concatenate(self.targets_list)

        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
        }

        return metrics


class RegressionMetrics:
    """Metrics for regression tasks (reservoir property prediction)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds_list = []
        self.targets_list = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.preds_list.append(pred.detach().cpu().numpy().flatten())
        self.targets_list.append(target.detach().cpu().numpy().flatten())

    def compute(self) -> Dict[str, float]:
        if not self.preds_list:
            return {}

        preds = np.concatenate(self.preds_list)
        targets = np.concatenate(self.targets_list)

        return {
            "mse": mean_squared_error(targets, preds),
            "rmse": np.sqrt(mean_squared_error(targets, preds)),
            "mae": mean_absolute_error(targets, preds),
            "r2": r2_score(targets, preds),
            "pearson_corr": np.corrcoef(preds, targets)[0, 1],
        }


class RetrievalMetrics:
    """
    Cross-modal retrieval metrics.

    Evaluates how well the model aligns seismic and well log embeddings.
    """

    @staticmethod
    def compute(
        seismic_embeds: torch.Tensor,
        well_embeds: torch.Tensor,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, float]:
        """
        Compute retrieval accuracy at K.

        Args:
            seismic_embeds: (N, D) normalized embeddings
            well_embeds: (N, D) normalized embeddings
            k_values: List of K values for Recall@K

        Returns:
            dict with recall@k metrics
        """
        N = seismic_embeds.shape[0]

        # Similarity matrix
        sim = seismic_embeds @ well_embeds.T  # (N, N)

        metrics = {}

        # Seismic -> Well retrieval
        for k in k_values:
            if k > N:
                continue
            _, top_k = sim.topk(k, dim=1)
            correct = sum(
                i in top_k[i].tolist() for i in range(N)
            )
            metrics[f"s2w_recall@{k}"] = correct / N

        # Well -> Seismic retrieval
        for k in k_values:
            if k > N:
                continue
            _, top_k = sim.T.topk(k, dim=1)
            correct = sum(
                i in top_k[i].tolist() for i in range(N)
            )
            metrics[f"w2s_recall@{k}"] = correct / N

        # Mean Reciprocal Rank
        mrr_s2w = 0.0
        mrr_w2s = 0.0
        for i in range(N):
            rank_s2w = (sim[i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
            rank_w2s = (sim[:, i].argsort(descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
            mrr_s2w += 1.0 / rank_s2w
            mrr_w2s += 1.0 / rank_w2s

        metrics["mrr_s2w"] = mrr_s2w / N
        metrics["mrr_w2s"] = mrr_w2s / N

        return metrics
