"""
Loss Functions

Collection of loss functions for multi-modal oil & gas tasks:
- Dice Loss for 3D segmentation (fault detection)
- Focal Loss for class-imbalanced classification
- SSIM Loss for structural similarity (reservoir property prediction)
- InfoNCE Loss for contrastive learning
- MultiTaskLoss with uncertainty weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math


# =====================================================================
# Dice Loss
# =====================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Commonly used in medical/geophysical segmentation
    due to robustness to class imbalance.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, D, H, W) predicted probabilities
            target: (B, 1, D, H, W) binary ground truth
            mask: Optional (B, 1, D, H, W) valid region mask

        Returns:
            dice_loss: scalar
        """
        batch_size = pred.shape[0]

        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        if mask is not None:
            mask_flat = mask.reshape(batch_size, -1)
            pred_flat = pred_flat * mask_flat
            target_flat = target_flat * mask_flat

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return 1.0 - dice.mean()
        elif self.reduction == "sum":
            return (1.0 - dice).sum()
        else:
            return 1.0 - dice


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for multi-class segmentation.

    Weights each class inversely proportional to its frequency.
    """

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) logits
            target: (B, D, H, W) class indices
        """
        B, C = pred.shape[:2]
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), num_classes=C)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        # Class weights
        class_weights = 1.0 / (target_one_hot.sum(dim=(0, 2, 3, 4)) ** 2 + self.smooth)

        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3, 4))
        union = (pred + target_one_hot).sum(dim=(0, 2, 3, 4))

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        weighted_dice = (class_weights * dice_per_class).sum() / class_weights.sum()

        return 1.0 - weighted_dice


# =====================================================================
# Focal Loss
# =====================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples and focuses on hard ones.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, *) logits
            target: (B, *) binary labels (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if mask is not None:
            focal_loss = focal_loss * mask

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# =====================================================================
# SSIM Loss
# =====================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss for 3D volumes.

    Encourages structural consistency between prediction and target.
    Useful for reservoir property prediction where exact pixel values
    matter less than the overall structural pattern.

    SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
    Loss = 1 - SSIM
    """

    def __init__(
        self,
        window_size: int = 7,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        # Gaussian window
        window_1d = torch.exp(
            -torch.arange(window_size).float() ** 2
            / (2 * (window_size / 6) ** 2)
        )
        window_1d = window_1d / window_1d.sum()
        window_3d = window_1d[:, None, None] * window_1d[None, :, None] * window_1d[None, None, :]
        self.register_buffer("window_3d", window_3d.unsqueeze(0).unsqueeze(0))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, D, H, W)
            target: (B, 1, D, H, W)

        Returns:
            ssim_loss: scalar
        """
        B = pred.shape[0]
        window = self.window_3d.to(pred.device)

        mu_pred = F.conv3d(pred, window, padding=self.window_size // 2, groups=1)
        mu_target = F.conv3d(target, window, padding=self.window_size // 2, groups=1)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv3d(pred ** 2, window, padding=self.window_size // 2) - mu_pred_sq
        sigma_target_sq = F.conv3d(target ** 2, window, padding=self.window_size // 2) - mu_target_sq
        sigma_pred_target = F.conv3d(pred * target, window, padding=self.window_size // 2) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1.0 - ssim_map.mean()


# =====================================================================
# InfoNCE Loss
# =====================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) Loss.

    Standard contrastive learning loss:
    L = -log(exp(sim(z_i, z_i+) / τ) / Σ_j exp(sim(z_i, z_j) / τ))

    Used in cross-modal contrastive pretraining.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / temperature))
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1: (B, D) L2-normalized embeddings (modality 1)
            z2: (B, D) L2-normalized embeddings (modality 2)

        Returns:
            loss: scalar
        """
        B = z1.shape[0]

        # Cosine similarity
        logits = (z1 @ z2.T) * self.logit_scale.exp()

        labels = torch.arange(B, device=logits.device)

        loss_s2w = F.cross_entropy(logits, labels)
        loss_w2s = F.cross_entropy(logits.T, labels)

        return (loss_s2w + loss_w2s) / 2


# =====================================================================
# Multi-Task Loss with Uncertainty Weighting
# =====================================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting.

    Implements Kendall et al. (2018) uncertainty-based loss weighting:
    L_total = Σ_i [1/(2σ_i²) * L_i + log(σ_i)]

    This automatically balances task-specific losses during training
    without manual hyperparameter tuning.
    """

    def __init__(
        self,
        task_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.task_names = task_names

        # Learnable log variances per task
        self.log_vars = nn.ParameterDict()
        for task in task_names:
            init_val = initial_log_vars.get(task, 0.0) if initial_log_vars else 0.0
            self.log_vars[task] = nn.Parameter(torch.tensor(init_val))

        # Task-specific loss functions
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ssim_loss = SSIMLoss()

    def compute_individual_losses(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual (unscaled) losses for each task.

        Override this in practice for specific task logic.
        """
        losses = {}

        if "fault" in preds and "fault_mask" in targets:
            fault_pred = preds["fault"]
            fault_target = targets["fault_mask"]
            losses["fault"] = (
                self.dice_loss(fault_pred, fault_target) +
                self.focal_loss(fault_pred, fault_target)
            ) / 2

        if "reservoir" in preds and "reservoir_mask" in targets:
            res_pred = preds["reservoir"]
            res_target = targets["reservoir_mask"]
            losses["reservoir"] = (
                self.dice_loss(res_pred, res_target) * 0.5 +
                self.ssim_loss(res_pred, res_target) * 0.5
            )

        if "lithology" in preds and "lithology" in targets:
            litho_pred = preds["lithology"]
            litho_target = targets["lithology"]
            losses["lithology"] = self.ce_loss(litho_pred, litho_target)

        return losses

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Returns:
            dict with total_loss and individual task losses
        """
        individual_losses = self.compute_individual_losses(preds, targets)

        total_loss = 0.0
        weighted_losses = {}

        for task_name in self.task_names:
            if task_name in individual_losses:
                raw_loss = individual_losses[task_name]
                log_var = self.log_vars[task_name]

                # Precision-weighted loss
                precision = torch.exp(-log_var)
                weighted_loss = precision * raw_loss + log_var
                total_loss += weighted_loss
                weighted_losses[task_name] = weighted_loss.detach()

        return {
            "total_loss": total_loss,
            "weighted_losses": weighted_losses,
            "raw_losses": individual_losses,
        }


# =====================================================================
# Combined Pretraining Loss
# =====================================================================

class PretrainingLoss(nn.Module):
    """
    Combined loss for multi-task self-supervised pretraining.

    L_total = Σ w_i * L_i

    where i ∈ {MSM, MWM, CMCL, SWM}
    """

    def __init__(
        self,
        msm_weight: float = 1.0,
        mwm_weight: float = 1.0,
        cmcl_weight: float = 0.5,
        swm_weight: float = 0.3,
    ):
        super().__init__()
        self.weights = {
            "msm": msm_weight,
            "mwm": mwm_weight,
            "cmcl": cmcl_weight,
            "swm": swm_weight,
        }

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Weight and combine pretraining losses."""
        total_loss = 0.0
        weighted = {}

        for key, weight in self.weights.items():
            if key in losses and losses[key] is not None:
                w_loss = weight * losses[key]
                total_loss += w_loss
                weighted[key] = w_loss.detach()

        return {
            "total_loss": total_loss,
            "weighted_losses": weighted,
        }
