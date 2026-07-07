"""
Cross-Modal Contrastive Learning (CMCL) Pretraining Task

Implements contrastive learning between seismic volumes and well log curves.
Follows a CLIP-style approach where paired seismic-well data form positive pairs
and all other combinations form negative pairs within a batch.

Key insight:
- Seismic trace at well position and the corresponding well log curves
  represent the same subsurface geology from different measurement modalities.
- This provides a natural self-supervision signal for cross-modal alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class CrossModalContrastiveLearning(nn.Module):
    """
    Cross-Modal Contrastive Learning (CMCL).

    Aligns seismic and well log representations in a shared embedding space
    using InfoNCE (contrastive) loss.

    The contrastive loss encourages:
    - Paired seismic-well representations to be close (positive pairs)
    - Unpaired combinations to be far apart (negative pairs, in-batch)

    Supports both global contrast (volume-level) and local contrast
    (trace-level at well position).
    """

    def __init__(
        self,
        seismic_dim: int = 384,
        well_dim: int = 384,
        projection_dim: int = 256,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature

        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * torch.log(torch.tensor(1 / temperature))
            )
        else:
            self.register_buffer(
                "logit_scale",
                torch.ones([]) * torch.log(torch.tensor(1 / temperature)),
            )

        # Projection heads
        self.seismic_proj = nn.Sequential(
            nn.Linear(seismic_dim, seismic_dim),
            nn.GELU(),
            nn.Linear(seismic_dim, projection_dim),
        )

        self.well_proj = nn.Sequential(
            nn.Linear(well_dim, well_dim),
            nn.GELU(),
            nn.Linear(well_dim, projection_dim),
        )

    def encode(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project features to contrastive embedding space.

        Args:
            seismic_feat: (B, seismic_dim)
            well_feat: (B, well_dim)

        Returns:
            z_seis: (B, projection_dim) L2-normalized
            z_well: (B, projection_dim) L2-normalized
        """
        z_seis = self.seismic_proj(seismic_feat)
        z_well = self.well_proj(well_feat)

        # L2 normalize
        z_seis = F.normalize(z_seis, dim=-1)
        z_well = F.normalize(z_well, dim=-1)

        return z_seis, z_well

    def compute_contrastive_loss(
        self,
        z_seis: torch.Tensor,
        z_well: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            z_seis: (B, D) L2-normalized seismic embeddings
            z_well: (B, D) L2-normalized well log embeddings

        Returns:
            loss: scalar
        """
        B = z_seis.shape[0]

        # Cosine similarity matrix
        logits = z_seis @ z_well.T  # (B, B)
        logits = logits * self.logit_scale.exp()

        # Labels: diagonal = positive pairs
        labels = torch.arange(B, device=logits.device)

        # Bidirectional InfoNCE
        loss_s2w = F.cross_entropy(logits, labels)
        loss_w2s = F.cross_entropy(logits.T, labels)

        loss = (loss_s2w + loss_w2s) / 2

        return loss

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CMCL.

        Args:
            seismic_feat: (B, seismic_dim) or (B, seq_len, seismic_dim)
            well_feat: (B, well_dim) or (B, seq_len, well_dim)

        Returns:
            dict with loss, similarity matrix, and embeddings
        """
        # Global pooling if sequence features
        if seismic_feat.dim() == 3:
            seismic_feat = seismic_feat.mean(dim=1)
        if well_feat.dim() == 3:
            well_feat = well_feat.mean(dim=1)

        # Project
        z_seis, z_well = self.encode(seismic_feat, well_feat)

        # Compute loss
        loss = self.compute_contrastive_loss(z_seis, z_well)

        # Compute accuracy for monitoring
        with torch.no_grad():
            logits = z_seis @ z_well.T * self.logit_scale.exp()
            acc = (logits.argmax(dim=1) == torch.arange(
                logits.shape[0], device=logits.device
            )).float().mean()

        return {
            "loss": loss,
            "contrastive_accuracy": acc,
            "seismic_embeddings": z_seis,
            "well_embeddings": z_well,
            "logit_scale": self.logit_scale.exp(),
        }


class LocalContrastiveLearning(nn.Module):
    """
    Local (trace-level) contrastive learning.

    Instead of contrasting global volume features with global well features,
    this contrasts individual seismic traces with their corresponding
    depth-matched well log values.

    This provides a stronger alignment signal for spatial correspondence.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature

        self.seismic_trace_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, projection_dim),
        )

        self.well_depth_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, projection_dim),
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / temperature))
        )

    def forward(
        self,
        seismic_trace_feat: torch.Tensor,
        well_depth_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seismic_trace_feat: (B, L, D) seismic features along well trajectory
            well_depth_feat: (B, L, D) well log features at each depth

        Returns:
            dict with local_contrastive_loss
        """
        B, L, D = seismic_trace_feat.shape

        # Project
        z_trace = F.normalize(
            self.seismic_trace_proj(seismic_trace_feat), dim=-1
        )
        z_depth = F.normalize(
            self.well_depth_proj(well_depth_feat), dim=-1
        )

        # Local contrast at each depth position
        # Positive: same depth, negative: other depths
        loss = 0
        for b in range(B):
            logits = z_trace[b] @ z_depth[b].T  # (L, L)
            logits = logits * self.logit_scale.exp()
            labels = torch.arange(L, device=logits.device)
            loss += (F.cross_entropy(logits, labels) +
                     F.cross_entropy(logits.T, labels)) / 2

        loss = loss / B

        return {
            "local_contrastive_loss": loss,
        }
