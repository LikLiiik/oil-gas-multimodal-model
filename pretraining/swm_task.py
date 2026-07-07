"""
Seismic-Well Matching (SWM) Pretraining Task

A binary classification task that predicts whether a given seismic volume
and well log curve pair come from the same well location (positive) or
different wells/locations (negative).

This task provides a global cross-modal alignment signal:
- Positive: seismic around wellbore + corresponding well logs
- Negative: randomly paired seismic + well logs from different locations

Helps the model learn high-level cross-modal correspondence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class SeismicWellMatching(nn.Module):
    """
    Seismic-Well Matching (SWM) head.

    Binary classifier that predicts whether a seismic-well pair
    corresponds to the same subsurface location.

    Architecture:
        [seismic_feat; well_feat; seismic_feat * well_feat; |seismic_feat - well_feat|]
        -> MLP -> sigmoid -> match probability
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_hidden_dim: int = 512,
        dropout: float = 0.1,
        num_mlp_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input: [s; w; s*w; |s-w|] = 4 * hidden_dim
        input_dim = hidden_dim * 4

        layers = []
        current_dim = input_dim

        for i in range(num_mlp_layers):
            out_dim = mlp_hidden_dim if i < num_mlp_layers - 1 else 1
            layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.GELU() if i < num_mlp_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_mlp_layers - 1 else nn.Identity(),
            ])
            current_dim = out_dim

        self.classifier = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def compute_matching_features(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching features from paired representations.

        Args:
            seismic_feat: (B, hidden_dim)
            well_feat: (B, hidden_dim)

        Returns:
            matching_features: (B, 4 * hidden_dim)
        """
        # Element-wise interaction features
        interaction = seismic_feat * well_feat
        difference = torch.abs(seismic_feat - well_feat)

        return torch.cat([seismic_feat, well_feat, interaction, difference], dim=-1)

    def create_negative_pairs(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
        shuffle: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create negative pairs by shuffling well features across batch.

        Args:
            seismic_feat: (B, hidden_dim)
            well_feat: (B, hidden_dim)
            shuffle: If True, shuffle to create negatives

        Returns:
            seismic_all: (2B, hidden_dim) positive and negative seismic
            well_all: (2B, hidden_dim) positive and negative well
            labels: (2B,) 1 for positive, 0 for negative
        """
        B = seismic_feat.shape[0]

        # Positive pairs
        positive_seismic = seismic_feat
        positive_well = well_feat
        positive_labels = torch.ones(B, device=seismic_feat.device)

        if shuffle:
            # Negative pairs by shuffling
            neg_indices = torch.randperm(B, device=seismic_feat.device)
            negative_seismic = seismic_feat
            negative_well = well_feat[neg_indices]
            negative_labels = torch.zeros(B, device=seismic_feat.device)
        else:
            # Use all non-diagonal pairs
            negative_seismic = seismic_feat.repeat(B, 1)
            negative_well = well_feat.repeat_interleave(B, dim=0)
            # Remove diagonal (positive pairs)
            mask = torch.eye(B, device=seismic_feat.device, dtype=torch.bool)
            mask = mask.repeat(B, 1)
            negative_seismic = negative_seismic[~mask.flatten()]
            negative_well = negative_well[~mask.flatten()]
            negative_labels = torch.zeros(len(negative_seismic), device=seismic_feat.device)

        # Concatenate
        seismic_all = torch.cat([positive_seismic, negative_seismic], dim=0)
        well_all = torch.cat([positive_well, negative_well], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0)

        return seismic_all, well_all, labels

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SWM.

        Args:
            seismic_feat: (B, hidden_dim)
            well_feat: (B, hidden_dim)
            labels: Optional (2B,) matching labels

        Returns:
            dict with loss and predictions
        """
        if labels is None:
            seismic_all, well_all, labels = self.create_negative_pairs(
                seismic_feat, well_feat
            )
        else:
            seismic_all = seismic_feat
            well_all = well_feat

        # Compute matching features
        match_features = self.compute_matching_features(seismic_all, well_all)

        # Predict
        logits = self.classifier(match_features).squeeze(-1)
        probs = self.sigmoid(logits)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(probs, labels)

        # Accuracy
        with torch.no_grad():
            pred_labels = (probs > 0.5).float()
            accuracy = (pred_labels == labels).float().mean()

        return {
            "loss": loss,
            "matching_accuracy": accuracy,
            "match_probabilities": probs,
            "match_labels": labels,
        }


class HardNegativeMiningMatching(nn.Module):
    """
    Seismic-Well Matching with hard negative mining.

    Extends basic SWM by identifying and emphasizing hard negatives
    (seismic-well pairs that are difficult to distinguish) during training.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_hidden_dim: int = 512,
        num_hard_negatives: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_hard_negatives = num_hard_negatives

        # Base matching head
        self.base_matcher = SeismicWellMatching(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hard negative mining.

        Args:
            seismic_feat: (B, hidden_dim)
            well_feat: (B, hidden_dim)

        Returns:
            dict with loss, accuracy, etc.
        """
        B = seismic_feat.shape[0]

        # 1. Compute all pairwise similarities
        with torch.no_grad():
            # Quick similarity check to find hard negatives
            sim_matrix = F.normalize(seismic_feat, dim=-1) @ \
                         F.normalize(well_feat, dim=-1).T  # (B, B)

        # 2. Standard matching with all pairs
        output = self.base_matcher(seismic_feat, well_feat)

        # 3. Additional hard negative loss
        # Hard negatives: non-diagonal pairs with high similarity
        mask = torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
        sim_off_diag = sim_matrix.masked_fill(mask, -float("inf"))

        # Select top-K hard negatives
        _, hard_indices = sim_off_diag.topk(
            min(self.num_hard_negatives, B), dim=1
        )

        hard_loss = 0.0
        count = 0

        for i in range(B):
            for j in hard_indices[i]:
                if j < B:
                    s_feat = seismic_feat[i:i+1]
                    w_feat = well_feat[j:j+1]
                    hard_output = self.base_matcher(
                        s_feat, w_feat,
                        labels=torch.zeros(1, device=seismic_feat.device),
                    )
                    hard_loss += hard_output["loss"]
                    count += 1

        if count > 0:
            output["hard_negative_loss"] = hard_loss / count
            output["loss"] = output["loss"] + 0.3 * output["hard_negative_loss"]

        return output
