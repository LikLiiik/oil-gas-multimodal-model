"""
Masked Well-log Modeling (MWM) Pretraining Task

Randomly masks segments of well log curves and trains the model
to reconstruct the missing values. Similar to BERT's masked token
prediction but adapted for continuous 1D time-series data.

Key features:
- Block masking: masks consecutive depth intervals
- Multi-curve reconstruction: predicts all curves simultaneously
- Depth-aware masking: higher mask probability in heterogeneous zones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import random


class MaskedWellLogModeling(nn.Module):
    """
    Masked Well-log Modeling (MWM) task.

    Masks random depth intervals and reconstructs log values.
    The model must leverage cross-curve correlations and
    depth context to fill in the missing data.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_curves: int = 7,
        mask_ratio: float = 0.5,
        mask_block_size: int = 16,
        decoder_layers: int = 2,
        decoder_hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_curves = num_curves
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

        # Get encoder output dimension
        self.encoder_dim = (
            encoder.get_output_dim()
            if hasattr(encoder, "get_output_dim")
            else 192
        )

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder_dim, decoder_hidden_dim),
            nn.GELU(),
        )

        # Additional transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_hidden_dim,
                nhead=4,
                dim_feedforward=decoder_hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_layers)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_hidden_dim)

        # Reconstruction head
        self.reconstruction_head = nn.Linear(decoder_hidden_dim, num_curves)

    def generate_block_mask(
        self, shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate block-based mask for well log sequences.

        Args:
            shape: (B, L) batch and sequence dimensions

        Returns:
            mask: (B, L) boolean mask (True = masked)
        """
        B, L = shape
        mask = torch.zeros(B, L, dtype=torch.bool)

        num_blocks = max(1, int(L * self.mask_ratio / self.mask_block_size))

        for b in range(B):
            masked_count = 0
            attempts = 0
            max_attempts = num_blocks * 10

            while masked_count < L * self.mask_ratio and attempts < max_attempts:
                block_size = random.randint(
                    max(1, self.mask_block_size // 2),
                    self.mask_block_size * 2,
                )
                start = random.randint(0, L - block_size)

                if mask[b, start:start + block_size].sum() == 0:
                    mask[b, start:start + block_size] = True
                    masked_count += block_size

                attempts += 1

        return mask

    def apply_mask(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mask to input by replacing masked positions with learnable token.

        Args:
            x: (B, L, C) well log features
            mask: (B, L) boolean mask

        Returns:
            masked_x: (B, L, C) with masked positions replaced
        """
        # Replace masked positions with 0 (or learnable mask token)
        mask_expanded = mask.unsqueeze(-1).float()
        masked_x = x * (1 - mask_expanded)
        return masked_x

    def forward_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss on masked positions.

        Args:
            pred: (B, L, num_curves) predicted values
            target: (B, L, num_curves) original values
            mask: (B, L) boolean mask

        Returns:
            loss: scalar
        """
        loss = F.smooth_l1_loss(
            pred[mask], target[mask], reduction="mean"
        )
        return loss

    def forward(
        self, well_log: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MWM pretraining.

        Args:
            well_log: (B, C, L) well log curves
            mask: Optional pre-defined mask (B, L)

        Returns:
            dict with loss and reconstructions
        """
        B, C, L = well_log.shape

        # Generate mask if not provided
        if mask is None:
            mask = self.generate_block_mask((B, L)).to(well_log.device)

        # Transpose for sequence processing: (B, C, L) -> (B, L, C)
        x = well_log.transpose(1, 2)

        # Apply mask
        masked_x = self.apply_mask(x, mask)

        # Encode (simplified - in production, use actual well log encoder)
        latent, _ = self.encoder(masked_x.transpose(1, 2), return_sequence=True)
        if latent is None:
            # Fallback if encoder doesn't return sequence
            latent = masked_x

        # If latent is (B, hidden_dim), expand to sequence
        if latent.dim() == 2:
            latent = latent.unsqueeze(1).expand(-1, L, -1)

        # Decode
        feat = self.decoder(latent)

        for layer in self.decoder_layers:
            feat = layer(feat)

        feat = self.decoder_norm(feat)

        # Reconstruction
        pred = self.reconstruction_head(feat)  # (B, L, num_curves)

        # Compute loss
        loss = self.forward_loss(pred, x, mask)

        return {
            "loss": loss,
            "pred": pred,
            "target": x,
            "mask": mask,
        }
