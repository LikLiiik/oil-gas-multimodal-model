"""
Task-Specific Prediction Heads

Downstream task heads for:
1. Fault Detection: 3D segmentation (seismic -> fault probability volume)
2. Reservoir Prediction: 3D segmentation with well log conditioning
3. Lithology Classification: Sequence classification along wellbore

Each head can operate on fused multi-modal features or single-modality features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from einops import rearrange


# =====================================================================
# 3D UNet Decoder Block
# =====================================================================

class UNetDecoderBlock3D(nn.Module):
    """Decoder block for 3D U-Net with skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsample: bool = True,
    ):
        super().__init__()
        self.upsample = upsample

        if upsample:
            self.up = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=2, stride=2,
            )
        else:
            self.up = nn.Identity()

        self.conv1 = nn.Conv3d(
            out_channels + skip_channels if upsample else in_channels + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.upsample:
            x = self.up(x)

        # Match spatial dimensions
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


# =====================================================================
# Fault Detection Head
# =====================================================================

class FaultDetectionHead(nn.Module):
    """
    3D Fault Detection Head.

    Uses a 3D U-Net decoder architecture with skip connections
    from the seismic encoder for high-resolution fault segmentation.

    Input:
        - fused_features: (B, hidden_dim) global features from fusion
        - encoder_features: List of (B, Ci, Di, Hi, Wi) from seismic encoder
        - seismic_input: (B, 1, D, H, W) original seismic volume (optional)

    Output:
        - fault_prob: (B, 1, D, H, W) fault probability volume
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        encoder_channels: List[int] = [192, 384, 768, 1536],
        decoder_channels: List[int] = [256, 128, 64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_channels = encoder_channels

        # Global feature injection
        self.global_to_3d = nn.Sequential(
            nn.Linear(hidden_dim, encoder_channels[-1] * 4 * 4 * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_spatial_size = (4, 4, 4)

        # Decoder blocks (from deep to shallow)
        decoder_ch = decoder_channels
        self.decoder_blocks = nn.ModuleList()

        # Bottom block (no skip, from global feature + deepest encoder)
        self.decoder_blocks.append(
            UNetDecoderBlock3D(
                in_channels=encoder_channels[-1],
                skip_channels=0,
                out_channels=decoder_ch[0],
                upsample=False,
            )
        )

        # Intermediate decoder blocks
        for i in range(len(encoder_channels) - 1):
            enc_idx = len(encoder_channels) - 2 - i
            self.decoder_blocks.append(
                UNetDecoderBlock3D(
                    in_channels=decoder_ch[i],
                    skip_channels=encoder_channels[enc_idx],
                    out_channels=decoder_ch[i + 1],
                    upsample=True,
                )
            )

        # Final output (use actual last decoder channel, not decoder_ch[-1])
        n_blocks = 1 + len(encoder_channels) - 1  # = len(encoder_channels)
        last_ch = decoder_ch[min(n_blocks - 1, len(decoder_ch) - 1)]
        self.final_conv = nn.Sequential(
            nn.Conv3d(last_ch, last_ch // 2, 3, padding=1),
            nn.BatchNorm3d(last_ch // 2),
            nn.GELU(),
            nn.Conv3d(last_ch // 2, 1, 1),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        encoder_features: List[torch.Tensor],
        seismic_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            fused_features: (B, hidden_dim)
            encoder_features: List of (B, Ci, Di, Hi, Wi) from encoder
            seismic_input: Optional (B, 1, D, H, W) for skip connection

        Returns:
            fault_prob: (B, 1, D, H, W)
        """
        B = fused_features.shape[0]

        # Project global features to 3D
        global_3d = self.global_to_3d(fused_features)  # (B, C*4*4*4)
        global_3d = global_3d.view(
            B, self.encoder_channels[-1],
            *self.global_spatial_size
        )  # (B, C_last, 4, 4, 4)

        # Combine with deepest encoder feature
        enc_deep = encoder_features[-1]
        if global_3d.shape[2:] != enc_deep.shape[2:]:
            global_3d = F.interpolate(
                global_3d, size=enc_deep.shape[2:],
                mode="trilinear", align_corners=False,
            )
        x = global_3d + enc_deep

        # Decoder with skip connections
        for i, dec_block in enumerate(self.decoder_blocks):
            if i == 0:
                x = dec_block(x, skip=None)
            else:
                skip = encoder_features[-(i + 1)]
                x = dec_block(x, skip=skip)

        # Upsample to original input size
        if seismic_input is not None and x.shape[2:] != seismic_input.shape[2:]:
            x = F.interpolate(
                x, size=seismic_input.shape[2:],
                mode="trilinear", align_corners=False,
            )

        # Final output
        fault_prob = self.final_conv(x)
        return torch.sigmoid(fault_prob)


# =====================================================================
# Reservoir Prediction Head
# =====================================================================

class ReservoirPredictionHead(nn.Module):
    """
    Reservoir Prediction Head.

    Predicts reservoir property volumes by conditioning seismic features
    on well log-derived lithology and fluid information.

    Two output modes:
    1. Reservoir probability volume (B, 1, D, H, W)
    2. Reservoir property volume (B, n_properties, D, H, W)

    Uses well log features as conditioning to guide seismic interpretation.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        encoder_channels: List[int] = [192, 384, 768, 1536],
        decoder_channels: List[int] = [256, 128, 64, 32],
        n_properties: int = 2,  # e.g., porosity, permeability
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_properties = n_properties

        # Well log conditioning module
        self.well_conditioning = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, encoder_channels[-1]),
            nn.Sigmoid(),  # Channel-wise attention weights
        )

        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Conv3d(encoder_channels[-1], encoder_channels[-1], 1),
            nn.BatchNorm3d(encoder_channels[-1]),
            nn.GELU(),
        )

        # Decoder blocks similar to fault head
        decoder_ch = decoder_channels
        self.decoder_blocks = nn.ModuleList()

        self.decoder_blocks.append(
            UNetDecoderBlock3D(
                in_channels=encoder_channels[-1],
                skip_channels=0,
                out_channels=decoder_ch[0],
                upsample=False,
            )
        )

        for i in range(len(encoder_channels) - 1):
            enc_idx = len(encoder_channels) - 2 - i
            self.decoder_blocks.append(
                UNetDecoderBlock3D(
                    in_channels=decoder_ch[i],
                    skip_channels=encoder_channels[enc_idx],
                    out_channels=decoder_ch[i + 1],
                    upsample=True,
                )
            )

        # Output heads
        n_blocks = 1 + len(encoder_channels) - 1
        last_ch = decoder_ch[min(n_blocks - 1, len(decoder_ch) - 1)]

        # Reservoir probability
        self.reservoir_prob_head = nn.Sequential(
            nn.Conv3d(last_ch, last_ch // 2, 3, padding=1),
            nn.BatchNorm3d(last_ch // 2),
            nn.GELU(),
            nn.Conv3d(last_ch // 2, 1, 1),
        )

        # Reservoir properties
        self.reservoir_prop_head = nn.Sequential(
            nn.Conv3d(last_ch, last_ch // 2, 3, padding=1),
            nn.BatchNorm3d(last_ch // 2),
            nn.GELU(),
            nn.Conv3d(last_ch // 2, n_properties, 1),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        well_features: torch.Tensor,
        encoder_features: List[torch.Tensor],
        seismic_input: Optional[torch.Tensor] = None,
        output_properties: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: (B, hidden_dim) from fusion module
            well_features: (B, hidden_dim) from well log encoder
            encoder_features: List of (B, Ci, Di, Hi, Wi)
            seismic_input: Optional (B, 1, D, H, W)
            output_properties: If True, also predict reservoir properties

        Returns:
            dict with:
                - reservoir_prob: (B, 1, D, H, W)
                - reservoir_props: (B, n_properties, D, H, W) if output_properties
        """
        B = fused_features.shape[0]

        # Well log conditioning: channel attention on deepest seismic features
        channel_weights = self.well_conditioning(well_features)  # (B, C_last)
        channel_weights = channel_weights.view(B, -1, 1, 1, 1)

        # Apply conditioning to deepest encoder feature
        enc_deep = encoder_features[-1]
        conditioned = enc_deep * channel_weights
        conditioned = self.feature_enhance(conditioned)

        x = conditioned

        # Decoder
        for i, dec_block in enumerate(self.decoder_blocks):
            if i == 0:
                x = dec_block(x, skip=None)
            else:
                skip = encoder_features[-(i + 1)]
                x = dec_block(x, skip=skip)

        # Upsample
        if seismic_input is not None and x.shape[2:] != seismic_input.shape[2:]:
            x = F.interpolate(
                x, size=seismic_input.shape[2:],
                mode="trilinear", align_corners=False,
            )

        # Outputs
        reservoir_prob = torch.sigmoid(self.reservoir_prob_head(x))

        result = {"reservoir_prob": reservoir_prob}

        if output_properties:
            reservoir_props = self.reservoir_prop_head(x)
            result["reservoir_props"] = reservoir_props

        return result


# =====================================================================
# Lithology Classification Head
# =====================================================================

class LithologyClassificationHead(nn.Module):
    """
    Lithology Classification Head.

    Classifies lithology along the wellbore by combining:
    - Well log sequence features (from well log encoder)
    - Seismic trace features at well position (from seismic encoder)

    Supports lithology classes: shale, sand, carbonate, coal, etc.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Feature fusion for classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sequence-level classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # Confidence estimation
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        well_features: torch.Tensor,
        seismic_trace_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            well_features: (B, L, hidden_dim) from well log encoder (sequence)
            seismic_trace_features: Optional (B, L, hidden_dim) from seismic encoder

        Returns:
            dict with:
                - logits: (B, L, num_classes)
                - confidence: (B, L, 1)
        """
        B, L, C = well_features.shape

        # Combine features
        if seismic_trace_features is not None:
            combined = torch.cat([well_features, seismic_trace_features], dim=-1)
        else:
            combined = torch.cat([well_features, torch.zeros_like(well_features)], dim=-1)

        # Fusion
        feat = self.fusion(combined)  # (B, L, hidden_dim//2)

        # Classification
        logits = self.classifier(feat)  # (B, L, num_classes)
        confidence = self.confidence(feat)  # (B, L, 1)

        return {
            "logits": logits,
            "confidence": confidence,
        }


# =====================================================================
# Multi-Task Head Wrapper
# =====================================================================

class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head wrapper.

    Combines multiple task heads and manages their interactions.
    Supports task-specific gradient flow control.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        encoder_channels: List[int] = [192, 384, 768, 1536],
        num_lithology_classes: int = 4,
        n_reservoir_properties: int = 2,
        tasks: List[str] = ["fault_detection", "reservoir_prediction", "lithology"],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tasks = tasks

        if "fault_detection" in tasks:
            self.fault_head = FaultDetectionHead(
                hidden_dim=hidden_dim,
                encoder_channels=encoder_channels,
                dropout=dropout,
            )
        else:
            self.fault_head = None

        if "reservoir_prediction" in tasks:
            self.reservoir_head = ReservoirPredictionHead(
                hidden_dim=hidden_dim,
                encoder_channels=encoder_channels,
                n_properties=n_reservoir_properties,
                dropout=dropout,
            )
        else:
            self.reservoir_head = None

        if "lithology" in tasks:
            self.lithology_head = LithologyClassificationHead(
                hidden_dim=hidden_dim,
                num_classes=num_lithology_classes,
                dropout=dropout,
            )
        else:
            self.lithology_head = None

    def forward(
        self,
        fused_features: torch.Tensor,
        well_features: torch.Tensor,
        encoder_features: List[torch.Tensor],
        seismic_input: Optional[torch.Tensor] = None,
        seismic_trace_features: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for specified task(s).

        Args:
            fused_features: (B, hidden_dim)
            well_features: (B, hidden_dim)
            encoder_features: List of (B, Ci, Di, Hi, Wi)
            seismic_input: Optional (B, 1, D, H, W)
            seismic_trace_features: Optional (B, L, hidden_dim)
            task: Specific task to run, or None for all

        Returns:
            dict of task-specific outputs
        """
        outputs = {}

        if task is None or task == "fault_detection":
            if self.fault_head is not None:
                fault_prob = self.fault_head(
                    fused_features, encoder_features, seismic_input
                )
                outputs["fault_prob"] = fault_prob

        if task is None or task == "reservoir_prediction":
            if self.reservoir_head is not None:
                reservoir_out = self.reservoir_head(
                    fused_features, well_features, encoder_features, seismic_input
                )
                outputs.update(reservoir_out)

        if task is None or task == "lithology":
            if self.lithology_head is not None and seismic_trace_features is not None:
                litho_out = self.lithology_head(
                    well_features, seismic_trace_features
                )
                outputs.update(litho_out)

        return outputs
