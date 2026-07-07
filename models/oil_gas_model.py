"""
Oil & Gas Multi-modal Large Model

Main model wrapper that integrates:
- 3D Seismic Encoder
- 1D Well Log Encoder
- Cross-Modal Fusion Module
- Task-Specific Prediction Heads

Supports:
1. Pretraining mode: returns features for self-supervised tasks
2. Finetuning mode: returns task-specific predictions

Architecture:
    Seismic (B,1,D,H,W) ──> SeismicEncoder3D ──> seismic_feat (B, hidden_dim)
                                                         │
    Well Log (B,C,L) ────> WellLogEncoder1D ──> well_feat (B, hidden_dim)
                                                         │
                                          ┌──────────────┴──────────────┐
                                          │    CrossModalFusion          │
                                          │  (coarse + cross-attn + gate)│
                                          └──────────────┬──────────────┘
                                                         │
                                          ┌──────────────┴──────────────┐
                                          │   MultiTaskHead              │
                                          │  ┌─────────┬─────────┐      │
                                          │  │  Fault  │Reservoir│ ...  │
                                          │  │  Detect │ Predict │      │
                                          │  └─────────┴─────────┘      │
                                          └──────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .seismic_encoder import SeismicEncoder3D
from .well_log_encoder import WellLogEncoder1D
from .fusion_module import CrossModalFusion, ModalityProjection
from .prediction_heads import MultiTaskHead

# Support both relative and absolute imports
try:
    from ..config.model_config import ModelConfig
except ImportError:
    from config.model_config import ModelConfig


class OilGasModel(nn.Module):
    """
    Complete Oil & Gas Multi-modal Model.

    This is the main model class for finetuning on downstream tasks.

    Args:
        config: ModelConfig with all sub-configurations
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = ModelConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # ==========================================
        # Encoders
        # ==========================================

        # 3D Seismic Encoder
        self.seismic_encoder = SeismicEncoder3D(
            in_channels=config.seismic_encoder.in_channels,
            stem_channels=config.seismic_encoder.stem_channels,
            embed_dim=config.seismic_encoder.embed_dim,
            depths=config.seismic_encoder.depths,
            num_heads=config.seismic_encoder.num_heads,
            window_size=config.seismic_encoder.window_size,
            patch_size=config.seismic_encoder.patch_size,
            mlp_ratio=config.seismic_encoder.mlp_ratio,
            dropout=config.seismic_encoder.dropout,
            use_checkpoint=config.seismic_encoder.use_checkpoint,
        )

        # 1D Well Log Encoder
        self.well_log_encoder = WellLogEncoder1D(
            num_curves=config.well_log_encoder.num_curves,
            stem_channels=config.well_log_encoder.stem_channels,
            kernel_sizes=config.well_log_encoder.kernel_sizes,
            embed_dim=config.well_log_encoder.embed_dim,
            num_layers=config.well_log_encoder.num_layers,
            num_heads=config.well_log_encoder.num_heads,
            mlp_ratio=config.well_log_encoder.mlp_ratio,
            dropout=config.well_log_encoder.dropout,
            max_seq_len=config.well_log_encoder.max_seq_len,
            use_physics_constraint=config.well_log_encoder.use_physics_constraint,
        )

        # ==========================================
        # Dimension Alignment
        # ==========================================

        seis_dim = self.seismic_encoder.get_output_dim()
        well_dim = self.well_log_encoder.get_output_dim()
        hidden_dim = config.hidden_dim

        self.modality_proj = ModalityProjection(
            seismic_dim=seis_dim,
            well_log_dim=well_dim,
            common_dim=hidden_dim,
            dropout=config.fusion.dropout,
        )

        # ==========================================
        # Cross-Modal Fusion
        # ==========================================

        self.fusion_module = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_cross_attention_heads=config.fusion.num_cross_attention_heads,
            num_fusion_layers=config.fusion.num_fusion_layers,
            dropout=config.fusion.dropout,
            use_gating=config.fusion.use_gating,
        )

        # ==========================================
        # Task Heads
        # ==========================================

        encoder_channels = self.seismic_encoder.out_channels

        self.task_head = MultiTaskHead(
            hidden_dim=hidden_dim,
            encoder_channels=encoder_channels,
            dropout=config.fusion.dropout,
        )

    def encode(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
        well_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode both modalities and fuse them.

        Args:
            seismic: (B, 1, D, H, W) 3D seismic volume
            well_log: (B, C, L) well log curves
            well_mask: Optional (B, L) mask for well log
            return_intermediate: Return intermediate features

        Returns:
            dict with keys:
                - fused: (B, hidden_dim) fused features
                - seismic_feat: (B, hidden_dim) projected seismic features
                - well_feat: (B, hidden_dim) projected well features
                - encoder_features: List of intermediate seismic features
                - well_sequence: (B, L_out, hidden_dim) if return_intermediate
        """
        # Encode seismic
        seismic_global, encoder_features = self.seismic_encoder(
            seismic, return_features=True
        )

        # Encode well log
        well_global, well_sequence = self.well_log_encoder(
            well_log, mask=well_mask, return_sequence=return_intermediate
        )

        # Project to common dimension
        seismic_proj, well_proj = self.modality_proj(seismic_global, well_global)

        # Multi-modal fusion
        fused = self.fusion_module(seismic_proj, well_proj)

        output = {
            "fused": fused,
            "seismic_feat": seismic_proj,
            "well_feat": well_proj,
            "encoder_features": encoder_features,
        }

        if return_intermediate and well_sequence is not None:
            output["well_sequence"] = well_sequence

        return output

    def forward(
        self,
        seismic: torch.Tensor,
        well_log: Optional[torch.Tensor] = None,
        well_mask: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            seismic: (B, 1, D, H, W) 3D seismic volume
            well_log: Optional (B, C, L) well log curves
            well_mask: Optional (B, L) mask for well log
            task: Specific task to run (fault_detection, reservoir_prediction, lithology)
            return_features: If True, return encoded features

        Returns:
            Task-specific outputs dict
        """
        # If no well log, use seismic-only encoding for specific tasks
        if well_log is None:
            seismic_global, encoder_features = self.seismic_encoder(
                seismic, return_features=True
            )
            # Create dummy well features
            well_proj = torch.zeros(
                seismic.shape[0], self.config.hidden_dim,
                device=seismic.device,
            )
            fused = seismic_global  # Use seismic features directly
        else:
            # Full multi-modal encoding
            encoded = self.encode(
                seismic, well_log, well_mask, return_intermediate=True
            )
            seismic_proj = encoded["seismic_feat"]
            well_proj = encoded["well_feat"]
            fused = encoded["fused"]
            encoder_features = encoded["encoder_features"]
            well_sequence = encoded.get("well_sequence", None)

        if return_features and well_log is not None:
            return {
                "fused": fused,
                "seismic_feat": seismic_proj,
                "well_feat": well_proj,
                "encoder_features": encoder_features,
            }

        # Run task-specific prediction
        outputs = self.task_head(
            fused_features=fused,
            well_features=well_proj,
            encoder_features=encoder_features,
            seismic_input=seismic,
            seismic_trace_features=well_sequence,
            task=task,
        )

        return outputs

    def freeze_encoders(self):
        """Freeze encoder parameters for staged training."""
        for param in self.seismic_encoder.parameters():
            param.requires_grad = False
        for param in self.well_log_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze encoder parameters."""
        for param in self.seismic_encoder.parameters():
            param.requires_grad = True
        for param in self.well_log_encoder.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class OilGasModelForPretraining(nn.Module):
    """
    Oil & Gas model variant for self-supervised pretraining.

    Extends the base model with pretraining-specific heads:
    - Seismic reconstruction decoder (for MSM)
    - Well log reconstruction decoder (for MWM)
    - Contrastive projection heads (for CMCL)
    - Matching classifier (for SWM)
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = ModelConfig()

        self.config = config

        # Base encoders
        self.seismic_encoder = SeismicEncoder3D(
            in_channels=config.seismic_encoder.in_channels,
            stem_channels=config.seismic_encoder.stem_channels,
            embed_dim=config.seismic_encoder.embed_dim,
            depths=config.seismic_encoder.depths,
            num_heads=config.seismic_encoder.num_heads,
            window_size=config.seismic_encoder.window_size,
            patch_size=config.seismic_encoder.patch_size,
            mlp_ratio=config.seismic_encoder.mlp_ratio,
            dropout=config.seismic_encoder.dropout,
            use_checkpoint=config.seismic_encoder.use_checkpoint,
        )

        self.well_log_encoder = WellLogEncoder1D(
            num_curves=config.well_log_encoder.num_curves,
            stem_channels=config.well_log_encoder.stem_channels,
            kernel_sizes=config.well_log_encoder.kernel_sizes,
            embed_dim=config.well_log_encoder.embed_dim,
            num_layers=config.well_log_encoder.num_layers,
            num_heads=config.well_log_encoder.num_heads,
            mlp_ratio=config.well_log_encoder.mlp_ratio,
            dropout=config.well_log_encoder.dropout,
            max_seq_len=config.well_log_encoder.max_seq_len,
            use_physics_constraint=config.well_log_encoder.use_physics_constraint,
        )

        seis_dim = self.seismic_encoder.get_output_dim()
        well_dim = self.well_log_encoder.get_output_dim()
        hidden_dim = config.hidden_dim

        self.modality_proj = ModalityProjection(
            seismic_dim=seis_dim,
            well_log_dim=well_dim,
            common_dim=hidden_dim,
            dropout=config.fusion.dropout,
        )

        # Contrastive learning projection heads
        proj_dim = 128  # Smaller dim for contrastive learning
        self.seismic_proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.well_proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Matching classifier for SWM
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.fusion_module = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_cross_attention_heads=config.fusion.num_cross_attention_heads,
            num_fusion_layers=config.fusion.num_fusion_layers,
            dropout=config.fusion.dropout,
            use_gating=config.fusion.use_gating,
        )

    def encode_seismic(self, seismic: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode seismic volume."""
        return self.seismic_encoder(seismic, return_features=True)

    def encode_well_log(
        self, well_log: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode well log curves."""
        return self.well_log_encoder(well_log, mask=mask, return_sequence=True)

    def forward_cmcl(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Cross-Modal Contrastive Learning.

        Returns projected features for contrastive loss computation.
        """
        seis_global = self.seismic_encoder(seismic, return_features=False)
        well_global, _ = self.well_log_encoder(well_log, return_sequence=False)

        seis_proj, well_proj = self.modality_proj(seis_global, well_global)

        # Project to contrastive space
        z_seis = self.seismic_proj_head(seis_proj)
        z_well = self.well_proj_head(well_proj)

        # L2 normalize
        z_seis = nn.functional.normalize(z_seis, dim=-1)
        z_well = nn.functional.normalize(z_well, dim=-1)

        return z_seis, z_well

    def forward_swm(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for Seismic-Well Matching.

        Returns matching probability.
        """
        seis_global = self.seismic_encoder(seismic, return_features=False)
        well_global, _ = self.well_log_encoder(well_log, return_sequence=False)

        seis_proj, well_proj = self.modality_proj(seis_global, well_global)

        # Concatenate and classify
        combined = torch.cat([seis_proj, well_proj], dim=-1)
        match_prob = self.matching_head(combined)

        return match_prob

    def forward(
        self,
        seismic: torch.Tensor,
        well_log: Optional[torch.Tensor] = None,
        task: str = "cmcl",
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward for pretraining.

        Args:
            seismic: (B, 1, D, H, W)
            well_log: Optional (B, C, L)
            task: Pretraining task name (cmcl, swm, fusion)

        Returns:
            task-specific outputs
        """
        if task == "cmcl":
            z_seis, z_well = self.forward_cmcl(seismic, well_log)
            return {"seismic_embed": z_seis, "well_embed": z_well}

        elif task == "swm":
            match_prob = self.forward_swm(seismic, well_log)
            return {"match_prob": match_prob}

        elif task == "fusion":
            seis_global, enc_feats = self.seismic_encoder(seismic, return_features=True)
            well_global, well_seq = self.well_log_encoder(
                well_log, return_sequence=True
            )
            seis_proj, well_proj = self.modality_proj(seis_global, well_global)
            fused = self.fusion_module(seis_proj, well_proj)
            return {
                "fused": fused,
                "seismic_feat": seis_proj,
                "well_feat": well_proj,
                "encoder_features": enc_feats,
                "well_sequence": well_seq,
            }

        else:
            raise ValueError(f"Unknown pretraining task: {task}")
