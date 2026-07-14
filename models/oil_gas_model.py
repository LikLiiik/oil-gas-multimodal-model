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
import numpy as np
from typing import Dict, List, Optional, Tuple

from .seismic_encoder import SeismicEncoder3D
from .ncs_seismic_encoder import NCSSeismicEncoder3D
from .well_log_encoder import WellLogEncoder1D
from .wlfm_well_log_encoder import WLFMWellLogEncoder1D
from .fusion_module import CrossModalFusion, ModalityProjection
from .prediction_heads import MultiTaskHead

# Support both relative and absolute imports
try:
    from ..config.model_config import ModelConfig
except ImportError:
    from config.model_config import ModelConfig


def _build_seismic_encoder(config) -> nn.Module:
    """Build seismic encoder based on config.backbone selection."""
    backbone = getattr(config.seismic_encoder, 'backbone', 'resnet3d')
    cfg = config.seismic_encoder

    if backbone == "ncs":
        scratch_kwargs = dict(
            in_channels=cfg.in_channels,
            img_size=getattr(cfg, 'img_size', (128, 256, 256)),
            embed_dim=cfg.embed_dim,
            num_layers=getattr(cfg, 'num_layers', 12),
            num_heads=getattr(cfg, 'num_heads', 3),
            mlp_ratio=getattr(cfg, 'mlp_ratio', 4.0),
            dropout=cfg.dropout,
            mode=getattr(cfg, 'ncs_mode', '2.5d'),
            use_checkpoint=getattr(cfg, 'use_checkpoint', True),
        )
        use_pretrained = getattr(cfg, 'use_pretrained', False)
        pretrained_name = getattr(cfg, 'ncs_pretrained', '') or ''
        if use_pretrained and pretrained_name:
            try:
                return NCSSeismicEncoder3D.from_pretrained(
                    pretrained_name=pretrained_name,
                    **scratch_kwargs,
                )
            except Exception:
                pass
        return NCSSeismicEncoder3D(**scratch_kwargs)
    elif backbone == "swin3d":
        from .seismic_encoder import SwinSeismicEncoder3D
        return SwinSeismicEncoder3D(
            in_channels=cfg.in_channels,
            stem_channels=cfg.stem_channels,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            patch_size=cfg.patch_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_checkpoint=cfg.use_checkpoint,
        )
    else:  # resnet3d (default)
        return SeismicEncoder3D(
            in_channels=cfg.in_channels,
            stem_channels=cfg.stem_channels,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            patch_size=cfg.patch_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_checkpoint=cfg.use_checkpoint,
        )


def _build_well_log_encoder(config) -> nn.Module:
    """Build well log encoder based on config.backbone selection."""
    backbone = getattr(config.well_log_encoder, 'backbone', 'cnn_transformer')
    cfg = config.well_log_encoder

    if backbone == "wlfm":
        wlfm_kwargs = dict(
            num_curves=cfg.num_curves,
            max_seq_len=cfg.max_seq_len,
            embed_dim=cfg.embed_dim,
            vq_embed_dim=getattr(cfg, 'wlfm_vq_embed_dim', 256),
            num_embeddings=getattr(cfg, 'wlfm_num_embeddings', 512),
            patch_len=getattr(cfg, 'wlfm_patch_len', 64),
            patch_stride=getattr(cfg, 'wlfm_patch_stride', 32),
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_physics_constraint=cfg.use_physics_constraint,
        )
        pretrained_path = getattr(cfg, 'wlfm_pretrained_path', None) or ''
        if pretrained_path.strip():
            return WLFMWellLogEncoder1D.from_pretrained(
                pretrained_path=pretrained_path,
                **wlfm_kwargs,
            )
        return WLFMWellLogEncoder1D(**wlfm_kwargs)
    else:  # cnn_transformer (default)
        return WellLogEncoder1D(
            num_curves=cfg.num_curves,
            stem_channels=cfg.stem_channels,
            kernel_sizes=cfg.kernel_sizes,
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
            use_physics_constraint=cfg.use_physics_constraint,
        )


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
        # Encoders (select by backbone config)
        # ==========================================

        # 3D Seismic Encoder
        self.seismic_encoder = _build_seismic_encoder(config)

        # 1D Well Log Encoder
        self.well_log_encoder = _build_well_log_encoder(config)

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

        # Base encoders (select by backbone config)
        self.seismic_encoder = _build_seismic_encoder(config)
        self.well_log_encoder = _build_well_log_encoder(config)

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
        self.fusion_proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Matching classifier for SWM (logits; use BCEWithLogitsLoss)
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.fusion_module = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_cross_attention_heads=config.fusion.num_cross_attention_heads,
            num_fusion_layers=config.fusion.num_fusion_layers,
            dropout=config.fusion.dropout,
            use_gating=config.fusion.use_gating,
        )

    def encode_seismic(self, seismic: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode seismic volume. Returns (global_features, encoder_features_list)."""
        result = self.seismic_encoder(seismic, return_features=True)
        if isinstance(result, tuple):
            return result
        return result, None

    def encode_well_log(
        self, well_log: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode well log curves. Returns (global_features, sequence_features)."""
        global_feat, seq = self.well_log_encoder(well_log, mask=mask, return_sequence=True)
        return global_feat, seq

    def forward_cmcl(
        self,
        seismic: torch.Tensor,
        well_log: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Cross-Modal Contrastive Learning.

        Returns projected features for contrastive loss computation.
        """
        seis_result = self.seismic_encoder(seismic, return_features=False)
        seis_global = seis_result[0] if isinstance(seis_result, tuple) else seis_result

        well_result = self.well_log_encoder(well_log, return_sequence=False)
        well_global = well_result[0] if isinstance(well_result, tuple) else well_result

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
        seis_result = self.seismic_encoder(seismic, return_features=False)
        seis_global = seis_result[0] if isinstance(seis_result, tuple) else seis_result

        well_result = self.well_log_encoder(well_log, return_sequence=False)
        well_global = well_result[0] if isinstance(well_result, tuple) else well_result

        seis_proj, well_proj = self.modality_proj(seis_global, well_global)

        # Concatenate and classify (return probability for API compatibility)
        combined = torch.cat([seis_proj, well_proj], dim=-1)
        match_logits = self.matching_head(combined)
        return torch.sigmoid(match_logits)

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


# ==============================================================================
# Industrial Inference Engine
# ==============================================================================

class IndustrialInferenceEngine:
    """
    Handles real-world deployment scenarios that differ from training:

    1. Variable-size seismic → sliding window inference
    2. Variable-length well logs → dynamic patching
    3. Missing modality → single-modality fallback
    4. Missing curves → curve dropout + imputation
    5. Few-shot adaptation → rapid fine-tuning on new wells

    Usage:
        engine = IndustrialInferenceEngine(model)

        # Scenario A: Full data, different size
        out = engine.infer(seismic_500x800, well_log_200)

        # Scenario B: Only seismic (no well)
        out = engine.infer(seismic=big_volume, well_log=None)

        # Scenario C: Only well log (no seismic)
        out = engine.infer(seismic=None, well_log=well_curves)

        # Scenario D: Missing DT curve
        out = engine.infer(seismic=vol, well_log=partial_log,
                          available_curves=['GR','RT','RHOB','NPHI'])
    """

    def __init__(self, model: OilGasModel, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()

    def _build_dummy_encoder_features(
        self, seismic: Optional[torch.Tensor], device: torch.device
    ) -> List[torch.Tensor]:
        """Build dummy multi-scale encoder features for skip connections."""
        out_ch = getattr(self.model.seismic_encoder, 'out_channels', [192, 96, 192, 96])
        features = []
        if seismic is not None:
            B, _, D, H, W = seismic.shape
            for i, ch in enumerate(out_ch):
                factor = 2 ** (i + 1)
                features.append(torch.zeros(B, ch, max(1, D//factor),
                               max(1, H//factor), max(1, W//factor), device=device))
        else:
            # No seismic — all zeros
            for i, ch in enumerate(out_ch):
                features.append(torch.zeros(1, ch, 2, 2, 2, device=device))
        return features

    @torch.no_grad()
    def infer(
        self,
        seismic: Optional[torch.Tensor] = None,
        well_log: Optional[torch.Tensor] = None,
        well_mask: Optional[torch.Tensor] = None,
        available_curves: Optional[List[str]] = None,
        task: Optional[str] = None,
        seismic_tile_size: Tuple[int, int, int] = (32, 64, 64),
    ) -> Dict[str, torch.Tensor]:
        """
        Unified industrial inference supporting all missing-data scenarios.

        Args:
            seismic: (B, 1, D, H, W) or None if missing
            well_log: (B, C, L) or None if missing
            well_mask: (B, L) valid depth mask
            available_curves: Names of curves present in well_log
            task: 'fault_detection' | 'reservoir_prediction' | 'lithology'
            seismic_tile_size: Tile size for sliding window (large volumes)

        Returns:
            Task-specific prediction dict
        """
        B = 1
        hidden_dim = self.model.config.hidden_dim

        # ---- Case 1: Both modalities available (full fusion) ----
        if seismic is not None and well_log is not None:
            seismic = seismic.to(self.device)
            well_log = well_log.to(self.device)

            # Handle variable-size seismic
            seis_encoder = self.model.seismic_encoder
            train_shape = getattr(seis_encoder, 'img_size', (32, 32, 32))
            if list(seismic.shape[2:]) != list(train_shape):
                seis_feat = seis_encoder.infer_variable_size(
                    seismic, tile_size=seismic_tile_size
                )
            else:
                seis_feat, _ = seis_encoder(seismic, return_features=False)
                if isinstance(seis_feat, tuple):
                    seis_feat = seis_feat[0]

            # Handle variable-length well logs (with optional missing curves)
            wl_encoder = self.model.well_log_encoder
            well_seq = None  # ensure defined

            # If curves are missing, pad to full curve count with zeros
            if available_curves is not None and len(available_curves) < wl_encoder.num_curves:
                B, _, L = well_log.shape
                padded_wl = torch.zeros(B, wl_encoder.num_curves, L,
                                        device=well_log.device, dtype=well_log.dtype)
                curve_mask = torch.zeros(B, wl_encoder.num_curves, device=well_log.device)
                for i, name in enumerate(available_curves):
                    if name in wl_encoder.curve_names and i < well_log.shape[1]:
                        idx = wl_encoder.curve_names.index(name)
                        padded_wl[:, idx, :] = well_log[:, i, :]
                        curve_mask[:, idx] = 1.0
                well_log_input = padded_wl
            else:
                well_log_input = well_log
                curve_mask = None

            if hasattr(wl_encoder, 'infer_variable_length'):
                well_feat, well_seq = wl_encoder.infer_variable_length(well_log_input)
            else:
                well_feat, well_seq = wl_encoder(well_log_input, mask=well_mask, return_sequence=True)

            # Project and fuse
            seis_proj, well_proj = self.model.modality_proj(seis_feat, well_feat)
            fused = self.model.fusion_module(seis_proj, well_proj)

        # ---- Case 2: Seismic only ----
        elif seismic is not None:
            seismic = seismic.to(self.device)
            seis_feat = self.model.seismic_encoder.infer_variable_size(
                seismic, tile_size=seismic_tile_size
            )
            well_proj = torch.zeros(B, hidden_dim, device=self.device)
            fused = seis_feat

        # ---- Case 3: Well log only ----
        elif well_log is not None:
            well_log = well_log.to(self.device)
            wl_encoder = self.model.well_log_encoder
            if available_curves and hasattr(wl_encoder, 'infer_missing_curves'):
                well_feat, well_seq = wl_encoder.infer_missing_curves(
                    well_log, available_curves
                )
            elif hasattr(wl_encoder, 'infer_variable_length'):
                well_feat, well_seq = wl_encoder.infer_variable_length(well_log)
            else:
                well_feat, well_seq = wl_encoder(
                    well_log, mask=well_mask, return_sequence=True
                )

            well_proj = self.model.modality_proj.well_log_proj(well_feat)
            fused = well_proj

        else:
            return {"error": "No input provided"}

        # ---- Run task head ----
        seismic_input = seismic if seismic is not None else None
        encoder_features = self._build_dummy_encoder_features(seismic_input, fused.device)
        well_sequence = well_seq if (well_log is not None and 'well_seq' in dir()) else None

        # For lithology, pass sequence features (B,L,D) else pooled (B,D)
        wf = well_sequence if (task == "lithology" and well_sequence is not None) else well_proj

        outputs = self.model.task_head(
            fused_features=fused,
            well_features=wf,
            encoder_features=encoder_features,
            seismic_input=seismic_input,
            seismic_trace_features=well_sequence,
            task=task,
        )

        return outputs

    @torch.no_grad()
    def infer_full_volume(
        self,
        segy_path: str,
        task: str = "fault_detection",
        tile_size: Tuple[int, int, int] = (32, 64, 64),
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Process an entire SEG-Y volume, tile by tile, producing a full prediction.

        Args:
            segy_path: Path to SEG-Y file
            task: Task name
            tile_size: Processing tile size
            batch_size: Tiles per batch

        Returns:
            prediction volume (numpy array)
        """
        from data.volve_dataset import SEGYLoader
        segy = SEGYLoader(segy_path)

        n_il = segy.inline_max - segy.inline_min + 1
        n_xl = segy.xline_max - segy.xline_min + 1
        n_samples = segy.num_samples

        output = np.zeros((n_il, n_xl, n_samples), dtype=np.float32)

        td, th, tw = tile_size
        sd, sh, sw = max(1, td//2), max(1, th//2), max(1, tw//2)

        tiles = []
        positions = []

        for d in range(0, n_samples, sd):
            for h in range(0, n_il, sh):
                for w in range(0, n_xl, sw):
                    d_end = min(n_samples, d + td)
                    h_end = min(n_il, h + th)
                    w_end = min(n_xl, w + tw)

                    tile = segy.read_volume(
                        il_range=(segy.inline_min + h, segy.inline_min + h_end),
                        xl_range=(segy.xline_min + w, segy.xline_min + w_end),
                    )

                    # Pad
                    if tile.shape != (th, tw, td):
                        pad_h = th - tile.shape[0]
                        pad_w = tw - tile.shape[1]
                        pad_d = td - tile.shape[2]
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, pad_d)))

                    tile_t = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                    tiles.append(tile_t)
                    positions.append((d, d_end, h, h_end, w, w_end))

                    if len(tiles) >= batch_size:
                        self._process_tile_batch(tiles, positions, output)
                        tiles = []
                        positions = []

        if tiles:
            self._process_tile_batch(tiles, positions, output)

        return output

    def _process_tile_batch(
        self, tiles: List, positions: List, output: np.ndarray
    ):
        """Process a batch of tiles and merge into output volume."""
        batch = torch.cat(tiles, dim=0).to(self.device)
        results = self.infer(seismic=batch, task="fault_detection")

        if "fault_prob" in results:
            probs = results["fault_prob"].cpu().numpy()
            for i, (d_start, d_end, h_start, h_end, w_start, w_end) in enumerate(positions):
                prob = probs[i, 0]
                # Crop to valid region and merge
                valid = prob[:d_end-d_start, :h_end-h_start, :w_end-w_start]
                existing = output[h_start:h_end, w_start:w_end, d_start:d_end].transpose(2, 0, 1)
                existing = np.maximum(existing, valid)  # max merge
                output[h_start:h_end, w_start:w_end, d_start:d_end] = existing.transpose(1, 2, 0)

    def few_shot_transfer(
        self,
        new_seismic: torch.Tensor,
        new_well_logs: List[torch.Tensor],
        new_labels: List[torch.Tensor],
        task: str = "lithology",
        n_steps: int = 100,
        lr: float = 1e-4,
    ) -> "IndustrialInferenceEngine":
        """
        Rapidly adapt the model to a new field with few labeled wells.

        Typical scenario: You have a pretrained model and 2-3 newly drilled
        wells with core/log labels. This fine-tunes the encoder on the new
        data in minutes instead of hours.

        Args:
            new_seismic: (1, 1, D, H, W) seismic from new field
            new_well_logs: List of (C, L) labeled well logs
            new_labels: List of (L,) corresponding labels
            task: Task type
            n_steps: Adaptation steps (small = fast)
            lr: Learning rate

        Returns:
            self (adapted engine)
        """
        self.model.train()

        # Only train encoder params (keep task heads frozen)
        for p in self.model.task_head.parameters():
            p.requires_grad = False
        for p in self.model.seismic_encoder.parameters():
            p.requires_grad = True
        for p in self.model.well_log_encoder.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(
            list(self.model.seismic_encoder.parameters()) +
            list(self.model.well_log_encoder.parameters()),
            lr=lr,
        )

        criterion = nn.MSELoss() if task == "reservoir_prediction" else nn.CrossEntropyLoss()

        for step in range(n_steps):
            opt.zero_grad()
            total_loss = 0.0

            for wl, lbl in zip(new_well_logs, new_labels):
                wl = wl.unsqueeze(0).to(self.device)  # (1, C, L)
                lbl = lbl.unsqueeze(0).to(self.device)  # (1, L)

                encoded = self.model.encode(
                    new_seismic.to(self.device), wl, return_intermediate=True
                )
                well_seq = encoded.get("well_sequence")

                if task == "lithology" and well_seq is not None:
                    pred = nn.Linear(well_seq.shape[-1], int(lbl.max().item()) + 1)
                    pred = pred.to(self.device)
                    loss = criterion(
                        pred(well_seq).reshape(-1, pred.out_features),
                        lbl.reshape(-1).long(),
                    )
                else:
                    loss = criterion(
                        encoded["fused"].squeeze(), lbl.float().mean(dim=-1)
                    )

                total_loss += loss

            total_loss.backward()
            opt.step()

            if step % 20 == 0:
                pass  # logger.info(f"Few-shot step {step}: loss={total_loss.item():.4f}")

        self.model.eval()
        return self
