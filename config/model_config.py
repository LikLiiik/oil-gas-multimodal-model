"""
Model Configuration Classes

Provides dataclass-based configuration with YAML loading support.
All configuration is centralized here for easy hyperparameter management.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class SeismicEncoderConfig:
    """Configuration for the 3D Seismic Encoder."""
    in_channels: int = 1
    stem_channels: int = 64
    embed_dim: int = 192
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    window_size: Tuple[int, int, int] = (7, 7, 7)
    patch_size: Tuple[int, int, int] = (2, 2, 2)
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_checkpoint: bool = True


@dataclass
class WellLogEncoderConfig:
    """Configuration for the 1D Well Log Encoder."""
    num_curves: int = 7
    stem_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    embed_dim: int = 192
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 512
    use_physics_constraint: bool = True


@dataclass
class FusionConfig:
    """Configuration for the Cross-Modal Fusion module."""
    hidden_dim: int = 384
    num_cross_attention_heads: int = 8
    num_fusion_layers: int = 2
    dropout: float = 0.1
    use_gating: bool = True


@dataclass
class PretrainingConfig:
    """Configuration for pretraining tasks."""
    # MSM
    msm_mask_ratio: float = 0.6
    msm_mask_block_size: Tuple[int, int, int] = (8, 16, 16)
    msm_decoder_depth: int = 4
    msm_decoder_embed_dim: int = 128
    msm_loss_weight: float = 1.0

    # MWM
    mwm_mask_ratio: float = 0.5
    mwm_mask_block_size: int = 16
    mwm_decoder_layers: int = 2
    mwm_loss_weight: float = 1.0

    # CMCL
    cmcl_temperature: float = 0.07
    cmcl_projection_dim: int = 256
    cmcl_loss_weight: float = 0.5

    # SWM
    swm_loss_weight: float = 0.3

    # Training
    epochs: int = 100
    warmup_epochs: int = 5
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.05
    scheduler: str = "cosine"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    seismic_volume_shape: Tuple[int, int, int] = (128, 256, 256)
    seismic_time_samples: int = 128
    seismic_inline_range: Tuple[int, int] = (0, 256)
    seismic_xline_range: Tuple[int, int] = (0, 256)
    seismic_sample_interval_ms: float = 2.0

    well_log_num_curves: int = 7
    well_log_curve_names: List[str] = field(
        default_factory=lambda: ["GR", "RT", "DEN", "POR", "AC", "SP", "CAL"]
    )
    well_log_sequence_length: int = 512
    well_log_depth_interval_m: float = 0.125

    batch_size: int = 8
    num_workers: int = 4
    train_ratio: float = 0.8


@dataclass
class FinetuningConfig:
    """Configuration for downstream task finetuning."""
    epochs: int = 50
    warmup_epochs: int = 3
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    freeze_encoder_epochs: int = 5


@dataclass
class TrainingConfig:
    """General training configuration."""
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 2
    log_interval: int = 50
    save_interval: int = 5
    eval_interval: int = 1
    early_stopping_patience: int = 10
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Loss weights for finetuning
    fault_bce_weight: float = 0.5
    fault_dice_weight: float = 0.5
    reservoir_mse_weight: float = 0.6
    reservoir_ssim_weight: float = 0.4
    lithology_ce_weight: float = 0.7
    lithology_focal_weight: float = 0.3


@dataclass
class ModelConfig:
    """Master configuration holding all sub-configs."""
    seismic_encoder: SeismicEncoderConfig = field(default_factory=SeismicEncoderConfig)
    well_log_encoder: WellLogEncoderConfig = field(default_factory=WellLogEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    hidden_dim: int = 384
    output_dim: int = 256

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        model_cfg = data.get("model", {})

        seismic_cfg = SeismicEncoderConfig(
            **model_cfg.get("seismic_encoder", {})
        )
        well_log_cfg = WellLogEncoderConfig(
            **model_cfg.get("well_log_encoder", {})
        )
        fusion_cfg = FusionConfig(
            **model_cfg.get("fusion", {})
        )

        return cls(
            seismic_encoder=seismic_cfg,
            well_log_encoder=well_log_cfg,
            fusion=fusion_cfg,
            hidden_dim=model_cfg.get("hidden_dim", 384),
            output_dim=model_cfg.get("output_dim", 256),
        )


def get_default_config() -> ModelConfig:
    """Return the default model configuration."""
    return ModelConfig()
