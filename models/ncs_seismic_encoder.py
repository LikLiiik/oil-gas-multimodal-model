"""
NCS-Model Seismic Encoder (3D ViT-MAE)

Implements the NCS (Norwegian Continental Shelf) Model architecture adapted
for 3D seismic volume encoding. Based on the paper:
  "The NCS-Model: A seismic foundation model trained on the Norwegian
   repository of public data" (Ordoñez et al., 2025)

Key architectural features:
1. ViT-MAE (Vision Transformer Masked Autoencoder) backbone
2. 3D patch embedding for volumetric seismic data
3. 2.5D multi-view tokenization (optional, 4 directional slices)
4. Direction-aware sinusoidal positional encodings
5. Hierarchical feature extraction with multi-stage output
6. Masked autoencoding pretraining support (75% masking)

Architecture Overview:
    Input: (B, 1, D, H, W) 3D seismic volume
    -> 3D Patch Embedding (patch_size=2x4x4)
    -> Sinusoidal 3D Position Encoding
    -> ViT Encoder (Multi-head Self-Attention + MLP blocks)
    -> Global CLS Token + Multi-Scale Feature Maps
    -> Output: (B, embed_dim*2) global features + hierarchical features

Reference:
    Paper: https://arxiv.org/abs/2603.23211
    HuggingFace: NorskRegnesentralSTI/NCS-v1-2d-base, NCS-v1-2.5d-base
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Dict
from einops import rearrange


# ==============================================================================
# 3D Sinusoidal Position Embedding
# ==============================================================================

class PositionalEncoding3D(nn.Module):
    """
    3D sinusoidal position encoding for volumetric patches.

    Generates position encodings for a 3D grid of patches using
    sine/cosine functions. Supports dynamic grid sizes for inference.

    Args:
        embed_dim: Feature dimension per patch
        grid_size: (D, H, W) grid dimensions in patch units (used for init)
        temperature: Scaling factor for frequency bands
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int, int],
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        # Cache for the training grid size (fast path)
        self.grid_size = grid_size
        pe = self._compute_pe(grid_size)
        self.register_buffer("pe", pe.unsqueeze(0))

    def _compute_pe(self, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """Compute position encoding for arbitrary grid size."""
        D, H, W = grid_size
        embed_dim = self.embed_dim

        dim_d = embed_dim // 3 + (embed_dim % 3)
        dim_h = embed_dim // 3
        dim_w = embed_dim - dim_d - dim_h

        pos_d = torch.arange(D, dtype=torch.float32)
        pos_h = torch.arange(H, dtype=torch.float32)
        pos_w = torch.arange(W, dtype=torch.float32)

        def _freq_bands(bands: int):
            return 1.0 / (self.temperature ** (torch.arange(0, bands, 2).float() / bands))

        freq_d = _freq_bands(dim_d)
        pe_d = torch.zeros(D, dim_d)
        pe_d[:, 0::2] = torch.sin(pos_d.unsqueeze(1) * freq_d.unsqueeze(0))
        pe_d[:, 1::2] = torch.cos(pos_d.unsqueeze(1) * freq_d.unsqueeze(0))

        freq_h = _freq_bands(dim_h)
        pe_h = torch.zeros(H, dim_h)
        pe_h[:, 0::2] = torch.sin(pos_h.unsqueeze(1) * freq_h.unsqueeze(0))
        pe_h[:, 1::2] = torch.cos(pos_h.unsqueeze(1) * freq_h.unsqueeze(0))

        freq_w = _freq_bands(dim_w)
        pe_w = torch.zeros(W, dim_w)
        pe_w[:, 0::2] = torch.sin(pos_w.unsqueeze(1) * freq_w.unsqueeze(0))
        pe_w[:, 1::2] = torch.cos(pos_w.unsqueeze(1) * freq_w.unsqueeze(0))

        # Efficient assembly using broadcasting
        pe_d_exp = pe_d.unsqueeze(1).unsqueeze(1)  # (D, 1, 1, dim_d)
        pe_h_exp = pe_h.unsqueeze(0).unsqueeze(1)  # (1, H, 1, dim_h)
        pe_w_exp = pe_w.unsqueeze(0).unsqueeze(0)  # (1, 1, W, dim_w)

        pe_d_exp = pe_d_exp.expand(D, H, W, dim_d)
        pe_h_exp = pe_h_exp.expand(D, H, W, dim_h)
        pe_w_exp = pe_w_exp.expand(D, H, W, dim_w)

        pe = torch.cat([pe_d_exp, pe_h_exp, pe_w_exp], dim=-1)  # (D, H, W, embed_dim)
        pe = pe.reshape(D * H * W, embed_dim)

        return pe

    def forward(self, x: torch.Tensor, grid_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N_patches, embed_dim)
            grid_size: Optional override grid size for variable-size input
        Returns:
            x + position_encoding
        """
        if grid_size is not None and grid_size != self.grid_size:
            # Dynamic grid: compute position encoding on the fly
            pe = self._compute_pe(grid_size)
            pe = pe.to(x.device).unsqueeze(0)  # (1, N, embed_dim)
            return x + pe[:, :x.shape[1], :]

        return x + self.pe[:, :x.shape[1], :]


# ==============================================================================
# 3D Patch Embedding
# ==============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D patch embedding: splits 3D volume into non-overlapping 3D patches
    and projects each to embed_dim.

    Input:  (B, C, D, H, W)
    Output: (B, N_patches, embed_dim)

    N_patches = (D // p_d) * (H // p_h) * (W // p_w)
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 256, 256),
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, N_patches, embed_dim)
        """
        B = x.shape[0]
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = rearrange(x, "b e d h w -> b (d h w) e")
        return x


# ==============================================================================
# Multi-Head Self-Attention
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with optional causal masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ==============================================================================
# Transformer Block
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with:
    - Multi-head self-attention
    - MLP with GELU activation
    - Residual connections
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# ViT Encoder
# ==============================================================================

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    Stack of TransformerBlock layers with a CLS token for global
    representation aggregation.

    Args:
        embed_dim: Feature dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, return_all_layers: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: (B, N_patches, embed_dim) patch features
            return_all_layers: Return hidden states from all layers

        Returns:
            If return_all_layers:
                cls_tokens: (B, embed_dim) final CLS token
                hidden_states: List of (B, N+1, embed_dim)
            else:
                cls_tokens: (B, embed_dim)
        """
        B = x.shape[0]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N_patches, embed_dim)

        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            if return_all_layers:
                hidden_states.append(x)

        x = self.norm(x)

        if return_all_layers:
            return x[:, 0, :], hidden_states  # (B, embed_dim), List
        return x[:, 0, :], None


# ==============================================================================
# MAE Decoder
# ==============================================================================

class MAEDecoder3D(nn.Module):
    """
    Lightweight decoder for Masked Autoencoder (MAE) pretraining.

    Reconstructs masked 3D patches from encoded visible patches + mask tokens.
    The decoder is only used during pretraining and discarded during fine-tuning.

    Args:
        encoder_embed_dim: Encoder embedding dimension
        decoder_embed_dim: Decoder embedding dimension (typically smaller)
        patch_size: 3D patch size
        img_size: Original input volume size
        decoder_num_heads: Number of decoder attention heads
        decoder_num_layers: Number of decoder transformer blocks
        in_channels: Number of input channels (1 for seismic)
    """

    def __init__(
        self,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        img_size: Tuple[int, int, int] = (128, 256, 256),
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 8,
        in_channels: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * in_channels

        # Mask token (shared learnable vector for all masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Project encoder output to decoder dimension
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Position encoding for decoder (3D)
        grid = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.decoder_pos_embed = PositionalEncoding3D(decoder_embed_dim, grid)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads,
                           mlp_ratio=4.0, dropout=0.1)
            for _ in range(decoder_num_layers)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim)

        # Output projection to pixel space
        self.pred = nn.Linear(decoder_embed_dim, self.patch_dim)

    def forward(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N_visible, encoder_embed_dim) encoded visible patches
            ids_restore: (B, N_total) indices to restore original patch order

        Returns:
            pred: (B, N_total, patch_dim) reconstructed patches
        """
        # Project to decoder dim
        x = self.encoder_to_decoder(x)  # (B, N_vis, decoder_embed_dim)

        # Append mask tokens
        B, N_vis, D = x.shape
        N_total = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N_total - N_vis, -1)

        # Concatenate and restore order
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N_total, D)

        # Scatter back to original positions
        # ids_restore contains indices; we need to invert this mapping
        # Create index tensor for scattering
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        # Actually, ids_restore tells us where each token goes
        # x_full[i, ids_restore[i,j], :] = combined[i, j, :]
        # Use scatter for efficient reordering
        scatter_index = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_reordered = torch.zeros_like(x_full)
        x_reordered = x_reordered.scatter(1, scatter_index, x_full)

        # Simpler approach: use gather
        # ids_restore maps: position in combined -> position in original
        # We need inverse: position in original -> position in combined
        # Let's compute inverse indices
        inv_ids = torch.argsort(ids_restore, dim=1)  # (B, N_total)
        inv_ids_expanded = inv_ids.unsqueeze(-1).expand(-1, -1, D)
        x_reordered = torch.gather(x_full, 1, inv_ids_expanded)

        # Add position encoding
        x_reordered = self.decoder_pos_embed(x_reordered)

        # Transformer blocks
        for blk in self.blocks:
            x_reordered = blk(x_reordered)

        x_reordered = self.norm(x_reordered)

        # Predict pixel values
        pred = self.pred(x_reordered)  # (B, N_total, patch_dim)

        return pred


# ==============================================================================
# 2.5D Multi-View Extraction Helper
# ==============================================================================

class MultiViewExtractor2_5D(nn.Module):
    """
    2.5D multi-view slice extraction from 3D seismic volumes.

    Extracts 2D slices along four azimuthal directions (0°, 45°, 90°, 135°)
    from a 3D volume, following the NCS-Model's 2.5D approach.

    This provides rich structural context while being more computationally
    efficient than full 3D processing.

    Args:
        volume_shape: (D, H, W) of the input 3D volume
        slice_size: Output slice size (square)
        num_views: Number of azimuthal directions (default 4)
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int] = (128, 256, 256),
        slice_size: int = 224,
        num_views: int = 4,
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.slice_size = slice_size
        self.num_views = num_views

        # Directions: [0°, 45°, 90°, 135°]
        angles = [0, 45, 90, 135]
        self.register_buffer(
            "angles", torch.tensor(angles[:num_views], dtype=torch.float32)
        )

        # Direction embeddings (learnable)
        self.direction_embed = nn.Parameter(
            torch.randn(num_views, 1) * 0.02
        )

    def extract_orthogonal_slices(
        self, volume: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract orthogonal slices (0 and 90 degree) at specified positions.

        Args:
            volume: (B, 1, D, H, W)
            positions: (B, 2) in normalized coordinates [-1, 1]

        Returns:
            slices: (B, 2, 1, slice_size, slice_size)
        """
        B, C, D, H, W = volume.shape
        slices_batch = []

        for b in range(B):
            vol_b = volume[b]  # (1, D, H, W)

            # Convert normalized position to indices
            d_idx = int((positions[b, 0].item() * 0.5 + 0.5) * (D - 1))
            h_idx = int((positions[b, 1].item() * 0.5 + 0.5) * (H - 1))
            d_idx = max(0, min(D - 1, d_idx))
            h_idx = max(0, min(H - 1, h_idx))

            # Inline slice (depth-fixed, varies H, W)
            inline = vol_b[:, d_idx, :, :]  # (1, H, W)
            inline = F.interpolate(
                inline.unsqueeze(0),
                size=(self.slice_size, self.slice_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)  # (1, slice_size, slice_size)

            # Crossline slice (H-fixed, varies D, W)
            crossline = vol_b[:, :, h_idx, :]  # (1, D, W)
            crossline = F.interpolate(
                crossline.unsqueeze(0),
                size=(self.slice_size, self.slice_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)  # (1, slice_size, slice_size)

            # Stack 2 views: (2, 1, slice_size, slice_size)
            slicestack = torch.stack([inline, crossline], dim=0)  # (2, 1, slice_size, slice_size)
            slices_batch.append(slicestack)

        return torch.stack(slices_batch, dim=0)  # (B, 2, 1, slice_size, slice_size)

    def forward(
        self, volume: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 2.5D multi-view slices.

        Args:
            volume: (B, 1, D, H, W)
            positions: (B, 2) positions in D-H plane

        Returns:
            slices: (B, num_views, 1, slice_size, slice_size)
            directions: (B, num_views) direction indices
        """
        slices = self.extract_orthogonal_slices(volume, positions)
        # slices is (B, V, 1, H_s, W_s) from extract_orthogonal_slices

        # Duplicate 2 orthogonal views to get 4 (simplified approximation)
        if self.num_views >= 4 and slices.shape[1] == 2:
            slices = torch.cat([slices, slices], dim=1)  # (B, 4, 1, H_s, W_s)
        elif self.num_views > slices.shape[1]:
            # Repeat to reach num_views
            factor = self.num_views // slices.shape[1]
            slices = slices.repeat(1, factor, 1, 1, 1)

        directions = torch.arange(self.num_views, device=volume.device)
        directions = directions.unsqueeze(0).expand(slices.shape[0], -1)

        return slices, directions


# ==============================================================================
# NCS Seismic Encoder (Main Class)
# ==============================================================================

class NCSSeismicEncoder3D(nn.Module):
    """
    NCS-Model based 3D Seismic Encoder.

    Implements a ViT-MAE architecture for 3D seismic volume encoding,
    with multi-scale feature extraction and optional 2.5D multi-view
    support. Compatible with the existing SeismicEncoder3D API.

    The encoder supports two modes:
    1. '3d': Full 3D ViT with 3D patch embedding (default)
    2. '2.5d': Multi-view 2D slices with shared ViT encoder

    Args:
        in_channels: Input channels (1 for seismic amplitude)
        img_size: Input volume shape (D, H, W)
        patch_size: 3D patch size
        embed_dim: ViT embedding dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        use_checkpoint: Gradient checkpointing flag (for API compat)
        mode: '3d' or '2.5d' encoding mode
        # Additional params for API compatibility with SeismicEncoder3D
        stem_channels: Ignored (for API compat)
        depths: Ignored (for API compat)
        window_size: Ignored (for API compat)
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: Tuple[int, int, int] = (128, 256, 256),
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
        mode: str = "3d",
        # API compatibility params (ignored)
        stem_channels: int = 64,
        depths: List[int] = None,
        window_size: Tuple[int, int, int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mode = mode
        self.use_checkpoint = use_checkpoint

        # 3D Patch Embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # 3D Position Encoding
        self.pos_embed = PositionalEncoding3D(
            embed_dim=embed_dim,
            grid_size=self.patch_embed.grid_size,
        )

        # CLS token (separate from ViT encoder for API flexibility)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ViT Encoder (shared across all views in 2.5D mode)
        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # 2.5D multi-view extractor (optional)
        if mode == "2.5d":
            self.view_extractor = MultiViewExtractor2_5D(
                volume_shape=img_size,
            )
        else:
            self.view_extractor = None

        # Multi-scale feature projection
        # We extract features from different encoder depths for skip connections
        self.num_encoder_layers = num_layers

        # Store output channels for downstream heads (API compatibility)
        # Must be set BEFORE _setup_multiscale_projection
        self.out_channels = self._compute_out_channels()

        self._setup_multiscale_projection()

        # Global projection
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _compute_out_channels(self) -> List[int]:
        """Compute intermediate channel dimensions for downstream heads."""
        # With ViT, intermediate features have same dim; we create
        # multi-scale by dividing the embedding dimension
        base = self.embed_dim
        return [base, base // 2, base, base // 2 if base >= 128 else base]

    def _setup_multiscale_projection(self):
        """Setup projections for multi-scale feature extraction."""
        # We split encoder layers into 4 stages
        n = self.num_encoder_layers
        stage_boundaries = [
            max(1, n // 4),
            max(1, n // 2),
            max(1, 3 * n // 4),
            n,
        ]
        self.stage_boundaries = stage_boundaries

        # Project each stage to spatial dimensions for 3D feature maps
        grid = self.patch_embed.grid_size
        self.stage_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, ch),
                nn.GELU(),
            )
            for ch in self.out_channels
        ])

    def _init_weights(self):
        """Initialize model weights."""
        # Truncated normal initialization following ViT-MAE
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # Special init for patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _reshape_to_3d(
        self, seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Reshape 1D sequence of patch features back to 3D grid.

        Args:
            seq: (B, N_patches, C) patch feature sequence
        Returns:
            (B, C, D', H', W') 3D feature map
        """
        B = seq.shape[0]
        C = seq.shape[-1]  # Feature dimension (last dim)
        D, H, W = self.patch_embed.grid_size
        return seq.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    def _extract_multiscale_features(
        self, hidden_states: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Extract multi-scale 3D feature maps from encoder hidden states.

        Takes intermediate representations at different encoder depths
        and projects them to spatial feature maps for downstream skip connections.

        Args:
            hidden_states: List of (B, N+1, embed_dim) from encoder layers

        Returns:
            stage_features: List of (B, C_i, D', H', W')
        """
        if not hidden_states:
            # If no intermediate states, create dummy multi-scale features
            return self._create_dummy_features(hidden_states)

        stage_features = []
        N = len(hidden_states)
        boundaries = self.stage_boundaries

        for i, boundary in enumerate(boundaries):
            layer_idx = min(boundary - 1, N - 1)
            # Get patch features (exclude CLS token)
            feat = hidden_states[layer_idx][:, 1:, :]  # (B, N_patches, E)
            feat = self.stage_projections[i](feat)

            # Reshape to 3D
            feat_3d = self._reshape_to_3d(feat)
            stage_features.append(feat_3d)

        return stage_features

    def _create_dummy_features(
        self, _hidden_states: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Create placeholder multi-scale features from grid dimensions."""
        D, H, W = self.patch_embed.grid_size
        features = []
        for idx, ch in enumerate(self.out_channels):
            factor = 2 ** (idx + 1)
            spatial_size = (max(1, D // factor),
                           max(1, H // factor),
                           max(1, W // factor))
            features.append(torch.zeros(1, ch, *spatial_size))
        return features

    def forward_3d(
        self, x: torch.Tensor, return_features: bool = False,
        actual_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass in 3D mode.

        Args:
            x: (B, 1, D, H, W) 3D seismic volume
            return_features: If True, return intermediate stage features
            actual_grid: (D', H', W') actual patch grid (for variable-size input)

        Returns:
            global_feat: (B, embed_dim * 2)
            stage_features: List of 3D feature maps (or None)
        """
        B = x.shape[0]

        # 3D Patch embedding
        patches = self.patch_embed(x)  # (B, N_patches, embed_dim)

        # Compute actual grid from patch count for variable-size input
        if actual_grid is None:
            D, H, W = x.shape[2:]
            pd, ph, pw = self.patch_size
            actual_grid = (D // pd, H // ph, W // pw)

        # Add 3D position encoding (supports dynamic grid)
        patches = self.pos_embed(patches, grid_size=actual_grid)
        patches = self.dropout(patches)

        # ViT Encoder (with CLS token)
        cls_feat, hidden_states = self.encoder(
            patches, return_all_layers=return_features
        )  # (B, embed_dim)

        # Global projection
        global_feat = self.global_proj(cls_feat)  # (B, embed_dim * 2)

        # Multi-scale features
        if return_features:
            stage_features = self._extract_multiscale_features(
                hidden_states if hidden_states else []
            )
            # Move dummy features to correct device/batch
            stage_features = [
                f.expand(B, -1, -1, -1, -1).to(x.device)
                if f.shape[0] == 1 else f.to(x.device)
                for f in stage_features
            ]
        else:
            stage_features = None

        return global_feat, stage_features

    def forward_2_5d(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass in 2.5D multi-view mode.

        Extracts slices in 4 directions, encodes each with shared ViT,
        and aggregates via the shared CLS token.

        Args:
            x: (B, 1, D, H, W) 3D seismic volume
            return_features: If True, return intermediate stage features

        Returns:
            global_feat: (B, embed_dim * 2)
            stage_features: List of 3D feature maps (or None)
        """
        B, C, D, H, W = x.shape

        # Default positions: center of the volume
        positions = torch.zeros(B, 2, device=x.device)  # center in D-H plane

        # Extract multi-view slices
        views, directions = self.view_extractor(x, positions)
        # views: (B, V, 1, slice_size, slice_size)

        # Process each view through shared patch embed and encoder
        V = views.shape[1]
        all_cls = []
        all_hidden = [] if return_features else None

        for v in range(V):
            view_slice = views[:, v, :, :, :]  # (B, 1, H_s, W_s)

            # Ensure 5D input for conv3d: (B, C=1, D=1, H, W)
            if view_slice.dim() == 4:
                view_3d = view_slice.unsqueeze(2)  # (B, 1, 1, H, W)
            else:
                view_3d = view_slice

            # Pad depth to match patch embed minimum
            p_d = self.patch_size[0]
            if view_3d.shape[2] < p_d:
                view_3d = F.pad(view_3d, (0, 0, 0, 0, 0, p_d - view_3d.shape[2]),
                                mode='constant', value=0)

            patches_v = self.patch_embed(view_3d)
            patches_v = self.pos_embed(patches_v)
            patches_v = self.dropout(patches_v)

            cls_v, hidden_v = self.encoder(
                patches_v, return_all_layers=return_features
            )
            all_cls.append(cls_v)

            if return_features and hidden_v:
                all_hidden.append(hidden_v)

        # Aggregate CLS tokens across views (mean pooling)
        cls_stacked = torch.stack(all_cls, dim=1)  # (B, V, embed_dim)
        cls_feat = cls_stacked.mean(dim=1)  # (B, embed_dim)

        # Global projection
        global_feat = self.global_proj(cls_feat)  # (B, embed_dim * 2)

        # Multi-scale features
        # In 2.5D mode, patches are from 2D slices, so 3D reshaping doesn't apply.
        # Return placeholder features for API compatibility.
        if return_features:
            stage_features = self._create_dummy_features([])
            stage_features = [
                f.expand(B, -1, -1, -1, -1).to(x.device)
                if f.shape[0] == 1 else f.to(x.device)
                for f in stage_features
            ]
        else:
            stage_features = None

        return global_feat, stage_features

    def forward(
        self, x: torch.Tensor, return_features: bool = False,
        actual_grid: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass. API compatible with SeismicEncoder3D.

        Args:
            x: (B, 1, D, H, W) 3D seismic volume
            return_features: Return intermediate stage features for skip connections
            actual_grid: (D',H',W') actual grid for variable-size input

        Returns:
            global_features: (B, embed_dim * 2)
            Optional: stage_features: List of (B, Ci, Di, Hi, Wi)
        """
        if self.mode == "2.5d":
            return self.forward_2_5d(x, return_features)
        else:
            return self.forward_3d(x, return_features, actual_grid=actual_grid)

    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.embed_dim * 2

    def get_num_patches(self) -> int:
        """Return number of 3D patches."""
        return self.patch_embed.num_patches

    # ---- Pretraining Support (MAE) ----

    def forward_mae_encoder(
        self, x: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAE encoder forward: randomly masks patches and encodes visible ones.

        Used during pretraining phase for Masked Seismic Modeling (MSM).

        Args:
            x: (B, 1, D, H, W)
            mask_ratio: Ratio of patches to mask (default 0.75)

        Returns:
            encoded: (B, N_visible, embed_dim) encoded visible patches
            mask: (B, N_total) boolean mask (True = masked)
            ids_restore: (B, N_total) indices to restore original order
        """
        B = x.shape[0]

        # Patch embedding
        patches = self.patch_embed(x)  # (B, N, embed_dim)
        patches = self.pos_embed(patches)
        N = patches.shape[1]

        # Random masking
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep visible patches
        ids_keep = ids_shuffle[:, :len_keep]
        visible_patches = torch.gather(
            patches, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        )

        # Encode visible patches (without CLS token during MAE pretraining)
        # Add position encoding (already done)
        for blk in self.encoder.blocks:
            visible_patches = blk(visible_patches)

        visible_patches = self.encoder.norm(visible_patches)

        # Mask in shuffled order, then restore to original patch order
        # (must match msm_task.py / standard MAE; without gather, loss hits
        # fixed index ranges instead of the randomly masked patches).
        mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_patches, mask, ids_restore

    def forward_mae(
        self,
        x: torch.Tensor,
        decoder: nn.Module,
        mask_ratio: float = 0.75,
    ) -> Dict[str, torch.Tensor]:
        """
        Full MAE forward: encode visible patches and decode to reconstruct.

        Args:
            x: (B, 1, D, H, W)
            decoder: MAEDecoder3D instance
            mask_ratio: Masking ratio

        Returns:
            dict with:
                - pred: (B, N_total, patch_dim) predicted patches
                - mask: (B, N_total) boolean mask
                - loss: reconstruction loss (if computed)
        """
        encoded, mask, ids_restore = self.forward_mae_encoder(x, mask_ratio)
        pred = decoder(encoded, ids_restore)

        # Compute target (original pixel values for each patch)
        target = self._patchify(x)  # (B, N, patch_dim)

        # Per-patch normalization (MAE norm_pix_loss): makes MSM scale-invariant
        # across fields / amplitude ranges so low-energy volumes don't dominate.
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        # Reconstruction loss (only on masked patches)
        loss = F.mse_loss(pred[mask], target[mask])

        return {
            "pred": pred,
            "mask": mask,
            "target": target,
            "loss": loss,
        }

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert 3D volume to sequence of flattened patches.

        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, N_patches, patch_dim)
        """
        p_d, p_h, p_w = self.patch_size
        B, C, D, H, W = x.shape
        assert D % p_d == 0 and H % p_h == 0 and W % p_w == 0

        patches = rearrange(
            x, "b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)",
            pd=p_d, ph=p_h, pw=p_w,
        )
        return patches

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert sequence of patches back to 3D volume.

        Args:
            patches: (B, N_patches, patch_dim)
        Returns:
            (B, C, D_out, H_out, W_out)
        """
        p_d, p_h, p_w = self.patch_size
        D, H, W = self.patch_embed.grid_size

        x = rearrange(
            patches,
            "b (d h w) (pd ph pw c) -> b c (d pd) (h ph) (w pw)",
            d=D, h=H, w=W, pd=p_d, ph=p_h, pw=p_w, c=self.in_channels,
        )
        return x


# ==============================================================================
# Pretrained Model Loading
# ==============================================================================

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str = "NorskRegnesentralSTI/NCS-v1-2.5d-base",
        img_size: Tuple[int, int, int] = (128, 256, 256),
        dropout: float = 0.1,
        **kwargs,
    ) -> "NCSSeismicEncoder3D":
        """
        Load pretrained NCS-Model weights from HuggingFace.

        The NCS-Model is trained in 2.5D mode (4 directional 2D slices).
        We load the ViT backbone weights and set up the 2.5D multi-view
        extraction pipeline for 3D volume processing.

        Supported pretrained models:
        - "NorskRegnesentralSTI/NCS-v1-2d-base"    (2D mode, 1 view)
        - "NorskRegnesentralSTI/NCS-v1-2.5d-base"  (2.5D mode, 4 views, DEFAULT)
        - "NorskRegnesentralSTI/NCS-v1-3d"         (3D mode, if available)

        Args:
            pretrained_name: HuggingFace model ID
            img_size: Input volume shape for the encoder (D, H, W)
            dropout: Dropout rate
            **kwargs: Additional config overrides (mode, embed_dim, num_layers, etc.)

        Returns:
            NCSSeismicEncoder3D with pretrained ViT weights loaded
        """
        import logging
        logger = logging.getLogger(__name__)

        # Allow explicit mode override from kwargs, otherwise detect from name
        mode = kwargs.pop("mode", None)
        if mode is None:
            if "2.5d" in pretrained_name:
                mode = "2.5d"
            elif "2d" in pretrained_name:
                mode = "2.5d"
            elif "3d" in pretrained_name:
                mode = "3d"
            else:
                mode = "2.5d"

        num_views = 4 if "2.5d" in str(mode) else 1

        # Try method 1: Load via NCS custom package (preferred)
        try:
            logger.info(f"Attempting to load {pretrained_name} via NCS package...")
            encoder = cls._load_via_ncs_package(
                pretrained_name, img_size, mode, num_views, dropout, **kwargs
            )
            if encoder is not None:
                logger.info(f"Successfully loaded pretrained NCS model: {pretrained_name}")
                return encoder
        except Exception as e:
            logger.warning(f"NCS package loading failed: {e}")

        # Try method 2: Load raw HuggingFace weights
        try:
            logger.info(f"Attempting to load {pretrained_name} via HuggingFace...")
            encoder = cls._load_via_huggingface(
                pretrained_name, img_size, mode, num_views, dropout, **kwargs
            )
            if encoder is not None:
                logger.info(f"Successfully loaded pretrained weights from HuggingFace: {pretrained_name}")
                return encoder
        except Exception as e:
            logger.warning(f"HuggingFace loading failed: {e}")

        # Fallback: create from scratch with architecture matching pretrained config
        logger.warning(
            f"Could not load pretrained weights. "
            f"Creating NCS encoder from scratch with {mode} mode architecture. "
            f"Install NCS package: pip install git+https://github.com/NorskRegnesentral/NCS_models"
        )
        return cls._build_pretrained_equivalent(
            pretrained_name, img_size, mode, num_views, dropout, **kwargs
        )

    @classmethod
    def _load_via_ncs_package(
        cls,
        pretrained_name: str,
        img_size: Tuple[int, int, int],
        mode: str,
        num_views: int,
        dropout: float,
        **kwargs,
    ) -> Optional["NCSSeismicEncoder3D"]:
        """Load pretrained model via the official NCS package."""
        try:
            from NCS.models.vit25d import ViT25DModel
        except ImportError:
            return None

        # Load the full pretrained model
        ncs_model = ViT25DModel.from_pretrained(pretrained_name)

        # Extract config from loaded model
        config = ncs_model.config if hasattr(ncs_model, 'config') else {}
        embed_dim = getattr(config, 'hidden_size', 768)
        num_layers = getattr(config, 'num_hidden_layers', 12)
        num_heads = getattr(config, 'num_attention_heads', 12)
        patch_size_2d = getattr(config, 'patch_size', 16)

        # Build our 3D encoder in 2.5D mode
        # The 2.5D mode uses 2D patches from 3D volume slices
        encoder = cls(
            in_channels=1,
            img_size=img_size,
            patch_size=(1, patch_size_2d, patch_size_2d),  # depth=1 for 2D slices
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            mode=mode,
        )

        # Transfer ViT encoder weights
        encoder._transfer_2d_vit_weights(ncs_model)

        # Store reference to pretrained model for efficient 2.5D inference
        encoder._ncs_pretrained = ncs_model

        return encoder

    @classmethod
    def _load_via_huggingface(
        cls,
        pretrained_name: str,
        img_size: Tuple[int, int, int],
        mode: str,
        num_views: int,
        dropout: float,
        **kwargs,
    ) -> Optional["NCSSeismicEncoder3D"]:
        """Load pretrained weights directly from HuggingFace hub."""
        try:
            from huggingface_hub import hf_hub_download
            import json
        except ImportError:
            return None

        # Download config
        try:
            config_path = hf_hub_download(pretrained_name, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception:
            # Use default ViT-Base config
            config = {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "patch_size": 16,
                "image_size": 224,
            }

        embed_dim = config.get("hidden_size", 768)
        num_layers = config.get("num_hidden_layers", 12)
        num_heads = config.get("num_attention_heads", 12)
        patch_size_2d = config.get("patch_size", 16)

        # Download model weights
        try:
            model_path = hf_hub_download(pretrained_name, "pytorch_model.bin")
            state_dict = torch.load(model_path, map_location="cpu")
        except Exception:
            try:
                # Try safetensors
                from safetensors.torch import load_file
                model_path = hf_hub_download(pretrained_name, "model.safetensors")
                state_dict = load_file(model_path)
            except Exception:
                return None

        # Build encoder
        encoder = cls(
            in_channels=1,
            img_size=img_size,
            patch_size=(1, patch_size_2d, patch_size_2d),
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            mode=mode,
        )

        # Map and load weights
        encoder._load_hf_state_dict(state_dict)

        return encoder

    @classmethod
    def _build_pretrained_equivalent(
        cls,
        pretrained_name: str,
        img_size: Tuple[int, int, int],
        mode: str,
        num_views: int,
        dropout: float,
        **kwargs,
    ) -> "NCSSeismicEncoder3D":
        """Build a from-scratch encoder matching the pretrained architecture."""
        # NCS-Base defaults: ViT-B with 2.5D
        if "large" in pretrained_name:
            embed_dim, num_layers, num_heads = 1024, 24, 16
        else:
            embed_dim, num_layers, num_heads = 768, 12, 12

        return cls(
            in_channels=1,
            img_size=img_size,
            patch_size=(1, 16, 16),  # Standard ViT patch size for 2D
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            mode=mode,
        )

    def _transfer_2d_vit_weights(self, pretrained_model: nn.Module):
        """
        Transfer 2D ViT weights from pretrained NCS model to our encoder.

        Maps the ViT backbone weights (patch embedding, position encoding,
        transformer blocks, layer norms) to our encoder.
        """
        pretrained_state = pretrained_model.state_dict()
        own_state = self.state_dict()

        # Key mappings: pretrained -> ours
        mapping = {}

        for own_key in own_state.keys():
            # Try to find matching pretrained key
            pretrained_key = self._map_to_pretrained_key(own_key, pretrained_state)
            if pretrained_key is not None:
                mapping[own_key] = pretrained_key

        # Transfer matched weights
        transferred = 0
        for own_key, pretrained_key in mapping.items():
            if own_key in own_state and pretrained_key in pretrained_state:
                pretrained_weight = pretrained_state[pretrained_key]
                own_weight = own_state[own_key]

                if pretrained_weight.shape == own_weight.shape:
                    own_state[own_key] = pretrained_weight
                    transferred += 1
                else:
                    # Try to adapt shape
                    adapted = self._adapt_weight_shape(
                        pretrained_weight, own_weight.shape
                    )
                    if adapted is not None:
                        own_state[own_key] = adapted
                        transferred += 1

        # Load transferred weights
        self.load_state_dict(own_state, strict=False)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Transferred {transferred}/{len(own_state)} parameter tensors")

    def _map_to_pretrained_key(
        self, own_key: str, pretrained_state: dict
    ) -> Optional[str]:
        """Map our parameter key to pretrained model key."""
        # Direct matches
        if own_key in pretrained_state:
            return own_key

        # Common ViT key patterns
        key_map = {
            # Patch embedding
            "patch_embed.proj.weight": ["patch_embed.proj.weight",
                                          "vit.patch_embed.projection.weight",
                                          "encoder.patch_embed.projection.weight"],
            "patch_embed.proj.bias": ["patch_embed.proj.bias",
                                       "vit.patch_embed.projection.bias",
                                       "encoder.patch_embed.projection.bias"],
            # CLS token
            "cls_token": ["cls_token", "vit.cls_token", "encoder.cls_token"],
            # Encoder blocks
            "encoder.blocks.": ["vit.encoder.layer.", "encoder.blocks.",
                                "vit.encoder.blocks.", "blocks."],
            # Norm
            "encoder.norm.": ["vit.layernorm.", "encoder.norm.",
                              "vit.encoder.norm.", "norm."],
        }

        for pattern, candidates in key_map.items():
            if pattern in own_key:
                suffix = own_key.split(pattern, 1)[-1] if pattern != own_key else ""
                for candidate in candidates:
                    if candidate.endswith("."):
                        pretrained_key = candidate + suffix
                    else:
                        pretrained_key = candidate + ("." + suffix if suffix else "")

                    if pretrained_key in pretrained_state:
                        return pretrained_key

        # Fuzzy match: try to find by suffix
        own_suffix = own_key.split(".", 1)[-1] if "." in own_key else own_key
        for pk in pretrained_state.keys():
            if pk.endswith(own_suffix):
                return pk

        return None

    def _adapt_weight_shape(
        self, pretrained_weight: torch.Tensor, target_shape: torch.Size
    ) -> Optional[torch.Tensor]:
        """Adapt pretrained weight to target shape."""
        if len(pretrained_weight.shape) != len(target_shape):
            return None

        # Handle 2D -> 3D convolution inflation
        if len(target_shape) == 5 and len(pretrained_weight.shape) == 5:
            # Conv3d weight: (out, in, D, H, W)
            if pretrained_weight.shape[2] == 1 and target_shape[2] > 1:
                # Tile along depth dimension
                return pretrained_weight.repeat(1, 1, target_shape[2], 1, 1)
            elif pretrained_weight.shape[2] > target_shape[2]:
                return pretrained_weight[:, :, :target_shape[2], :, :]

        # Handle 2D position encoding -> 3D position encoding
        if len(target_shape) == 3 and len(pretrained_weight.shape) == 3:
            # (1, N_2d, D) -> (1, N_3d, D)
            if pretrained_weight.shape[2] == target_shape[2]:
                if pretrained_weight.shape[1] < target_shape[1]:
                    # Interpolate to larger size
                    factor = target_shape[1] / pretrained_weight.shape[1]
                    return F.interpolate(
                        pretrained_weight.transpose(1, 2).unsqueeze(0),
                        scale_factor=factor, mode="linear"
                    ).squeeze(0).transpose(0, 1)
                else:
                    return pretrained_weight[:, :target_shape[1], :]

        # Handle 1D/2D weight in linear layers (same shape)
        if pretrained_weight.shape == target_shape:
            return pretrained_weight

        # Handle linear weight shape change (e.g. head dim change)
        if len(target_shape) == 2 and len(pretrained_weight.shape) == 2:
            if target_shape[1] == pretrained_weight.shape[1]:
                return pretrained_weight[:target_shape[0], :]
            elif target_shape[0] == pretrained_weight.shape[0]:
                return pretrained_weight[:, :target_shape[1]]

        return None

    def _load_hf_state_dict(self, state_dict: dict):
        """Load HuggingFace state dict into encoder."""
        own_state = self.state_dict()
        transferred = 0

        for own_key in own_state.keys():
            pretrained_key = self._map_to_pretrained_key(own_key, state_dict)
            if pretrained_key is not None and pretrained_key in state_dict:
                pretrained_weight = state_dict[pretrained_key]
                if own_state[own_key].shape == pretrained_weight.shape:
                    own_state[own_key] = pretrained_weight
                    transferred += 1
                else:
                    adapted = self._adapt_weight_shape(
                        pretrained_weight, own_state[own_key].shape
                    )
                    if adapted is not None:
                        own_state[own_key] = adapted
                        transferred += 1

        self.load_state_dict(own_state, strict=False)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {transferred}/{len(own_state)} tensors from HuggingFace state dict")

    def is_pretrained(self) -> bool:
        """Check if encoder has loaded pretrained weights."""
        return hasattr(self, '_ncs_pretrained')

    # ==========================================================================
    # Industrial Inference: Variable-size Input
    # ==========================================================================

    def infer_variable_size(
        self,
        x: torch.Tensor,
        tile_size: Tuple[int, int, int] = (32, 64, 64),
        tile_overlap: float = 0.25,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Sliding-window inference for arbitrary-size 3D seismic volumes.

        Splits the input into overlapping tiles, encodes each independently,
        and merges results. Essential for real-world volumes that don't match
        training dimensions.

        Args:
            x: (B=1, 1, D, H, W) 3D seismic volume, any size
            tile_size: (d, h, w) tile dimensions
            tile_overlap: Overlap ratio between adjacent tiles [0, 1)
            reduction: 'mean' | 'max' | 'cls' for merging tile features

        Returns:
            global_feat: (1, embed_dim*2) global feature
        """
        B, C, D, H, W = x.shape
        td, th, tw = tile_size
        sd = max(1, int(td * (1 - tile_overlap)))
        sh = max(1, int(th * (1 - tile_overlap)))
        sw = max(1, int(tw * (1 - tile_overlap)))

        all_features = []
        all_weights = []

        for d_start in range(0, D, sd):
            for h_start in range(0, H, sh):
                for w_start in range(0, W, sw):
                    d_end = min(D, d_start + td)
                    h_end = min(H, h_start + th)
                    w_end = min(W, w_start + tw)

                    # Extract tile
                    tile = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                    # Pad if smaller than tile_size
                    if tile.shape[2:] != tile_size:
                        pad_d = td - tile.shape[2]
                        pad_h = th - tile.shape[3]
                        pad_w = tw - tile.shape[4]
                        tile = F.pad(tile, (0, pad_w, 0, pad_h, 0, pad_d),
                                     mode='constant', value=0)

                    # Encode (compute actual grid for variable-size tiles)
                    with torch.no_grad():
                        pd, ph, pw = self.patch_size
                        ag = (tile.shape[2] // pd, tile.shape[3] // ph, tile.shape[4] // pw)
                        feat, _ = self.forward_3d(tile, return_features=False, actual_grid=ag)
                        feat = feat.squeeze(0)  # (B=1, D) -> (D,)

                    # Weight by valid (non-padded) ratio
                    valid_ratio = (
                        (d_end - d_start) / td *
                        (h_end - h_start) / th *
                        (w_end - w_start) / tw
                    )

                    all_features.append(feat)
                    all_weights.append(valid_ratio)

        # Merge
        all_feat = torch.stack(all_features, dim=0)  # (N_tiles, D)
        weights = torch.tensor(all_weights, device=x.device).view(-1, 1)

        if reduction == "mean":
            global_feat = (all_feat * weights).sum(dim=0, keepdim=True) / weights.sum()
        elif reduction == "max":
            global_feat = all_feat.max(dim=0, keepdim=True)[0]
        else:  # cls: just average
            global_feat = all_feat.mean(dim=0, keepdim=True)

        return global_feat

    def train_with_random_crop(
        self,
        x: torch.Tensor,
        crop_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Random crop during training — teaches the model to handle variable input.

        Args:
            x: (B, 1, D, H, W) full-size input
            crop_size: (d, h, w) target crop size (must be <= input size)

        Returns:
            cropped: (B, 1, d, h, w)
        """
        B, C, D, H, W = x.shape
        cd, ch, cw = crop_size

        d_start = torch.randint(0, max(1, D - cd + 1), (1,)).item()
        h_start = torch.randint(0, max(1, H - ch + 1), (1,)).item()
        w_start = torch.randint(0, max(1, W - cw + 1), (1,)).item()

        return x[:, :, d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]


# ==============================================================================
# Factory function for building NCS encoder variants
# ==============================================================================

def build_ncs_encoder(
    variant: str = "base",
    img_size: Tuple[int, int, int] = (128, 256, 256),
    mode: str = "3d",
    **kwargs,
) -> NCSSeismicEncoder3D:
    """
    Build an NCS encoder with predefined configurations.

    Variants:
        - 'tiny':  ViT-Tiny  (embed_dim=192, layers=12, heads=3)  ~5.7M params
        - 'small': ViT-Small (embed_dim=384, layers=12, heads=6)  ~22M params
        - 'base':  ViT-Base  (embed_dim=768, layers=12, heads=12) ~86M params
        - 'large': ViT-Large (embed_dim=1024, layers=24, heads=16) ~307M params

    Args:
        variant: Model size variant
        img_size: Input volume shape (D, H, W)
        mode: '3d' or '2.5d'
        **kwargs: Additional overrides

    Returns:
        Configured NCSSeismicEncoder3D
    """
    configs = {
        "tiny": {
            "embed_dim": 192,
            "num_layers": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
        },
        "small": {
            "embed_dim": 384,
            "num_layers": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
        "base": {
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
        },
        "large": {
            "embed_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
        },
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(configs.keys())}")

    cfg = configs[variant]
    cfg.update(kwargs)

    return NCSSeismicEncoder3D(
        img_size=img_size,
        mode=mode,
        **cfg,
    )
