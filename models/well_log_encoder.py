"""
1D Well Log Sequence Encoder

Encodes multi-curve well log sequences using:
- Multi-scale 1D CNN for local feature extraction
- Rotary Position Embedding (RoPE) Transformer for long-range depth dependencies
- Optional physics-aware encoding for petrophysical constraints

Architecture:
    Input: (B, num_curves, L) well log curves
    -> Multi-scale 1D CNN (parallel conv branches)
    -> Physics-aware normalization (optional)
    -> RoPE Transformer Encoder
    -> Attention Pooling
    -> Output: (B, hidden_dim) global well features + (B, L', hidden_dim) sequence features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional
from einops import rearrange


# =====================================================================
# Rotary Position Embedding (RoPE)
# =====================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 1D sequences.

    Provides relative position awareness without learnable parameters.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin values for rotary embedding.

        Args:
            x: input tensor (for device/dtype)
            seq_len: sequence length

        Returns:
            cos, sin: (1, seq_len, 1, dim) tensors
        """
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos = emb.cos()[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = emb.sin()[None, :, None, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =====================================================================
# Multi-Scale 1D CNN Stem
# =====================================================================

class MultiScaleConv1D(nn.Module):
    """
    Multi-scale 1D convolutional stem.
    Uses parallel conv branches with different kernel sizes
    to capture features at multiple scales.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 128,
        kernel_sizes: List[int] = [3, 5, 7],
        stride: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Parallel conv branches
        branch_out = out_channels // len(kernel_sizes)
        remainder = out_channels - branch_out * len(kernel_sizes)

        self.branches = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            branch_ch = branch_out + (remainder if i == 0 else 0)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_ch, kernel_size=ks,
                              stride=stride, padding=ks // 2),
                    nn.BatchNorm1d(branch_ch),
                    nn.GELU(),
                )
            )

        # Fusion after multi-scale
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) well log curves
        Returns:
            (B, out_channels, L/2)
        """
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)  # (B, out_channels, L/2)
        x = self.fusion(x)
        return x


# =====================================================================
# Physics-Aware Encoding
# =====================================================================

class PhysicsAwareEncoding(nn.Module):
    """
    Petrophysical constraint encoding.

    Encodes known physical relationships between well log curves:
    - GR vs clay content (high GR = more clay/shale)
    - Density vs Porosity relationship
    - Acoustic velocity vs Porosity (Wyllie time-average)

    Acts as an inductive bias to improve feature learning.
    """

    def __init__(
        self,
        num_curves: int = 7,
        embed_dim: int = 128,
        curve_names: List[str] = None,
    ):
        super().__init__()
        if curve_names is None:
            curve_names = ["GR", "RT", "DEN", "POR", "AC", "SP", "CAL"]

        self.curve_names = curve_names
        self.num_curves = num_curves

        # Learnable physical relationship embeddings
        self.phys_embed = nn.Parameter(torch.randn(num_curves, num_curves, embed_dim // num_curves))
        nn.init.normal_(self.phys_embed, std=0.02)

        # Curve type embeddings
        self.curve_type_embed = nn.Parameter(torch.randn(num_curves, embed_dim))
        nn.init.normal_(self.curve_type_embed, std=0.02)

        # MLP for cross-curve interaction
        self.interaction = nn.Sequential(
            nn.Linear(num_curves * embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) well log features
        Returns:
            (B, L, embed_dim) physics-aware features
        """
        B, L, C = x.shape
        assert C == self.num_curves, f"Expected {self.num_curves} curves, got {C}"

        # Add curve type embedding
        curve_embed = self.curve_type_embed[:C].unsqueeze(0).unsqueeze(0)  # (1, 1, C, E)
        x_expanded = x.unsqueeze(-1) * curve_embed  # (B, L, C, E)

        # Cross-curve interaction
        x_flat = x_expanded.reshape(B, L, -1)  # (B, L, C*E)
        x = self.interaction(x_flat)

        return x


# =====================================================================
# RoPE Multi-Head Attention
# =====================================================================

class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
            mask: Optional attention mask
        """
        B, L, C = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rope(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Reshape for attention
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = attn @ v  # (B, H, L, D)
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.out_proj(x)
        return x


# =====================================================================
# Transformer Encoder Layer
# =====================================================================

class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


# =====================================================================
# Attention Pooling
# =====================================================================

class AttentionPooling1D(nn.Module):
    """
    Learnable attention pooling to aggregate sequence features
    into a fixed-length global representation.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) sequence features
            mask: (B, L) optional mask
        Returns:
            (B, C) pooled global feature
        """
        attn_weights = self.attention(x)  # (B, L, 1)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(
                mask.unsqueeze(-1) == 0, float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=1)
        x = (attn_weights * x).sum(dim=1)  # (B, C)
        return x


# =====================================================================
# Well Log Encoder
# =====================================================================

class WellLogEncoder1D(nn.Module):
    """
    1D Well Log Sequence Encoder.

    Combines multi-scale CNN, RoPE Transformer, and optional
    physics-aware encoding for robust well log feature extraction.

    Args:
        num_curves: Number of input log curves
        stem_channels: Multi-scale CNN output channels per stage
        kernel_sizes: Kernel sizes for multi-scale CNN
        embed_dim: Transformer embedding dimension
        num_layers: Number of Transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        use_physics_constraint: Enable physics-aware encoding
    """

    def __init__(
        self,
        num_curves: int = 7,
        stem_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 5, 7],
        embed_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_physics_constraint: bool = True,
    ):
        super().__init__()
        self.num_curves = num_curves
        self.embed_dim = embed_dim
        self.use_physics_constraint = use_physics_constraint

        # Multi-scale CNN stem
        cnn_channels = stem_channels[-1] if stem_channels else 128
        self.cnn_stem = MultiScaleConv1D(
            in_channels=num_curves,
            out_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
        )

        # Optional additional CNN stages
        self.cnn_stages = nn.ModuleList()
        current_channels = cnn_channels
        for ch in [cnn_channels * 2] if len(stem_channels) > 1 else []:
            self.cnn_stages.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, ch, 3, stride=2, padding=1),
                    nn.BatchNorm1d(ch),
                    nn.GELU(),
                )
            )
            current_channels = ch

        # Project CNN features to transformer dim
        self.input_proj = nn.Linear(current_channels, embed_dim)

        # Physics-aware encoding (optional)
        if use_physics_constraint:
            self.physics_encoding = PhysicsAwareEncoding(
                num_curves=num_curves, embed_dim=embed_dim
            )
            # Fuse: CNN features + physics features
            self.modality_fuse = nn.Linear(embed_dim * 2, embed_dim)

        # RoPE Transformer Encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Attention pooling
        self.attn_pool = AttentionPooling1D(embed_dim)

        # Global projection
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, L) well log curves (B, num_curves, seq_len)
            mask: Optional (B, L) attention mask
            return_sequence: If True, also return full sequence features

        Returns:
            global_features: (B, embed_dim)
            If return_sequence: also returns (B, L_out, embed_dim)
        """
        B, C, L = x.shape

        # Multi-scale CNN
        cnn_feat = self.cnn_stem(x)  # (B, cnn_ch, L')

        # Additional CNN stages
        for cnn_stage in self.cnn_stages:
            cnn_feat = cnn_stage(cnn_feat)

        # Transpose for transformer: (B, L', cnn_ch) -> (B, L', embed_dim)
        cnn_feat = cnn_feat.transpose(1, 2)  # (B, L', C_cnn)
        cnn_feat = self.input_proj(cnn_feat)  # (B, L', embed_dim)
        L_out = cnn_feat.shape[1]

        # Physics-aware encoding
        if self.use_physics_constraint:
            # Raw log input for physics features
            raw_x = x.transpose(1, 2)  # (B, L, C)
            # Interpolate to match CNN output length
            if L != L_out:
                raw_x = F.interpolate(
                    raw_x.transpose(1, 2), size=L_out, mode="linear"
                ).transpose(1, 2)

            phys_feat = self.physics_encoding(raw_x)  # (B, L', embed_dim)
            # Fuse CNN and physics features
            seq_feat = self.modality_fuse(
                torch.cat([cnn_feat, phys_feat], dim=-1)
            )
        else:
            seq_feat = cnn_feat

        # Mask interpolation if provided
        attn_mask = None
        if mask is not None and mask.shape[1] != L_out:
            mask = F.interpolate(
                mask.float().unsqueeze(1), size=L_out, mode="nearest"
            ).squeeze(1).bool()

        # Transformer layers
        for layer in self.transformer_layers:
            seq_feat = layer(seq_feat, attn_mask)

        # Global pooling
        global_feat = self.attn_pool(seq_feat, mask)
        global_feat = self.global_proj(global_feat)
        global_feat = self.output_norm(global_feat)

        if return_sequence:
            return global_feat, seq_feat
        return global_feat, None

    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.embed_dim


# =====================================================================
# Depth-Aware Positional Encoding
# =====================================================================

class DepthPositionalEncoding(nn.Module):
    """
    Depth-aware positional encoding for well log sequences.

    Unlike standard position encoding, this encodes actual
    depth information which carries geological meaning
    (compaction trends, temperature gradients, etc.).
    """

    def __init__(self, embed_dim: int, max_depth: float = 5000.0):
        super().__init__()
        self.max_depth = max_depth
        self.embed_dim = embed_dim

        # Depth value encoder
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Geological trend encoder
        self.trend_encoder = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )
        nn.init.normal_(self.trend_encoder, std=0.02)

    def forward(self, x: torch.Tensor, depths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) features
            depths: (B, L) depth values in meters, or None

        Returns:
            depth-encoded features
        """
        B, L, C = x.shape

        if depths is not None:
            # Use actual depth values
            depth_norm = depths / self.max_depth
            depth_embed = self.depth_mlp(depth_norm.unsqueeze(-1))  # (B, L, C)
        else:
            # Use linear depth proxy
            depth_proxy = torch.linspace(0, 1, L, device=x.device)
            depth_proxy = depth_proxy.unsqueeze(0).unsqueeze(-1).expand(B, L, 1)
            depth_embed = self.depth_mlp(depth_proxy)

        # Add geological trend encoding
        trend = self.trend_encoder.expand(B, L, -1)

        return x + depth_embed + trend
