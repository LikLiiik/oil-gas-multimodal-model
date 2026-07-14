"""
WLFM Well-Log Foundation Model Encoder

Implements the WLFM (Well-Logs Foundation Model) architecture for
1D well log sequence encoding. Based on the paper:
  "WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well
   Geological Interpretation" (Qi et al., 2025)

Key architectural features:
1. VQ-VAE Tokenizer: Discretizes continuous log patches into a learned
   codebook of "geological vocabulary" tokens
2. Depthwise Separable CNN Encoder for efficient patch processing
3. Masked Token Modeling (MTM): Self-supervised pretraining by masking
   and predicting discrete tokens
4. Stratigraphy-aware Contrastive Learning (SCL): Aligns representations
   across wells using stratigraphic proximity
5. Curve-type embeddings for multi-curve awareness
6. Relative-depth positional encodings

Architecture Overview:
    Input: (B, C, L) multi-curve well log sequences
    -> Per-well Z-score normalization
    -> Patch Segmentation (L -> P patches of length patch_len)
    -> Curve-type Embeddings + Depth Positional Encoding
    -> VQ-VAE Encoder (Depthwise Separable CNN + Vector Quantizer)
    -> Discrete Codebook Tokens (geological vocabulary)
    -> Transformer Encoder with Masked Token Modeling
    -> Attention Pooling
    -> Output: (B, embed_dim) global features + (B, P', embed_dim) sequence

Reference:
    Paper: https://arxiv.org/abs/2509.18152
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional, Dict
from einops import rearrange


# ==============================================================================
# Vector Quantizer (VQ-VAE Codebook)
# ==============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with exponential moving average (EMA) codebook updates.

    Maps continuous input vectors to the nearest entry in a learned codebook.
    Uses straight-through estimator for gradient propagation during training,
    with EMA-based codebook maintenance for stable training.

    This creates the "geological vocabulary" — discrete tokens that represent
    recurring well log patterns (shale-sand transitions, high-resistivity zones, etc.)

    Args:
        num_embeddings: Codebook size (number of discrete tokens)
        embedding_dim: Dimension of each codebook entry
        commitment_cost: Weight for commitment loss (default 0.25)
        decay: EMA decay rate for codebook updates (default 0.99)
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        dead_code_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # EMA tracking buffers (not parameters)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        # Accumulate encoder vectors only. Initializing this from the random
        # codebook makes never-used entries explode when divided by an almost
        # zero EMA cluster size.
        self.register_buffer("ema_w", torch.zeros_like(self.embedding.weight.data))

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous vectors to nearest codebook entries.

        Args:
            z: (B, N, D) continuous encoder outputs

        Returns:
            z_q: (B, N, D) quantized vectors (straight-through)
            indices: (B, N) codebook indices
            loss: Quantization loss (commitment + codebook)
        """
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*N, D)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Find nearest codebook entry
        indices = torch.argmin(distances, dim=1)  # (B*N,)
        z_q = self.embedding(indices).view(B, N, D)  # (B, N, D)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Commitment loss: encoder outputs must move toward codebook entries.
        # EMA already updates the codebook, so only β ||z - sg[e]||^2 is needed.
        commitment_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())

        # Update EMA buffers (only during training)
        if self.training:
            with torch.no_grad():
                # Update cluster sizes
                encodings = F.one_hot(indices, self.num_embeddings).float()
                self.ema_cluster_size.data.mul_(self.decay).add_(
                    encodings.sum(dim=0), alpha=1 - self.decay
                )

                # Update embeddings via EMA
                dw = torch.matmul(encodings.t(), z_flat)  # (K, D)
                self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Only update entries that have actually received assignments.
                # Keeping unused entries at their finite random initialization
                # lets them become active later and avoids inf/huge vectors.
                active = self.ema_cluster_size > 1e-5
                if active.any():
                    embed_normalized = (
                        self.ema_w[active]
                        / self.ema_cluster_size[active].unsqueeze(1).clamp_min(1e-5)
                    )
                    self.embedding.weight.data[active].copy_(embed_normalized)

                # Revive dead codes by reinitializing them from random encoder
                # outputs. Without this the codebook collapses to a handful of
                # entries and MTM validation CE stops improving.
                dead = self.ema_cluster_size < self.dead_code_threshold
                n_dead = int(dead.sum().item())
                if n_dead > 0 and z_flat.shape[0] > 0:
                    rand_idx = torch.randint(
                        0, z_flat.shape[0], (n_dead,), device=z_flat.device
                    )
                    # Use .data.dtype: under AMP, Embedding weight views may
                    # look like fp16 while the parameter storage stays fp32.
                    embed_dtype = self.embedding.weight.data.dtype
                    resurrected = (
                        z_flat[rand_idx].float().to(embed_dtype)
                        + 0.01
                        * torch.randn(
                            n_dead,
                            D,
                            device=z_flat.device,
                            dtype=embed_dtype,
                        )
                    )
                    self.embedding.weight.data[dead].copy_(resurrected)
                    self.ema_w.data[dead].copy_(resurrected)
                    self.ema_cluster_size.data[dead] = 1.0

        # VQ loss
        vq_loss = commitment_loss

        return z_q_st, indices.view(B, N), vq_loss


# ==============================================================================
# Depthwise Separable CNN Encoder (for VQ-VAE)
# ==============================================================================

class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise separable 1D convolution block.

    Efficient convolution that factorizes standard conv into:
    1. Depthwise: spatial-only convolution (one filter per channel)
    2. Pointwise: channel-only mixing (1x1 convolution)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


# ==============================================================================
# VQ-VAE Encoder (Well Log Tokenizer)
# ==============================================================================

class WellLogTokenizer(nn.Module):
    """
    VQ-VAE based well log tokenizer.

    Converts continuous multi-curve well log patches into discrete tokens
    from a learned codebook (geological vocabulary).

    Architecture:
        Input: (B, C, patch_len) well log patches
        -> Depthwise Separable CNN stack
        -> Residual blocks
        -> Vector Quantizer
        -> Output: (B, embedding_dim) discrete tokens

    Args:
        num_curves: Number of well log curves
        patch_len: Length of each well log patch
        embed_dim: Token embedding dimension
        hidden_dim: Hidden dimension in CNN
        num_embeddings: Codebook size
        num_res_blocks: Number of residual CNN blocks
    """

    def __init__(
        self,
        num_curves: int = 7,
        patch_len: int = 64,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_embeddings: int = 512,
        num_res_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_curves = num_curves
        self.patch_len = patch_len
        self.embed_dim = embed_dim

        # Initial projection
        self.input_proj = nn.Conv1d(num_curves, hidden_dim, kernel_size=1)

        # Depthwise separable CNN blocks (downsampling)
        self.encoder_blocks = nn.ModuleList()
        current_dim = hidden_dim
        current_len = patch_len

        # Block 1: hidden_dim -> hidden_dim * 2, stride 2
        self.encoder_blocks.append(
            DepthwiseSeparableConv1D(
                current_dim, hidden_dim * 2,
                kernel_size=4, stride=2, padding=1, dropout=dropout,
            )
        )
        current_dim = hidden_dim * 2
        current_len = current_len // 2

        # Block 2: hidden_dim*2 -> hidden_dim*4, stride 2
        self.encoder_blocks.append(
            DepthwiseSeparableConv1D(
                current_dim, hidden_dim * 4,
                kernel_size=4, stride=2, padding=1, dropout=dropout,
            )
        )
        current_dim = hidden_dim * 4
        current_len = current_len // 2

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                ResidualConvBlock1D(current_dim, dropout)
            )

        # Final projection to embedding dimension
        self.output_proj = nn.Sequential(
            nn.Conv1d(current_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Global average pooling over the temporal dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Vector quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
        )

    def forward(
        self, x: torch.Tensor, return_indices: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize well log patches.

        Args:
            x: (B, C, patch_len) well log curve patches
            return_indices: Return discrete codebook indices

        Returns:
            dict with:
                - tokens: (B, 1, embed_dim) quantized token vectors
                - indices: (B, 1) codebook indices (if return_indices)
                - vq_loss: scalar quantization loss
                - z_continuous: (B, 1, embed_dim) pre-quantization features
        """
        # CNN encoding
        h = self.input_proj(x)

        for blk in self.encoder_blocks:
            h = blk(h)

        for blk in self.res_blocks:
            h = blk(h)

        h = self.output_proj(h)
        h = self.pool(h)  # (B, embed_dim, 1)
        h = h.squeeze(-1).unsqueeze(1)  # (B, 1, embed_dim)

        # Vector quantization
        z_q, indices, vq_loss = self.quantizer(h)

        result = {
            "tokens": z_q,  # (B, 1, embed_dim) quantized
            "z_continuous": h,  # (B, 1, embed_dim) continuous
            "vq_loss": vq_loss,
        }

        if return_indices:
            result["indices"] = indices

        return result

    def get_codebook(self) -> torch.Tensor:
        """Return the full codebook."""
        return self.quantizer.embedding.weight  # (K, embed_dim)


# ==============================================================================
# Residual Conv Block 1D
# ==============================================================================

class ResidualConvBlock1D(nn.Module):
    """Residual 1D convolution block with two conv layers."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


# ==============================================================================
# Curve-Type Embeddings
# ==============================================================================

class CurveTypeEmbedding(nn.Module):
    """
    Learnable embedding for each well log curve type.

    Provides the model with awareness of which physical measurement
    each curve represents (GR, resistivity, density, etc.),
    similar to how token type embeddings work in BERT.
    """

    def __init__(
        self,
        num_curves: int = 7,
        embed_dim: int = 192,
        curve_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_curves = num_curves

        if curve_names is None:
            curve_names = ["GR", "RT", "DEN", "POR", "AC", "SP", "CAL"]

        self.curve_names = curve_names

        # Learnable curve type embeddings
        self.curve_embed = nn.Parameter(torch.zeros(num_curves, embed_dim))
        nn.init.trunc_normal_(self.curve_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) well log curves
        Returns:
            (B, L, C*embed_dim) curve-type aware features
        """
        B, C, L = x.shape
        # Add curve embedding to each channel
        curve_embed = self.curve_embed[:C]  # (C, embed_dim)
        # Broadcast: x is (B, C, L) -> (B, C, L, 1), curve_embed is (C, embed_dim)
        x = x.unsqueeze(-1)  # (B, C, L, 1)
        x = x * curve_embed.unsqueeze(0).unsqueeze(2)  # (B, C, L, embed_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, L, C * self.curve_embed.shape[-1])
        return x


# ==============================================================================
# Relative Depth Positional Encoding
# ==============================================================================

class RelativeDepthEncoding(nn.Module):
    """
    Relative depth positional encoding for well log sequences.

    Encodes the relative depth position of each patch within the wellbore.
    Uses learnable embeddings with sinusoidal initialization, providing
    the model with awareness of stratigraphic ordering.

    Different from absolute depth encoding — this is a learnable
    position embedding that the model can adapt during pretraining.
    """

    def __init__(self, max_patches: int = 128, embed_dim: int = 192):
        super().__init__()
        self.max_patches = max_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        # Initialize with sinusoidal pattern
        self._init_sinusoidal()

    def _init_sinusoidal(self):
        """Initialize position embeddings with sinusoidal pattern."""
        position = torch.arange(self.max_patches, dtype=torch.float32).unsqueeze(1)
        dim = self.pos_embed.shape[-1]
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(self.max_patches, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, D)
        Returns:
            x + position encoding
        """
        P = x.shape[1]
        if P <= self.max_patches:
            return x + self.pos_embed[:, :P, :]
        # Dynamic extension: interpolate position embeddings for longer sequences
        pos = self.pos_embed  # (1, max_patches, D)
        pos = F.interpolate(
            pos.transpose(1, 2), size=P, mode='linear', align_corners=False
        ).transpose(1, 2)
        return x + pos


# ==============================================================================
# WLFM Transformer Encoder
# ==============================================================================

class WLFMTransformerBlock(nn.Module):
    """Transformer block with pre-norm, following WLFM architecture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# Attention Pooling
# ==============================================================================

class AttentionPooling1D(nn.Module):
    """
    Learnable attention-based pooling for aggregating sequence features
    into a single global representation.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, P, D)
            mask: (B, P) optional mask
        Returns:
            (B, D) pooled global feature
        """
        attn = self.attention(x)  # (B, P, 1)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = F.softmax(attn, dim=1)
        return (attn * x).sum(dim=1)


# ==============================================================================
# WLFM Well Log Encoder (Main Class)
# ==============================================================================

class WLFMWellLogEncoder1D(nn.Module):
    """
    WLFM-based Well Log Encoder.

    Implements the WLFM architecture for multi-curve well log encoding
    with VQ-VAE discretization, masked token modeling pretraining,
    and transformer-based feature extraction.

    API compatible with WellLogEncoder1D.

    Args:
        num_curves: Number of well log curves (default 7)
        patch_len: Length of each patch for tokenization
        patch_stride: Stride between consecutive patches
        embed_dim: Transformer embedding dimension
        vq_embed_dim: VQ-VAE tokenizer embedding dimension
        num_embeddings: VQ codebook size
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        max_seq_len: Maximum sequence length (number of patches)
        use_physics_constraint: Enable physics-aware features (API compat)
        curve_names: Names of well log curves
    """

    def __init__(
        self,
        num_curves: int = 7,
        patch_len: int = 64,
        patch_stride: int = 32,
        embed_dim: int = 192,
        vq_embed_dim: int = 256,
        num_embeddings: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_physics_constraint: bool = True,
        curve_names: Optional[List[str]] = None,
        # API compatibility params (ignored)
        stem_channels: List[int] = None,
        kernel_sizes: List[int] = None,
    ):
        super().__init__()
        self.num_curves = num_curves
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.vq_embed_dim = vq_embed_dim
        self.num_embeddings = num_embeddings
        self.use_physics_constraint = use_physics_constraint

        if curve_names is None:
            curve_names = ["GR", "RT", "DEN", "POR", "AC", "SP", "CAL"]

        self.curve_names = curve_names

        # ===== VQ-VAE Tokenizer =====
        self.tokenizer = WellLogTokenizer(
            num_curves=num_curves,
            patch_len=patch_len,
            embed_dim=vq_embed_dim,
            hidden_dim=128,
            num_embeddings=num_embeddings,
            num_res_blocks=3,
            dropout=dropout,
        )

        # ===== Embeddings =====
        # Curve type embedding (pre-tokenization)
        self.curve_embed = CurveTypeEmbedding(
            num_curves=num_curves,
            embed_dim=patch_len,  # embed along the measurement dimension
        )

        # Project to tokenizer input
        self.input_proj = nn.Linear(num_curves * patch_len, num_curves * patch_len)

        # Token projection: VQ embedding -> Transformer embedding
        self.token_proj = nn.Linear(vq_embed_dim, embed_dim)

        # Relative depth positional encoding
        max_patches = max_seq_len // patch_stride + 1
        self.depth_encoding = RelativeDepthEncoding(
            max_patches=max_patches,
            embed_dim=embed_dim,
        )

        # ===== Physical Constraint Encoding =====
        if use_physics_constraint:
            self.physics_encoder = nn.Sequential(
                nn.Linear(num_curves, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.phys_fuse = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
            )
        else:
            self.physics_encoder = None
            self.phys_fuse = None

        # ===== Transformer Encoder =====
        self.transformer_blocks = nn.ModuleList([
            WLFMTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ===== Mask Token (for MTM pretraining) =====
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ===== Pooling and Projection =====
        self.attn_pool = AttentionPooling1D(embed_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(embed_dim)

        # ===== MTM Prediction Head =====
        self.mtm_head = nn.Linear(embed_dim, num_embeddings)

        # ===== Stratigraphy-aware Contrastive Projection =====
        self.scl_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _segment_patches(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Segment well log sequences into overlapping patches.

        Args:
            x: (B, C, L) well log curves
        Returns:
            patches: (B, P, C, patch_len)
            num_patches: int
        """
        B, C, L = x.shape
        patches = x.unfold(2, self.patch_len, self.patch_stride)  # (B, C, P, patch_len)
        patches = patches.permute(0, 2, 1, 3)  # (B, P, C, patch_len)
        return patches, patches.shape[1]

    def _apply_curve_mask(
        self,
        x: torch.Tensor,
        curve_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Zero out missing curve channels using (B, C) mask."""
        if curve_mask is None:
            return x
        return x * curve_mask.unsqueeze(-1)

    def _per_well_normalize(
        self,
        x: torch.Tensor,
        curve_mask: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-well z-score normalization (following WLFM).

        Args:
            x: (B, C, L) well log curves
            curve_mask: Optional (B, C) binary mask, 1=valid curve
            depth_mask: Optional (B, L) binary mask, 1=valid depth sample
            value_mask: Optional (B, C, L) per-value validity mask
        Returns:
            normalized curves
        """
        if curve_mask is not None:
            mask = curve_mask.unsqueeze(-1).expand_as(x)
        else:
            mask = (x != 0).float()
        if depth_mask is not None:
            mask = mask * depth_mask.unsqueeze(1).expand_as(x)
        if value_mask is not None:
            mask = mask * value_mask.float()

        mean = (x * mask).sum(dim=2, keepdim=True) / (mask.sum(dim=2, keepdim=True) + 1e-8)
        var = ((x - mean) * mask).pow(2).sum(dim=2, keepdim=True) / (mask.sum(dim=2, keepdim=True) + 1e-8)
        std = torch.sqrt(var + 1e-8)

        x_norm = (x - mean) / std
        x_norm = x_norm * mask
        return x_norm

    def tokenize(
        self,
        x: torch.Tensor,
        curve_mask: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize well log sequences into discrete geological tokens.

        Args:
            x: (B, C, L) well log curves

        Returns:
            dict with:
                - tokens: (B, P, vq_embed_dim) discrete token vectors
                - indices: (B, P) codebook indices
                - vq_loss: scalar quantization loss
                - num_patches: int
        """
        B, C, L = x.shape

        x = self._apply_curve_mask(x, curve_mask)

        # Per-well normalization
        x_norm = self._per_well_normalize(
            x,
            curve_mask=curve_mask,
            depth_mask=depth_mask,
            value_mask=value_mask,
        )

        # Segment into patches
        patches, P = self._segment_patches(x_norm)  # (B, P, C, patch_len)

        # Tokenize all patches in one VQ batch so the EMA codebook sees B*P
        # vectors per step instead of only B (strongly reduces collapse).
        Bp = B * P
        flat_patches = patches.reshape(Bp, C, self.patch_len)
        result = self.tokenizer(flat_patches, return_indices=True)
        tokens = result["tokens"].squeeze(1).view(B, P, -1)
        indices = result["indices"].squeeze(1).view(B, P)
        vq_loss = result["vq_loss"]

        return {
            "tokens": tokens,
            "indices": indices,
            "vq_loss": vq_loss,
            "num_patches": P,
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        curve_mask: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
        return_token_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass. API compatible with WellLogEncoder1D.

        Args:
            x: (B, C, L) well log curves
            mask: Optional (B, L) depth mask (adapted to patch level)
            curve_mask: Optional (B, C) binary mask, 1=valid curve, 0=missing
            value_mask: Optional (B, C, L) per-value validity mask
            return_sequence: Return full patch sequence features
            return_token_info: Return tokenization details

        Returns:
            global_features: (B, embed_dim)
            sequence_features: (B, P, embed_dim) if return_sequence, else None
        """
        x = self._apply_curve_mask(x, curve_mask)

        # Per-well normalization
        x_norm = self._per_well_normalize(
            x,
            curve_mask=curve_mask,
            depth_mask=mask,
            value_mask=value_mask,
        )

        # Segment into patches
        patches, P = self._segment_patches(x_norm)  # (B, P, C, patch_len)

        # Tokenize patches through VQ-VAE
        tokenized = self.tokenize(
            x,
            curve_mask=curve_mask,
            depth_mask=mask,
            value_mask=value_mask,
        )
        tokens = tokenized["tokens"]  # (B, P, vq_embed_dim)
        vq_loss = tokenized["vq_loss"]

        # Project tokens to transformer dimension
        seq_feat = self.token_proj(tokens)  # (B, P, embed_dim)

        # Physical constraint encoding (optional)
        if self.use_physics_constraint and self.physics_encoder is not None:
            # Compute physics features from patch statistics
            phys_features = []
            for p in range(P):
                patch_mean = patches[:, p, :, :].mean(dim=-1)  # (B, C)
                phys_feat = self.physics_encoder(patch_mean)  # (B, embed_dim)
                phys_features.append(phys_feat)
            phys_features = torch.stack(phys_features, dim=1)  # (B, P, embed_dim)
            seq_feat = self.phys_fuse(
                torch.cat([seq_feat, phys_features], dim=-1)
            )

        # Position is added after modality-specific fusion so Stage 1 MTM and
        # the regular encoder consume the same representation distribution.
        seq_feat = self.depth_encoding(seq_feat)

        # Adapt mask to patch level
        attn_mask = None
        if mask is not None:
            # Downsample mask to patch level
            patch_mask = F.max_pool1d(
                mask.float().unsqueeze(1), self.patch_len, self.patch_stride
            ).squeeze(1)
            attn_mask = (patch_mask > 0).float()

        # Transformer blocks
        for blk in self.transformer_blocks:
            seq_feat = blk(seq_feat)

        seq_feat = self.encoder_norm(seq_feat)

        # Global pooling
        global_feat = self.attn_pool(seq_feat, attn_mask)
        global_feat = self.global_proj(global_feat)
        global_feat = self.output_norm(global_feat)

        if return_token_info:
            return global_feat, seq_feat if return_sequence else None, tokenized
        return global_feat, seq_feat if return_sequence else None

    def forward_mtm(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.5,
        curve_mask: Optional[torch.Tensor] = None,
        depth_mask: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Masked Token Modeling (MTM) forward pass.

        Randomly masks well log patches and predicts the discrete codebook
        indices of masked positions. This is the core self-supervised
        pretraining task of WLFM.

        Args:
            x: (B, C, L) well log curves
            mask_ratio: Fraction of patches to mask (default 0.5)
            curve_mask: Optional (B, C) available-curve mask
            depth_mask: Optional (B, L) valid-depth mask
            value_mask: Optional (B, C, L) per-value validity mask

        Returns:
            dict with:
                - logits: (B, P, num_embeddings) token prediction logits
                - target_indices: (B, P) ground truth codebook indices
                - mask: (B, P) boolean mask (True = masked)
                - loss: MTM cross-entropy loss
        """
        B, C, L = x.shape

        x = self._apply_curve_mask(x, curve_mask)

        # Normalize and segment
        x_norm = self._per_well_normalize(
            x,
            curve_mask=curve_mask,
            depth_mask=depth_mask,
            value_mask=value_mask,
        )
        patches, P = self._segment_patches(x_norm)

        # Tokenize all patches (get ground truth indices)
        tokenized = self.tokenize(
            x,
            curve_mask=curve_mask,
            depth_mask=depth_mask,
            value_mask=value_mask,
        )
        target_indices = tokenized["indices"].detach()  # (B, P)
        tokens = tokenized["tokens"]  # (B, P, vq_embed_dim)

        # Patches without valid depth samples must not contribute to MTM.
        if value_mask is not None:
            patch_valid = F.max_pool1d(
                value_mask.float(),
                self.patch_len,
                self.patch_stride,
            ).amax(dim=1) > 0
        elif depth_mask is not None:
            patch_valid = F.max_pool1d(
                depth_mask.float().unsqueeze(1),
                self.patch_len,
                self.patch_stride,
            ).squeeze(1) > 0
        else:
            patch_valid = patches.abs().sum(dim=(2, 3)) > 1e-6
        if curve_mask is not None:
            patch_valid = patch_valid & curve_mask.bool().any(dim=1, keepdim=True)

        # Select random valid patches without changing sequence order. The old
        # implementation shuffled tokens but compared them with unshuffled
        # targets, making the classification labels incorrect.
        mask = torch.zeros(B, P, device=x.device, dtype=torch.bool)
        noise = torch.rand(B, P, device=x.device)
        for b in range(B):
            valid_indices = torch.where(patch_valid[b])[0]
            if valid_indices.numel() == 0:
                continue
            num_mask = max(1, int(valid_indices.numel() * mask_ratio))
            num_mask = min(num_mask, valid_indices.numel())
            chosen = valid_indices[
                torch.argsort(noise[b, valid_indices])[:num_mask]
            ]
            mask[b, chosen] = True

        # Build the same token + physics representation used by forward().
        seq_feat = self.token_proj(tokens)  # (B, P, embed_dim)
        if self.use_physics_constraint and self.physics_encoder is not None:
            phys_features = torch.stack(
                [
                    self.physics_encoder(patches[:, p].mean(dim=-1))
                    for p in range(P)
                ],
                dim=1,
            )
            seq_feat = self.phys_fuse(
                torch.cat([seq_feat, phys_features], dim=-1)
            )

        # Replace masked positions with mask token
        mask_token_expanded = self.mask_token.expand(B, P, -1)
        seq_feat = torch.where(
            mask.unsqueeze(-1), mask_token_expanded, seq_feat
        )

        seq_feat = self.depth_encoding(seq_feat)

        # Encode through transformer
        for blk in self.transformer_blocks:
            seq_feat = blk(seq_feat)
        seq_feat = self.encoder_norm(seq_feat)

        # Make MTM depend on the same global representation consumed by Stage
        # 2, so attention pooling and global projection are pretrained too.
        pooling_mask = patch_valid.clone()
        empty = ~pooling_mask.any(dim=1)
        if empty.any():
            pooling_mask[empty] = True
        global_feat = self.attn_pool(seq_feat, pooling_mask.float())
        global_feat = self.output_norm(self.global_proj(global_feat))
        logits = self.mtm_head(
            seq_feat + global_feat.unsqueeze(1)
        )  # (B, P, num_embeddings)

        # Compute loss on aligned masked positions and include the tokenizer's
        # commitment loss to prevent encoder/codebook drift.
        train_mask = mask & patch_valid
        if train_mask.any():
            mtm_loss = F.cross_entropy(
                logits[train_mask],
                target_indices[train_mask],
            )
        else:
            mtm_loss = logits.sum() * 0.0
        vq_loss = tokenized["vq_loss"]
        loss = mtm_loss + vq_loss

        return {
            "logits": logits,
            "target_indices": target_indices,
            "mask": mask,
            "loss": loss,
            "mtm_loss": mtm_loss,
            "vq_loss": vq_loss,
            "patch_valid": patch_valid,
        }

    def forward_scl(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Stratigraphy-aware Contrastive Learning (SCL) projection.

        Projects well log features to a contrastive embedding space
        where patches from the same stratigraphic interval are close.

        Args:
            x: (B, C, L) well log curves

        Returns:
            projections: (B, P, 128) L2-normalized contrastive embeddings
        """
        global_feat, seq_feat = self.forward(x, return_sequence=True)

        # Project to contrastive space
        projections = self.scl_proj(seq_feat)  # (B, P, 128)
        projections = F.normalize(projections, dim=-1)

        return projections

    def get_output_dim(self) -> int:
        """Return the output dimension."""
        return self.embed_dim

    def get_codebook(self) -> torch.Tensor:
        """Return the geological vocabulary codebook."""
        return self.tokenizer.get_codebook()

    def get_num_codebook_entries(self) -> int:
        """Return codebook size."""
        return self.num_embeddings

    # ==========================================================================
    # Industrial Inference: Variable Length + Missing Curves
    # ==========================================================================

    def infer_variable_length(
        self,
        x: torch.Tensor,
        curve_mask: Optional[torch.Tensor] = None,
        max_patches: int = 64,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode well logs of ANY length (not constrained to training seq_len).

        Dynamically segments the input into patches, processes them through
        the VQ-VAE tokenizer and transformer, and pools to a fixed-size output.

        Args:
            x: (B, C, L) well log curves, L can be any value
            curve_mask: (B, C) binary mask, 1=valid curve, 0=missing
            max_patches: Maximum number of patches (truncates long sequences)

        Returns:
            global_feat: (B, embed_dim)
            seq_feat: (B, P, embed_dim) or None
        """
        B, C, L = x.shape

        # Per-well normalize
        x_norm = self._per_well_normalize(x)

        # Segment into patches (L can vary)
        patches, P = self._segment_patches(x_norm)
        P = min(P, max_patches)  # Truncate long sequences
        patches = patches[:, :P, :, :]

        # Handle missing curves: replace NaN channels with zeros + flag embedding
        if curve_mask is not None:
            # Zero out missing curves
            curve_mask_exp = curve_mask.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            patches = patches * curve_mask_exp.unsqueeze(1)  # (B, P, C, patch_len)

        # Tokenize each patch
        all_tokens = []
        for p in range(P):
            patch = patches[:, p, :, :]
            result = self.tokenizer(patch, return_indices=False)
            all_tokens.append(result["tokens"].squeeze(1))

        tokens = torch.stack(all_tokens, dim=1)  # (B, P, vq_embed_dim)

        # Project to transformer dim
        seq_feat = self.token_proj(tokens)

        # Add position encoding (use first P positions)
        seq_feat = self.depth_encoding(seq_feat)

        # Physical constraint encoding
        if self.use_physics_constraint and self.physics_encoder is not None:
            phys_features = []
            for p in range(P):
                patch_mean = patches[:, p, :, :].mean(dim=-1)  # (B, C)
                phys_feat = self.physics_encoder(patch_mean)
                phys_features.append(phys_feat)
            phys = torch.stack(phys_features, dim=1)
            seq_feat = self.phys_fuse(torch.cat([seq_feat, phys], dim=-1))

        # Transformer
        for blk in self.transformer_blocks:
            seq_feat = blk(seq_feat)
        seq_feat = self.encoder_norm(seq_feat)

        # Global pooling
        global_feat = self.attn_pool(seq_feat)
        global_feat = self.global_proj(global_feat)
        global_feat = self.output_norm(global_feat)

        return global_feat, seq_feat

    def forward_with_curve_dropout(
        self,
        x: torch.Tensor,
        dropout_rate: float = 0.3,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Training with curve dropout — randomly drops entire well log curves.

        This teaches the model to be robust to missing measurements
        (e.g., no DT curve, failed RHOB tool, etc.) at inference time.

        During training: randomly mask out some curves (set to zero).
        During inference: pass curve_mask with 0 for truly missing curves.

        Args:
            x: (B, C, L) well log curves
            dropout_rate: Probability of dropping each curve independently
            mask: Optional pre-existing mask to combine with dropout
            return_sequence: Return full sequence

        Returns:
            global_feat: (B, embed_dim)
            seq_feat: (B, P, embed_dim) or None
        """
        B, C, L = x.shape

        # Generate random curve dropout mask
        if self.training and dropout_rate > 0:
            keep_prob = 1.0 - dropout_rate
            rand_mask = torch.bernoulli(
                torch.full((B, C), keep_prob, device=x.device)
            )
            if mask is not None:
                rand_mask = rand_mask * mask
        else:
            rand_mask = mask if mask is not None else torch.ones(B, C, device=x.device)

        # Apply mask: zero out dropped curves
        x_masked = x * rand_mask.unsqueeze(-1)

        # Encode with masked input
        return self.infer_variable_length(x_masked, curve_mask=rand_mask)

    def infer_missing_curves(
        self,
        x: torch.Tensor,
        available_curves: List[str],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inference with a SUBSET of the training curves available.

        Common in industry: some logging tools fail or aren't run.
        The model was trained with curve dropout, so it handles missing data.

        Args:
            x: (B, n_available, L) available curves only
            available_curves: Names of available curves (e.g. ['GR', 'RT', 'RHOB'])

        Returns:
            global_feat, seq_feat
        """
        # Build full input with zeros for missing curves
        B, _, L = x.shape
        full_x = torch.zeros(B, self.num_curves, L, device=x.device, dtype=x.dtype)
        curve_mask = torch.zeros(B, self.num_curves, device=x.device)

        for i, name in enumerate(available_curves):
            if name in self.curve_names:
                idx = self.curve_names.index(name)
                if i < x.shape[1]:
                    full_x[:, idx, :] = x[:, i, :]
                    curve_mask[:, idx] = 1.0

        return self.infer_variable_length(full_x, curve_mask=curve_mask)

    def few_shot_adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_steps: int = 50,
        lr: float = 1e-4,
    ) -> torch.Tensor:
        """
        Few-shot adaptation: fine-tune encoder on a few labeled samples.

        Common scenario: few labeled wells in a new field.
        Adapts the encoder by minimizing prediction error on support set.

        Args:
            support_x: (N_support, C, L) support well logs
            support_y: (N_support, L) support labels (e.g., lithology)
            query_x: (N_query, C, L) query well logs
            n_steps: Number of adaptation steps
            lr: Learning rate

        Returns:
            query_pred: (N_query, L, num_classes) predictions
        """
        # Clone encoder for fast adaptation
        import copy
        adapted = copy.deepcopy(self)
        adapted.train()

        opt = torch.optim.AdamW(adapted.parameters(), lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            _, seq_feat = adapted(support_x, return_sequence=True)

            # Simple linear probe
            pred = nn.Linear(seq_feat.shape[-1], support_y.max().long().item() + 1)
            pred = pred.to(support_x.device)
            loss = F.cross_entropy(
                pred(seq_feat).reshape(-1, pred.out_features),
                support_y.reshape(-1).long(),
            )
            loss.backward()
            opt.step()

        # Predict on query
        adapted.eval()
        with torch.no_grad():
            _, query_seq = adapted(query_x, return_sequence=True)
            pred = pred.to(query_x.device)
            query_logits = pred(query_seq)

        return query_logits


# ==============================================================================
# Pretrained Model Loading
# ==============================================================================

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Optional[str] = None,
        pretrained_name: Optional[str] = None,
        num_curves: int = 7,
        max_seq_len: int = 512,
        **kwargs,
    ) -> "WLFMWellLogEncoder1D":
        """
        Load pretrained WLFM weights.

        NOTE: As of 2025-07, the official WLFM pretrained weights are NOT yet
        publicly released on HuggingFace or GitHub. The paper (arXiv:2509.18152)
        mentions code availability but weights haven't been published.

        This method supports:
        1. Loading from a local checkpoint (.pt / .pth file)
        2. Loading from HuggingFace when weights become available
        3. Building from scratch with the architecture (fallback)

        Usage:
            # From local checkpoint
            encoder = WLFMWellLogEncoder1D.from_pretrained(
                pretrained_path="./checkpoints/wlfm_pretrained.pt"
            )

            # From HuggingFace (when available)
            encoder = WLFMWellLogEncoder1D.from_pretrained(
                pretrained_name="ustc/wlfm-base"
            )

            # Build from scratch (default)
            encoder = WLFMWellLogEncoder1D.from_pretrained(variant="base")

        Args:
            pretrained_path: Local path to checkpoint file
            pretrained_name: HuggingFace model ID (when available)
            num_curves: Number of well log curves
            max_seq_len: Maximum sequence length
            **kwargs: Additional config overrides (variant, embed_dim, etc.)

        Returns:
            WLFMWellLogEncoder1D with pretrained weights if available
        """
        import logging
        logger = logging.getLogger(__name__)

        variant = kwargs.pop("variant", "base")

        # Try local checkpoint
        if pretrained_path is not None and pretrained_path.strip():
            try:
                logger.info(f"Loading WLFM from local checkpoint: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location="cpu")

                # Extract config from checkpoint if available
                config = checkpoint.get("config", {})
                cfg = {
                    "num_curves": config.get("num_curves", num_curves),
                    "embed_dim": config.get("embed_dim", 256),
                    "vq_embed_dim": config.get("vq_embed_dim", 256),
                    "num_embeddings": config.get("num_embeddings", 512),
                    "num_layers": config.get("num_layers", 8),
                    "num_heads": config.get("num_heads", 8),
                    "mlp_ratio": config.get("mlp_ratio", 4.0),
                }
                cfg.update(kwargs)

                encoder = cls(**cfg, max_seq_len=max_seq_len)

                # Load weights
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                encoder.load_state_dict(state_dict, strict=False)
                logger.info("Loaded WLFM weights from local checkpoint")
                return encoder
            except Exception as e:
                logger.warning(f"Failed to load local checkpoint: {e}")

        # Try HuggingFace (for future use)
        if pretrained_name is not None:
            try:
                from huggingface_hub import hf_hub_download
                import json

                logger.info(f"Attempting to load WLFM from HuggingFace: {pretrained_name}")

                # Try to get config
                try:
                    config_path = hf_hub_download(pretrained_name, "config.json")
                    with open(config_path, "r") as f:
                        config = json.load(f)
                except Exception:
                    config = build_wlfm_encoder(variant).config if hasattr(
                        build_wlfm_encoder(variant), 'config'
                    ) else {}

                cfg = {
                    "num_curves": config.get("num_curves", num_curves),
                    "embed_dim": config.get("embed_dim", 256),
                    "vq_embed_dim": config.get("vq_embed_dim", 256),
                    "num_embeddings": config.get("num_embeddings", 512),
                    "num_layers": config.get("num_layers", 8),
                    "num_heads": config.get("num_heads", 8),
                    "mlp_ratio": config.get("mlp_ratio", 4.0),
                }
                cfg.update(kwargs)

                encoder = cls(**cfg, max_seq_len=max_seq_len)

                # Download and load weights
                try:
                    model_path = hf_hub_download(pretrained_name, "pytorch_model.bin")
                    state_dict = torch.load(model_path, map_location="cpu")
                except Exception:
                    from safetensors.torch import load_file
                    model_path = hf_hub_download(pretrained_name, "model.safetensors")
                    state_dict = load_file(model_path)

                encoder.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded WLFM from HuggingFace: {pretrained_name}")
                return encoder
            except Exception as e:
                logger.warning(f"WLFM not available on HuggingFace yet: {e}")

        # Fallback: Build from scratch
        logger.info(
            "Building WLFM encoder from scratch. "
            "Pretrained weights are not publicly available as of 2025-07. "
            "The model will be randomly initialized. "
            "Check https://arxiv.org/abs/2509.18152 for updates on weight release."
        )
        return build_wlfm_encoder(
            variant=variant,
            num_curves=num_curves,
            max_seq_len=max_seq_len,
            **kwargs,
        )

    def save_pretrained(self, save_path: str):
        """
        Save encoder weights and config to disk.

        Args:
            save_path: Path to save checkpoint (.pt file)
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "num_curves": self.num_curves,
                "embed_dim": self.embed_dim,
                "vq_embed_dim": self.vq_embed_dim,
                "num_embeddings": self.num_embeddings,
                "num_layers": len(self.transformer_blocks),
                "num_heads": self.transformer_blocks[0].attn.num_heads
                if hasattr(self.transformer_blocks[0].attn, 'num_heads')
                else 8,
                "patch_len": self.patch_len,
                "patch_stride": self.patch_stride,
            },
        }
        torch.save(checkpoint, save_path)


# ==============================================================================
# Factory function for WLFM encoder variants
# ==============================================================================

def build_wlfm_encoder(
    variant: str = "base",
    num_curves: int = 7,
    max_seq_len: int = 512,
    **kwargs,
) -> WLFMWellLogEncoder1D:
    """
    Build a WLFM encoder with predefined configurations.

    Variants:
        - 'tiny':  embed_dim=128, layers=4, heads=4, codebook=256  ~3M params
        - 'small': embed_dim=192, layers=6, heads=6, codebook=512  ~8M params
        - 'base':  embed_dim=256, layers=8, heads=8, codebook=512  ~18M params

    Args:
        variant: Model size variant
        num_curves: Number of well log curves
        max_seq_len: Maximum sequence length
        **kwargs: Additional overrides

    Returns:
        Configured WLFMWellLogEncoder1D
    """
    configs = {
        "tiny": {
            "embed_dim": 128,
            "vq_embed_dim": 128,
            "num_embeddings": 256,
            "num_layers": 4,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        },
        "small": {
            "embed_dim": 192,
            "vq_embed_dim": 256,
            "num_embeddings": 512,
            "num_layers": 6,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
        "base": {
            "embed_dim": 256,
            "vq_embed_dim": 256,
            "num_embeddings": 512,
            "num_layers": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
        },
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(configs.keys())}")

    cfg = configs[variant]
    cfg.update(kwargs)

    return WLFMWellLogEncoder1D(
        num_curves=num_curves,
        max_seq_len=max_seq_len,
        **cfg,
    )
