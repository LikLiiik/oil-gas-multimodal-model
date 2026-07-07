"""
Cross-Modal Fusion Module

Fuses features from 3D seismic and 1D well log modalities using:
1. Coarse-grained fusion: Global feature concatenation + MLP
2. Fine-grained fusion: Bi-directional Cross-Attention
3. Adaptive Fusion Gate: Learned weighting of modality contributions

Architecture:
    seismic_feat (B, hidden_dim)  +  well_feat (B, hidden_dim)
    -> Coarse Fusion (MLP)
    -> Cross-Attention (seismic as Q, well as K/V; and vice versa)
    -> Fusion Gate (dynamic weighting)
    -> Output: (B, hidden_dim) fused representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from einops import rearrange


# =====================================================================
# Cross-Attention Layer
# =====================================================================

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention between two modalities.
    One modality provides queries, the other provides keys and values.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, L_q, dim) or (B, dim)
            keys: (B, L_kv, dim) or (B, dim)
            values: (B, L_kv, dim) or (B, dim)

        Returns:
            (B, dim) attended features
        """
        # Ensure 3D tensors
        if queries.dim() == 2:
            queries = queries.unsqueeze(1)  # (B, 1, dim)
        if keys.dim() == 2:
            keys = keys.unsqueeze(1)
        if values.dim() == 2:
            values = values.unsqueeze(1)

        B = queries.shape[0]

        # Layer norm
        q = self.norm_q(queries)
        k = self.norm_kv(keys)
        v = self.norm_kv(values)

        # Project
        q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L_q, L_kv)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        x = attn @ v  # (B, H, L_q, head_dim)
        x = x.transpose(1, 2).reshape(B, -1, self.dim)  # (B, L_q, dim)
        x = self.out_proj(x)

        # Pool to (B, dim)
        x = x.mean(dim=1)

        return x


# =====================================================================
# Bi-directional Cross-Attention Fusion
# =====================================================================

class BiCrossAttention(nn.Module):
    """
    Bi-directional cross-attention fusion.

    Implements two cross-attention paths:
    1. Seismic features attend to well log features
    2. Well log features attend to seismic features

    This enriches each modality with information from the other.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # Seismic -> Well cross-attention layers
        self.s2w_layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Well -> Seismic cross-attention layers
        self.w2s_layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Self-attention layers for fused features
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seismic_feat: (B, dim) global seismic features
            well_feat: (B, dim) global well log features

        Returns:
            s2w_feat: (B, dim) seismic enhanced by well info
            w2s_feat: (B, dim) well enhanced by seismic info
            fused_feat: (B, dim) combined fused features
        """
        s_feat = seismic_feat
        w_feat = well_feat

        for i in range(self.num_layers):
            # Seismic attends to well logs
            s_enhanced = self.s2w_layers[i](s_feat, w_feat, w_feat)

            # Well logs attend to seismic
            w_enhanced = self.w2s_layers[i](w_feat, s_feat, s_feat)

            # Combine and self-attend
            combined = torch.stack([s_enhanced, w_enhanced], dim=1)  # (B, 2, dim)
            combined_norm = self.norm(combined)
            combined, _ = self.self_attn[i](
                combined_norm, combined_norm, combined_norm
            )

            s_feat = combined[:, 0, :]
            w_feat = combined[:, 1, :]

        # Final fused feature
        fused = s_feat + w_feat

        return s_feat, w_feat, fused


# =====================================================================
# Coarse-grained Fusion
# =====================================================================

class CoarseFusion(nn.Module):
    """
    Coarse-grained fusion using feature concatenation + MLP.

    Simple but effective global fusion strategy.
    """

    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)

        # Feature interaction terms
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),  # [s; w; s*w]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seismic_feat: (B, dim)
            well_feat: (B, dim)

        Returns:
            (B, dim) coarse fused features
        """
        # Feature interaction: [s; w; s * w]
        interaction = seismic_feat * well_feat
        concat = torch.cat([seismic_feat, well_feat, interaction], dim=-1)

        fused = self.fc(concat)
        fused = self.norm(fused + seismic_feat + well_feat)  # Residual

        return fused


# =====================================================================
# Adaptive Fusion Gate
# =====================================================================

class AdaptiveFusionGate(nn.Module):
    """
    Learnable gate mechanism that dynamically adjusts
    the contribution weight of each modality.

    α = sigmoid(MLP([seismic_feat; well_feat; cross_feat]))
    output = α * seismic_feat + (1-α) * well_feat + cross_feat
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
        cross_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seismic_feat: (B, dim)
            well_feat: (B, dim)
            cross_feat: (B, dim) from cross-attention

        Returns:
            (B, dim) gated fused features
        """
        # Compute gate values
        gate_input = torch.cat([seismic_feat, well_feat, cross_feat], dim=-1)
        alpha = self.gate_mlp(gate_input)  # (B, dim)

        # Weighted fusion
        gated = alpha * seismic_feat + (1 - alpha) * well_feat + cross_feat

        # Final projection
        output = self.output_proj(gated)
        output = self.norm(output)

        return output, alpha


# =====================================================================
# Main Cross-Modal Fusion Module
# =====================================================================

class CrossModalFusion(nn.Module):
    """
    Complete Cross-Modal Fusion Module.

    Combines coarse-grained fusion, bi-directional cross-attention,
    and adaptive gating for robust multi-modal representation learning.

    Args:
        hidden_dim: Feature dimension
        num_cross_attention_heads: Number of attention heads
        num_fusion_layers: Number of fusion layers
        dropout: Dropout rate
        use_gating: Enable adaptive fusion gate
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_cross_attention_heads: int = 8,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        use_gating: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gating = use_gating

        # Coarse fusion
        self.coarse_fusion = CoarseFusion(
            dim=hidden_dim,
            dropout=dropout,
        )

        # Fine-grained cross-attention fusion
        self.cross_attn = BiCrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attention_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
        )

        # Adaptive gating
        if use_gating:
            self.fusion_gate = AdaptiveFusionGate(
                dim=hidden_dim,
                dropout=dropout,
            )

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse seismic and well log features.

        Args:
            seismic_feat: (B, hidden_dim) global seismic features
            well_feat: (B, hidden_dim) global well log features
            return_details: If True, return intermediate fusion details

        Returns:
            fused_feat: (B, hidden_dim)
            If return_details: (fused_feat, details_dict)
        """
        # 1. Coarse fusion
        coarse_feat = self.coarse_fusion(seismic_feat, well_feat)

        # 2. Fine-grained cross-attention
        s_enhanced, w_enhanced, cross_feat = self.cross_attn(
            seismic_feat, well_feat
        )

        # 3. Gated fusion
        if self.use_gating:
            fused, gate_weights = self.fusion_gate(
                coarse_feat, cross_feat, s_enhanced + w_enhanced
            )
        else:
            fused = coarse_feat + cross_feat
            gate_weights = None

        # 4. Final projection
        output = self.final_proj(fused)
        output = self.output_norm(output + fused)  # Residual connection

        if return_details:
            details = {
                "coarse_feat": coarse_feat,
                "cross_feat": cross_feat,
                "s_enhanced": s_enhanced,
                "w_enhanced": w_enhanced,
            }
            if gate_weights is not None:
                details["gate_weights"] = gate_weights
            return output, details

        return output


# =====================================================================
# Modality Projection Layer
# =====================================================================

class ModalityProjection(nn.Module):
    """
    Projects encoder outputs to a common dimension for fusion.

    Ensures seismic and well log features are in the same space
    before cross-modal fusion.
    """

    def __init__(
        self,
        seismic_dim: int,
        well_log_dim: int,
        common_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seismic_proj = nn.Sequential(
            nn.Linear(seismic_dim, common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(common_dim),
        )

        self.well_log_proj = nn.Sequential(
            nn.Linear(well_log_dim, common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(common_dim),
        )

    def forward(
        self,
        seismic_feat: torch.Tensor,
        well_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project both modalities to common dimension.

        Args:
            seismic_feat: (B, seismic_dim)
            well_feat: (B, well_log_dim)

        Returns:
            seismic_proj: (B, common_dim)
            well_proj: (B, common_dim)
        """
        return (
            self.seismic_proj(seismic_feat),
            self.well_log_proj(well_feat),
        )
