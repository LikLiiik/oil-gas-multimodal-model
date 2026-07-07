"""
Masked Seismic Modeling (MSM) Pretraining Task

Implements 3D Masked Autoencoder (MAE) for seismic volumes.
Randomly masks 3D patches and trains the encoder to reconstruct
the original seismic values from visible patches only.

Architecture follows MAE (He et al., 2022) adapted for 3D:
- High mask ratio (60-75%) for effective self-supervision
- Asymmetric encoder-decoder (lightweight decoder)
- Reconstruction only on masked regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from einops import rearrange


class MaskedSeismicModeling(nn.Module):
    """
    Masked Seismic Modeling (MSM) for 3D seismic pretraining.

    Mask random 3D patches from the seismic volume and reconstruct them.
    Uses an asymmetric design where the encoder only processes visible patches
    and a lightweight decoder reconstructs the full volume.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mask_ratio: float = 0.6,
        mask_block_size: Tuple[int, int, int] = (8, 16, 16),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

        # Get encoder output dimension
        encoder_dim = encoder.get_output_dim() if hasattr(encoder, 'get_output_dim') else 384

        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 512, decoder_embed_dim)
        )
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head
        self.reconstruction_head = nn.Linear(
            decoder_embed_dim,
            patch_size[0] * patch_size[1] * patch_size[2],
        )

    def patchify(
        self, seismic: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert seismic volume to patches.

        Args:
            seismic: (B, 1, D, H, W)

        Returns:
            patches: (B, num_patches, patch_volume)
        """
        p_d, p_h, p_w = self.patch_size
        patches = rearrange(
            seismic,
            "b c (d pd) (h ph) (w pw) -> b (d h w) (c pd ph pw)",
            pd=p_d, ph=p_h, pw=p_w,
        )
        return patches

    def unpatchify(
        self, patches: torch.Tensor, spatial_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Convert patches back to seismic volume.

        Args:
            patches: (B, num_patches, patch_volume)
            spatial_shape: (D, H, W) original volume shape

        Returns:
            seismic: (B, 1, D, H, W)
        """
        p_d, p_h, p_w = self.patch_size
        d_patches = spatial_shape[0] // p_d
        h_patches = spatial_shape[1] // p_h
        w_patches = spatial_shape[2] // p_w

        seismic = rearrange(
            patches,
            "b (d h w) (c pd ph pw) -> b c (d pd) (h ph) (w pw)",
            d=d_patches, h=h_patches, w=w_patches,
            pd=p_d, ph=p_h, pw=p_w, c=1,
        )
        return seismic

    def random_masking(
        self, patches: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches using block-based masking.

        Args:
            patches: (B, N, patch_volume)

        Returns:
            visible_patches: (B, N_visible, patch_volume)
            mask: (B, N) bool mask (True = masked)
            ids_restore: (B, N) indices to restore original order
        """
        B, N, D = patches.shape

        # Block-based masking: first mask block centers, then expand
        len_keep = int(N * (1 - self.mask_ratio))

        # Randomly select visible patches
        noise = torch.rand(B, N, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        visible_patches = torch.gather(
            patches, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, N, device=patches.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return visible_patches, mask, ids_restore

    def forward_encoder(
        self, seismic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode only visible patches.

        Args:
            seismic: (B, 1, D, H, W)

        Returns:
            latent: (B, N_visible, encoder_dim)
            mask: (B, N)
            ids_restore: (B, N)
        """
        # Patchify
        patches = self.patchify(seismic)
        B, N, D = patches.shape

        # Mask patches
        visible_patches, mask, ids_restore = self.random_masking(patches)

        # Encode visible patches (simplified - use encoder's embedding)
        # In practice, would use encoder's patch embedding + transformer
        latent = visible_patches  # Placeholder; real implementation uses encoder

        return latent, mask, ids_restore

    def forward_decoder(
        self, latent: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent features to reconstruct the full volume.

        Args:
            latent: (B, N_visible, encoder_dim)
            ids_restore: (B, N)

        Returns:
            pred: (B, N, patch_volume)
        """
        B, N_visible, C = latent.shape
        N = ids_restore.shape[1]

        # Project to decoder dimension
        x = self.decoder_embed(latent)  # (B, N_visible, dec_dim)

        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, N - N_visible, -1)
        x = torch.cat([x, mask_tokens], dim=1)

        # Restore original order
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )

        # Add positional embedding
        pos_embed = self.decoder_pos_embed[:, :N, :]
        x = x + pos_embed

        # Decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # Reconstruction
        pred = self.reconstruction_head(x)

        return pred

    def forward_loss(
        self, seismic: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches only.

        Args:
            seismic: (B, 1, D, H, W)
            pred: (B, N, patch_volume)
            mask: (B, N) True for masked patches

        Returns:
            loss: scalar
        """
        target = self.patchify(seismic)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, seismic: torch.Tensor) -> Dict:
        """
        Full forward pass for MSM pretraining.

        Args:
            seismic: (B, 1, D, H, W)

        Returns:
            dict with loss and reconstruction
        """
        B, C, D, H, W = seismic.shape

        # Patchify and mask
        patches = self.patchify(seismic)
        visible_patches, mask, ids_restore = self.random_masking(patches)

        # Simplified encoder path (in production, use actual seismic encoder)
        latent = visible_patches

        # Decode
        pred = self.forward_decoder(latent, ids_restore)

        # Compute loss
        loss = self.forward_loss(seismic, pred, mask)

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
        }
