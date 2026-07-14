"""Regression: MAE mask must align with randomly kept patches."""

import torch

from models.ncs_seismic_encoder import NCSSeismicEncoder3D


def test_forward_mae_encoder_mask_matches_ids_keep():
    enc = NCSSeismicEncoder3D(
        in_channels=1,
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        use_checkpoint=False,
    )
    enc.eval()
    x = torch.randn(2, 1, 16, 16, 16)
    mask_ratio = 0.75

    with torch.no_grad():
        visible, mask, ids_restore = enc.forward_mae_encoder(x, mask_ratio=mask_ratio)

    B, N = mask.shape
    len_keep = int(N * (1 - mask_ratio))
    assert visible.shape[1] == len_keep
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [N - len_keep] * B

    # ids_restore maps shuffled -> original; visible slots are first len_keep
    # in shuffled order. After gather, mask[b, ids_keep] must be False.
    ids_shuffle = torch.argsort(ids_restore, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    for b in range(B):
        assert not mask[b, ids_keep[b]].any(), "kept patches must be unmasked"
        assert mask[b, ids_shuffle[b, len_keep:]].all(), "dropped patches must be masked"


def test_merge_norm_stats_pooled_std():
    from data.multimodal_dataset import merge_norm_stats

    stats = merge_norm_stats(
        [
            {"seismic_mean": 0.0, "seismic_std": 1.0, "n_samples": 100},
            {"seismic_mean": 2.0, "seismic_std": 1.0, "n_samples": 100},
        ],
        well_curves=[],
    )
    # Equal mixture of N(0,1) and N(2,1): mean=1, var=1+1=2 → std≈√2
    assert abs(stats["seismic_mean"] - 1.0) < 1e-6
    assert abs(stats["seismic_std"] - (2.0 ** 0.5 + 1e-8)) < 1e-5


if __name__ == "__main__":
    test_forward_mae_encoder_mask_matches_ids_keep()
    test_merge_norm_stats_pooled_std()
    print("ok")
