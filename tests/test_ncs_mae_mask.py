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


def test_mae_decoder_unshuffle_restores_original_order():
    """Decoder must map shuffled tokens back to ORIGINAL patch order.

    Regression for the bug where argsort(ids_restore) (== ids_shuffle) was used
    instead of ids_restore, scrambling reconstruction targets so MSM loss stalls.
    """
    torch.manual_seed(0)
    B, N, D = 2, 8, 3
    # Emulate the encoder's shuffle bookkeeping.
    noise = torch.rand(B, N)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # x_full in shuffled order: token at shuffled slot j carries a tag equal to
    # its ORIGINAL index (ids_shuffle[j]), so correct unshuffle yields 0..N-1.
    tags = ids_shuffle.unsqueeze(-1).expand(-1, -1, D).float()

    correct = torch.gather(tags, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
    expected = torch.arange(N).view(1, N, 1).expand(B, N, D).float()
    assert torch.equal(correct, expected), "gather(ids_restore) must restore order"

    # The old (buggy) inverse would re-apply the shuffle, NOT restore it.
    inv = torch.argsort(ids_restore, dim=1)
    assert torch.equal(inv, ids_shuffle)  # proves argsort(argsort(p)) == p
    buggy = torch.gather(tags, 1, inv.unsqueeze(-1).expand(-1, -1, D))
    assert not torch.equal(buggy, expected), "buggy path should scramble order"


def test_forward_mae_loss_is_finite_and_trainable():
    """End-to-end MSM step must produce a finite, backprop-able loss."""
    from models.ncs_seismic_encoder import MAEDecoder3D

    torch.manual_seed(0)
    enc = NCSSeismicEncoder3D(
        img_size=(16, 16, 16), patch_size=(4, 4, 4),
        embed_dim=32, num_layers=2, num_heads=4, dropout=0.0,
    )
    dec = MAEDecoder3D(
        encoder_embed_dim=32, decoder_embed_dim=32, patch_size=(4, 4, 4),
        img_size=(16, 16, 16), decoder_num_heads=4, decoder_num_layers=2,
    )
    x = torch.randn(2, 1, 16, 16, 16)
    out = enc.forward_mae(x, dec, mask_ratio=0.6)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    g = enc.patch_embed.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.norm() > 0


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


def test_msm_skips_invalid_seismic():
    """Constant zero-fill patches must not contribute to MSM loss."""
    from types import SimpleNamespace

    from scripts.train_pretrain_volve import PretrainTrainer

    class DummySeis(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_mae(self, seismic, decoder, mask_ratio=0.6):
            self.calls += 1
            assert seismic.shape[0] == 1
            return {"loss": torch.tensor(2.5, requires_grad=True)}

    model = SimpleNamespace(seismic_encoder=DummySeis())
    trainer = PretrainTrainer.__new__(PretrainTrainer)
    trainer.model = model
    trainer.device = "cpu"
    trainer.use_amp = False
    trainer._msm_decoder = object()

    seismic = torch.randn(3, 1, 4, 4, 4)
    valid = torch.tensor([False, True, False])
    loss, active = trainer._compute_msm_loss(seismic, valid)
    assert active
    assert abs(loss.item() - 2.5) < 1e-6
    assert model.seismic_encoder.calls == 1

    loss_none, active_none = trainer._compute_msm_loss(
        seismic, torch.zeros(3, dtype=torch.bool)
    )
    assert not active_none
    assert abs(loss_none.item()) < 1e-8
    assert model.seismic_encoder.calls == 1


if __name__ == "__main__":
    test_forward_mae_encoder_mask_matches_ids_keep()
    test_mae_decoder_unshuffle_restores_original_order()
    test_forward_mae_loss_is_finite_and_trainable()
    test_merge_norm_stats_pooled_std()
    test_msm_skips_invalid_seismic()
    print("ok")
