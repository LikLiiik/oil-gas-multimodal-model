"""Regression tests for WLFM masked-token pretraining."""

import torch

from models.wlfm_well_log_encoder import WLFMWellLogEncoder1D


def _tiny_encoder() -> WLFMWellLogEncoder1D:
    return WLFMWellLogEncoder1D(
        num_curves=3,
        patch_len=8,
        patch_stride=4,
        embed_dim=24,
        vq_embed_dim=16,
        num_embeddings=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        max_seq_len=32,
        use_physics_constraint=True,
    )


def test_normalization_keeps_invalid_values_zero():
    encoder = _tiny_encoder()
    x = torch.tensor(
        [[[1.0, 2.0, 999.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    curve_mask = torch.tensor([[1.0, 1.0, 0.0]])
    value_mask = torch.tensor(
        [[[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )

    normalized = encoder._per_well_normalize(
        x, curve_mask=curve_mask, value_mask=value_mask
    )

    assert normalized[0, 0, 2].item() == 0.0
    assert torch.count_nonzero(normalized[0, 2]).item() == 0
    assert torch.isfinite(normalized).all()


def test_mtm_masks_only_valid_patches_and_trains_stage2_path():
    torch.manual_seed(7)
    encoder = _tiny_encoder()
    encoder.train()
    x = torch.randn(2, 3, 32)
    curve_mask = torch.ones(2, 3)
    value_mask = torch.ones(2, 3, 32)
    value_mask[0, :, 20:] = 0
    x[0, :, 20:] = 0
    depth_mask = value_mask.any(dim=1).float()

    result = encoder.forward_mtm(
        x,
        mask_ratio=0.5,
        curve_mask=curve_mask,
        depth_mask=depth_mask,
        value_mask=value_mask,
    )

    assert result["logits"].shape[:2] == result["target_indices"].shape
    assert torch.all(result["mask"] <= result["patch_valid"])
    assert result["mask"].any(dim=1).all()
    assert torch.isfinite(result["loss"])

    result["loss"].backward()
    # These modules are consumed by the regular Stage 2 encoder path and must
    # receive Stage 1 gradients.
    assert encoder.physics_encoder[0].weight.grad is not None
    assert encoder.phys_fuse[0].weight.grad is not None
    assert encoder.attn_pool.attention[0].weight.grad is not None
    assert encoder.global_proj[0].weight.grad is not None
    assert encoder.tokenizer.input_proj.weight.grad is not None


def test_vq_ema_does_not_create_exploding_unused_entries():
    torch.manual_seed(11)
    encoder = _tiny_encoder()
    encoder.train()
    x = torch.randn(2, 3, 32)

    encoder.forward_mtm(x, mask_ratio=0.5)

    codebook = encoder.get_codebook().detach()
    assert torch.isfinite(codebook).all()
    assert codebook.norm(dim=1).max().item() < 100.0


def test_mtm_targets_align_with_masked_positions():
    """Labels must stay in original patch order (no shuffle desync)."""
    torch.manual_seed(3)
    encoder = _tiny_encoder()
    encoder.eval()  # freeze EMA so tokenize / forward_mtm share indices
    x = torch.randn(2, 3, 32)

    with torch.no_grad():
        tokenized = encoder.tokenize(x)
        result = encoder.forward_mtm(x, mask_ratio=0.5)

    assert torch.equal(result["target_indices"], tokenized["indices"])
    assert result["mask"].shape == result["target_indices"].shape
    # At least one masked valid patch per sample, and loss includes VQ term.
    assert result["mask"].any()
    assert "vq_loss" in result
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["vq_loss"])


def test_trainer_stage1_passes_value_mask():
    """Ensure Stage1 wiring forwards depth/value masks into forward_mtm."""
    from scripts.train_pretrain_volve import PretrainTrainer
    from types import SimpleNamespace

    class DummyWL(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = {}

        def forward_mtm(self, well_log, mask_ratio=0.5, curve_mask=None,
                        depth_mask=None, value_mask=None):
            self.called = {
                "curve_mask": curve_mask is not None,
                "depth_mask": depth_mask is not None,
                "value_mask": value_mask is not None,
                "mask_ratio": mask_ratio,
            }
            return {"loss": well_log.sum() * 0.0 + 1.0}

    class DummySeis(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.zeros(1, 8), None

    model = SimpleNamespace(
        seismic_encoder=DummySeis(),
        well_log_encoder=DummyWL(),
    )
    trainer = PretrainTrainer.__new__(PretrainTrainer)
    trainer.model = model
    trainer.device = "cpu"
    trainer.use_amp = False
    trainer.config = {"stage1_weights": {"msm": 1.0, "mwm": 1.0}}
    trainer.training_stage = 1
    trainer._msm_decoder = None

    def fake_msm(_seismic):
        return torch.tensor(0.5)

    trainer._compute_msm_loss = fake_msm

    batch = {
        "seismic": torch.randn(2, 1, 8, 8, 8),
        "well_log": torch.randn(2, 3, 32),
        "well_mask": torch.ones(2, 32),
        "curve_mask": torch.ones(2, 3),
        "well_value_mask": torch.ones(2, 3, 32),
    }
    losses = trainer._compute_stage1_losses(batch)
    assert model.well_log_encoder.called["curve_mask"]
    assert model.well_log_encoder.called["depth_mask"]
    assert model.well_log_encoder.called["value_mask"]
    assert losses["mwm_loss"].item() == 1.0
