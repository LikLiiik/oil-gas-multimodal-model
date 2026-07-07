"""
Model Framework Tests

Unit tests to verify:
1. Model forward propagation
2. Dimension correctness
3. Gradient flow
4. End-to-end pipeline
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, get_default_config
from models.seismic_encoder import SeismicEncoder3D
from models.well_log_encoder import WellLogEncoder1D
from models.fusion_module import CrossModalFusion, ModalityProjection
from models.prediction_heads import (
    FaultDetectionHead, ReservoirPredictionHead,
    LithologyClassificationHead, MultiTaskHead,
)
from models.oil_gas_model import OilGasModel, OilGasModelForPretraining
from data.synthetic_data import SyntheticDataGenerator
from data.dataset import PretrainDataset, FinetuneDataset, SeismicWellDataset
from data.transforms import SeismicAugmentation, WellLogAugmentation
from training.losses import DiceLoss, FocalLoss, SSIMLoss, InfoNCELoss
from pretraining.msm_task import MaskedSeismicModeling
from pretraining.mwm_task import MaskedWellLogModeling
from pretraining.cmcl_task import CrossModalContrastiveLearning
from pretraining.swm_task import SeismicWellMatching


# =====================================================================
# Configuration Tests
# =====================================================================

class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        config = get_default_config()
        assert config.hidden_dim == 384
        assert config.seismic_encoder.embed_dim == 192
        assert config.well_log_encoder.num_curves == 7
        print("✓ Default config OK")

    def test_yaml_config(self):
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if config_path.exists():
            config = ModelConfig.from_yaml(str(config_path))
            assert config.hidden_dim > 0
            print("✓ YAML config loading OK")


# =====================================================================
# Data Tests
# =====================================================================

class TestDataGeneration:
    """Test synthetic data generation."""

    def test_synthetic_generator(self):
        gen = SyntheticDataGenerator(seed=42)
        sample = gen.generate_well_seismic_pair()

        assert sample["seismic"].shape == (1, 128, 256, 256)
        assert sample["well_log"].shape == (512, 7)
        assert sample["lithology"].shape == (512,)
        assert sample["fault_mask"].shape == (128, 256, 256)
        assert sample["reservoir_mask"].shape == (128, 256, 256)
        print("✓ Synthetic data generation OK")

    def test_pretrain_dataset(self):
        dataset = PretrainDataset(num_samples=10, use_synthetic=True)
        assert len(dataset) == 10
        sample = dataset[0]
        assert "seismic" in sample
        assert "well_log" in sample
        assert sample["seismic"].shape[0] == 1  # (1, D, H, W)
        assert sample["well_log"].shape[1] == 7  # (L, C)
        print("✓ Pretrain dataset OK")

    def test_finetune_dataset(self):
        for task in ["fault_detection", "reservoir_prediction", "lithology_classification"]:
            dataset = FinetuneDataset(num_samples=5, use_synthetic=True, task=task)
            sample = dataset[0]
            if task == "fault_detection":
                assert "fault_mask" in sample
            elif task == "reservoir_prediction":
                assert "reservoir_mask" in sample
            elif task == "lithology_classification":
                assert "lithology" in sample
        print("✓ Finetune datasets OK")

    def test_augmentations(self):
        gen = SyntheticDataGenerator(seed=42)
        sample = gen.generate_well_seismic_pair()

        seis_aug = SeismicAugmentation()
        aug_result = seis_aug(sample["seismic"])
        assert "seismic" in aug_result

        well_aug = WellLogAugmentation()
        aug_log, _ = well_aug(sample["well_log"])
        assert aug_log.shape == sample["well_log"].shape
        print("✓ Data augmentations OK")


# =====================================================================
# Encoder Tests
# =====================================================================

class TestSeismicEncoder:
    """Test 3D Seismic Encoder."""

    def test_forward(self):
        encoder = SeismicEncoder3D(
            in_channels=1,
            stem_channels=64,
            embed_dim=96,
            depths=[2, 2, 2],
            num_heads=[3, 6, 12],
            window_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            use_checkpoint=False,
        )

        x = torch.randn(2, 1, 64, 64, 64)
        global_feat = encoder(x)
        assert global_feat.shape[0] == 2
        print(f"Seismic encoder output: {global_feat.shape}")
        print("✓ Seismic encoder forward OK")

    def test_with_features(self):
        encoder = SeismicEncoder3D(
            embed_dim=96,
            depths=[2, 2, 2],
            num_heads=[3, 6, 12],
            window_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            use_checkpoint=False,
        )

        x = torch.randn(2, 1, 64, 64, 64)
        global_feat, stage_features = encoder(x, return_features=True)
        assert len(stage_features) == 3
        print(f"Stage features: {[f.shape for f in stage_features]}")
        print("✓ Seismic encoder features OK")

    def test_gradient_flow(self):
        encoder = SeismicEncoder3D(
            embed_dim=96,
            depths=[2, 2, 2],
            num_heads=[3, 6, 12],
            window_size=(4, 4, 4),
            patch_size=(2, 2, 2),
            use_checkpoint=False,
        )

        x = torch.randn(2, 1, 64, 64, 64)
        output = encoder(x)
        loss = output.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
        print("✓ Seismic encoder gradients OK")


class TestWellLogEncoder:
    """Test Well Log Encoder."""

    def test_forward(self):
        encoder = WellLogEncoder1D(
            num_curves=7,
            stem_channels=[32, 64, 128],
            embed_dim=96,
            num_layers=2,
            num_heads=4,
            max_seq_len=512,
            use_physics_constraint=False,
        )

        x = torch.randn(2, 7, 512)
        global_feat, seq_feat = encoder(x, return_sequence=True)
        assert global_feat.shape[0] == 2
        print(f"Well log encoder output: global={global_feat.shape}, seq={seq_feat.shape if seq_feat is not None else None}")
        print("✓ Well log encoder forward OK")

    def test_with_physics(self):
        encoder = WellLogEncoder1D(
            num_curves=7,
            embed_dim=96,
            num_layers=2,
            num_heads=4,
            max_seq_len=256,
            use_physics_constraint=True,
        )

        x = torch.randn(2, 7, 256)
        output, _ = encoder(x)
        assert output.shape[0] == 2
        print("✓ Well log encoder with physics OK")


# =====================================================================
# Fusion Module Tests
# =====================================================================

class TestFusion:
    """Test Cross-Modal Fusion."""

    def test_fusion_forward(self):
        fusion = CrossModalFusion(
            hidden_dim=128,
            num_cross_attention_heads=4,
            num_fusion_layers=1,
            use_gating=True,
        )

        seismic_feat = torch.randn(4, 128)
        well_feat = torch.randn(4, 128)

        fused = fusion(seismic_feat, well_feat)
        assert fused.shape == (4, 128)
        print(f"Fusion output: {fused.shape}")
        print("✓ Fusion forward OK")

    def test_fusion_with_details(self):
        fusion = CrossModalFusion(
            hidden_dim=128,
            num_cross_attention_heads=4,
            use_gating=True,
        )

        s = torch.randn(4, 128)
        w = torch.randn(4, 128)
        fused, details = fusion(s, w, return_details=True)

        assert "coarse_feat" in details
        assert "cross_feat" in details
        assert "gate_weights" in details
        print("✓ Fusion details OK")

    def test_modality_projection(self):
        proj = ModalityProjection(
            seismic_dim=768,
            well_log_dim=96,
            common_dim=384,
        )

        s = torch.randn(4, 768)
        w = torch.randn(4, 96)

        s_proj, w_proj = proj(s, w)
        assert s_proj.shape == (4, 384)
        assert w_proj.shape == (4, 384)
        print("✓ Modality projection OK")


# =====================================================================
# Prediction Heads Tests
# =====================================================================

class TestPredictionHeads:
    """Test task-specific prediction heads."""

    def test_fault_head(self):
        head = FaultDetectionHead(
            hidden_dim=384,
            encoder_channels=[96, 192, 384],
            decoder_channels=[256, 128, 64],
        )

        fused = torch.randn(2, 384)
        enc_feats = [
            torch.randn(2, 96, 8, 16, 16),
            torch.randn(2, 192, 4, 8, 8),
            torch.randn(2, 384, 2, 4, 4),
        ]

        output = head(fused, enc_feats)
        assert output.shape[0] == 2
        assert output.shape[1] == 1  # 1 channel output
        print(f"Fault head output: {output.shape}")
        print("✓ Fault detection head OK")

    def test_reservoir_head(self):
        head = ReservoirPredictionHead(
            hidden_dim=384,
            encoder_channels=[96, 192, 384],
        )

        fused = torch.randn(2, 384)
        well_feat = torch.randn(2, 384)
        enc_feats = [
            torch.randn(2, 96, 8, 16, 16),
            torch.randn(2, 192, 4, 8, 8),
            torch.randn(2, 384, 2, 4, 4),
        ]

        outputs = head(fused, well_feat, enc_feats)
        assert "reservoir_prob" in outputs
        assert "reservoir_props" in outputs
        print(f"Reservoir prob: {outputs['reservoir_prob'].shape}")
        print(f"Reservoir props: {outputs['reservoir_props'].shape}")
        print("✓ Reservoir prediction head OK")

    def test_lithology_head(self):
        head = LithologyClassificationHead(
            hidden_dim=384,
            num_classes=4,
        )

        well_seq = torch.randn(2, 128, 384)
        seis_trace = torch.randn(2, 128, 384)

        outputs = head(well_seq, seis_trace)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 128, 4)
        print(f"Lithology logits: {outputs['logits'].shape}")
        print("✓ Lithology classification head OK")


# =====================================================================
# Full Model Tests
# =====================================================================

class TestFullModel:
    """Test the complete OilGasModel."""

    def test_model_creation(self):
        config = get_default_config()
        config.seismic_encoder.depths = [2, 2, 2]
        config.seismic_encoder.num_heads = [3, 6, 12]
        config.seismic_encoder.use_checkpoint = False
        config.well_log_encoder.num_layers = 2
        config.well_log_encoder.use_physics_constraint = False

        model = OilGasModel(config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {n_params:,}")
        assert n_params > 0
        print("✓ Full model creation OK")

    def test_model_encode(self):
        config = get_default_config()
        config.seismic_encoder.depths = [2, 2, 2]
        config.seismic_encoder.num_heads = [3, 6, 12]
        config.seismic_encoder.use_checkpoint = False
        config.well_log_encoder.num_layers = 2
        config.well_log_encoder.use_physics_constraint = False

        model = OilGasModel(config)

        seismic = torch.randn(2, 1, 64, 64, 64)
        well_log = torch.randn(2, 7, 512)

        encoded = model.encode(seismic, well_log)
        assert "fused" in encoded
        assert "seismic_feat" in encoded
        assert "well_feat" in encoded
        print(f"Fused: {encoded['fused'].shape}")
        print(f"Seismic feat: {encoded['seismic_feat'].shape}")
        print(f"Well feat: {encoded['well_feat'].shape}")
        print("✓ Model encode OK")

    def test_model_fault_detection(self):
        config = get_default_config()
        config.seismic_encoder.depths = [2, 2, 2]
        config.seismic_encoder.num_heads = [3, 6, 12]
        config.seismic_encoder.use_checkpoint = False
        config.well_log_encoder.num_layers = 2
        config.well_log_encoder.use_physics_constraint = False

        model = OilGasModel(config)
        seismic = torch.randn(2, 1, 64, 64, 64)

        outputs = model(seismic, task="fault_detection")
        assert "fault_prob" in outputs
        print(f"Fault prob: {outputs['fault_prob'].shape}")
        print("✓ Model fault detection OK")

    def test_pretrain_model(self):
        config = get_default_config()
        config.seismic_encoder.depths = [2, 2, 2]
        config.seismic_encoder.num_heads = [3, 6, 12]
        config.seismic_encoder.use_checkpoint = False
        config.well_log_encoder.num_layers = 2
        config.well_log_encoder.use_physics_constraint = False

        model = OilGasModelForPretraining(config)

        seismic = torch.randn(2, 1, 64, 64, 64)
        well_log = torch.randn(2, 7, 512)

        # Test CMCL
        cmcl_out = model(seismic, well_log, task="cmcl")
        assert "seismic_embed" in cmcl_out
        assert "well_embed" in cmcl_out

        # Test SWM
        swm_out = model(seismic, well_log, task="swm")
        assert "match_prob" in swm_out

        print(f"CMCL seismic embed: {cmcl_out['seismic_embed'].shape}")
        print(f"SWM match prob: {swm_out['match_prob'].shape}")
        print("✓ Pretrain model OK")


# =====================================================================
# Pretraining Task Tests
# =====================================================================

class TestPretrainingTasks:
    """Test individual pretraining tasks."""

    def test_cmcl(self):
        cmcl = CrossModalContrastiveLearning(
            seismic_dim=384,
            well_dim=384,
            projection_dim=128,
        )

        s_feat = torch.randn(8, 384)
        w_feat = torch.randn(8, 384)

        outputs = cmcl(s_feat, w_feat)
        assert "loss" in outputs
        assert "contrastive_accuracy" in outputs
        print(f"CMCL loss: {outputs['loss']:.4f}, acc: {outputs['contrastive_accuracy']:.4f}")
        print("✓ CMCL task OK")

    def test_swm(self):
        swm = SeismicWellMatching(hidden_dim=384)

        s_feat = torch.randn(8, 384)
        w_feat = torch.randn(8, 384)

        outputs = swm(s_feat, w_feat)
        assert "loss" in outputs
        assert "matching_accuracy" in outputs
        print(f"SWM loss: {outputs['loss']:.4f}, acc: {outputs['matching_accuracy']:.4f}")
        print("✓ SWM task OK")


# =====================================================================
# Loss Function Tests
# =====================================================================

class TestLosses:
    """Test loss functions."""

    def test_dice_loss(self):
        dice_loss = DiceLoss()
        pred = torch.sigmoid(torch.randn(4, 1, 32, 32, 32))
        target = (torch.rand(4, 1, 32, 32, 32) > 0.9).float()
        loss = dice_loss(pred, target)
        assert loss.item() >= 0
        print(f"Dice loss: {loss:.4f}")
        print("✓ Dice loss OK")

    def test_focal_loss(self):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.randn(4, 1, 32, 32, 32)
        target = (torch.rand(4, 1, 32, 32, 32) > 0.9).float()
        loss = focal_loss(pred, target)
        assert loss.item() >= 0
        print(f"Focal loss: {loss:.4f}")
        print("✓ Focal loss OK")

    def test_ssim_loss(self):
        ssim_loss = SSIMLoss(window_size=5)
        pred = torch.rand(2, 1, 16, 32, 32)
        target = pred + 0.01 * torch.randn_like(pred)
        loss = ssim_loss(pred, target)
        assert loss.item() >= 0
        print(f"SSIM loss: {loss:.4f}")
        print("✓ SSIM loss OK")

    def test_infonce_loss(self):
        infonce = InfoNCELoss(temperature=0.07)
        z1 = torch.randn(16, 128)
        z2 = torch.randn(16, 128)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        z2 = torch.nn.functional.normalize(z2, dim=-1)
        loss = infonce(z1, z2)
        assert loss.item() >= 0
        print(f"InfoNCE loss: {loss:.4f}")
        print("✓ InfoNCE loss OK")


# =====================================================================
# End-to-End Pipeline Test
# =====================================================================

class TestEndToEnd:
    """End-to-end pipeline test."""

    def test_full_pipeline(self):
        """Test complete pipeline: data -> model -> loss -> backward."""
        print("\n=== End-to-End Pipeline Test ===")

        # 1. Generate synthetic data
        gen = SyntheticDataGenerator(seed=42)
        sample = gen.generate_well_seismic_pair(
            seismic_shape=(64, 128, 128),
            log_length=256,
        )

        seismic = torch.from_numpy(sample["seismic"]).unsqueeze(0).float()  # (1, 1, 64, 128, 128)
        well_log = torch.from_numpy(sample["well_log"]).unsqueeze(0).float()  # (1, 256, 7)

        # 2. Create model
        config = get_default_config()
        config.seismic_encoder.depths = [2, 2, 2]
        config.seismic_encoder.num_heads = [3, 6, 12]
        config.seismic_encoder.use_checkpoint = False
        config.well_log_encoder.num_layers = 2
        config.well_log_encoder.use_physics_constraint = False

        model = OilGasModel(config)

        # 3. Forward pass
        outputs = model(seismic, well_log, task="fault_detection")
        assert "fault_prob" in outputs

        # 4. Compute loss
        dice_loss = DiceLoss()
        fault_target = torch.from_numpy(sample["fault_mask"]).unsqueeze(0).unsqueeze(0).float()
        loss = dice_loss(outputs["fault_prob"], fault_target)

        # 5. Backward pass
        loss.backward()

        # 6. Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients in model!"

        print(f"Pipeline loss: {loss.item():.4f}")
        print("✓ End-to-end pipeline OK")

    def test_pretraining_pipeline(self):
        """Test pretraining pipeline."""
        print("\n=== Pretraining Pipeline Test ===")

        config = get_default_config()
        config.seismic_encoder.depths = [2, 2, 2]
        config.seismic_encoder.num_heads = [3, 6, 12]
        config.seismic_encoder.use_checkpoint = False
        config.well_log_encoder.num_layers = 2

        model = OilGasModelForPretraining(config)

        seismic = torch.randn(4, 1, 64, 64, 64)
        well_log = torch.randn(4, 7, 512)

        # CMCL forward
        cmcl_out = model(seismic, well_log, task="cmcl")
        # Compute contrastive loss
        infonce = InfoNCELoss()
        loss = infonce(cmcl_out["seismic_embed"], cmcl_out["well_embed"])
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad
        print(f"Pretraining pipeline loss: {loss.item():.4f}")
        print("✓ Pretraining pipeline OK")


# =====================================================================
# Runner
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Oil & Gas Multi-modal Model - Test Suite")
    print("=" * 60)

    # Run tests
    TestConfig().test_default_config()
    TestConfig().test_yaml_config()

    TestDataGeneration().test_synthetic_generator()
    TestDataGeneration().test_pretrain_dataset()
    TestDataGeneration().test_finetune_dataset()
    TestDataGeneration().test_augmentations()

    TestSeismicEncoder().test_forward()
    TestSeismicEncoder().test_with_features()
    TestSeismicEncoder().test_gradient_flow()

    TestWellLogEncoder().test_forward()
    TestWellLogEncoder().test_with_physics()

    TestFusion().test_fusion_forward()
    TestFusion().test_fusion_with_details()
    TestFusion().test_modality_projection()

    TestPredictionHeads().test_fault_head()
    TestPredictionHeads().test_reservoir_head()
    TestPredictionHeads().test_lithology_head()

    TestFullModel().test_model_creation()
    TestFullModel().test_model_encode()
    TestFullModel().test_model_fault_detection()
    TestFullModel().test_pretrain_model()

    TestPretrainingTasks().test_cmcl()
    TestPretrainingTasks().test_swm()

    TestLosses().test_dice_loss()
    TestLosses().test_focal_loss()
    TestLosses().test_ssim_loss()
    TestLosses().test_infonce_loss()

    TestEndToEnd().test_full_pipeline()
    TestEndToEnd().test_pretraining_pipeline()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
