"""
Combined multimodal dataset: Volve + RMOTC + Penobscot.

All fields use verified geometry only (no estimated well positions).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .field_dataset import FieldDataset

DEFAULT_FIELDS = ("volve", "rmotc", "penobscot")


def _pooled_mean_std(
    means: List[float], stds: List[float], weights: List[float]
) -> Tuple[float, float]:
    """Sample-weighted pooled mean / std across fields (not average of stds)."""
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(stds, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return 0.0, 1.0
    mean = float(np.average(m, weights=w))
    # E[X^2] = sum w_i (std_i^2 + mean_i^2) / sum w; Var = E[X^2] - mean^2
    second = float(np.average(s * s + m * m, weights=w))
    var = max(second - mean * mean, 0.0)
    return mean, float(np.sqrt(var) + 1e-8)


def merge_norm_stats(stats_list: List[Dict], well_curves: List[str]) -> Dict:
    """Merge per-field normalization stats (weighted by n_samples when present)."""
    merged: Dict = {"seismic_mean": 0.0, "seismic_std": 1.0}
    seis_means, seis_stds, weights = [], [], []

    for stats in stats_list:
        w = float(stats.get("n_samples", 1.0))
        weights.append(max(w, 1.0))
        seis_means.append(stats.get("seismic_mean", 0.0))
        seis_stds.append(stats.get("seismic_std", 1.0))

    if weights:
        merged["seismic_mean"], merged["seismic_std"] = _pooled_mean_std(
            seis_means, seis_stds, weights
        )

    for curve in well_curves:
        means, stds, cw = [], [], []
        for stats in stats_list:
            key_m, key_s = f"{curve}_mean", f"{curve}_std"
            if key_m in stats and stats.get(key_s, 1.0) > 1e-6:
                means.append(stats[key_m])
                stds.append(stats[key_s])
                cw.append(float(stats.get("n_samples", 1.0)))
        if means:
            merged[f"{curve}_mean"], merged[f"{curve}_std"] = _pooled_mean_std(
                means, stds, cw
            )
        else:
            merged[f"{curve}_mean"] = 0.0
            merged[f"{curve}_std"] = 1.0

    return merged


class CombinedMultimodalDataset(Dataset):
    """Concatenates multiple FieldDataset instances with shared normalization."""

    def __init__(
        self,
        project_root: str,
        fields: Optional[List[str]] = None,
        mode: str = "pretrain",
        seismic_patch_size: Tuple[int, int, int] = (32, 32, 32),
        well_seq_len: int = 128,
        well_curves: Optional[List[str]] = None,
        norm_stats: Optional[Dict] = None,
        field_datasets: Optional[Dict[str, FieldDataset]] = None,
        require_verified_geometry: bool = True,
    ):
        super().__init__()
        self.project_root = project_root
        self.fields = list(fields or DEFAULT_FIELDS)
        self.mode = mode
        self.well_curves = well_curves

        if field_datasets is not None:
            self.field_datasets = field_datasets
        else:
            self.field_datasets = {}
            per_field_stats = []
            for field in self.fields:
                ds = FieldDataset(
                    project_root=project_root,
                    field=field,
                    mode=mode,
                    seismic_patch_size=seismic_patch_size,
                    well_seq_len=well_seq_len,
                    well_curves=well_curves,
                    require_verified_geometry=require_verified_geometry,
                )
                self.field_datasets[field] = ds
                per_field_stats.append(ds.norm_stats)

            if norm_stats is None:
                curves = well_curves or next(iter(self.field_datasets.values())).well_curves
                norm_stats = merge_norm_stats(per_field_stats, curves)

        self.norm_stats = norm_stats
        # Well-curve stats are pooled across fields (physical units are
        # comparable), but seismic amplitude scaling is arbitrary per survey
        # (e.g. AGC vs raw). Pooling seismic into one global std lets the
        # highest-amplitude field dominate and crushes the others to near-zero
        # variance, starving MSM of signal. So keep each field's OWN seismic
        # mean/std and only overlay the shared curve stats.
        for ds in self.field_datasets.values():
            field_stats = dict(self.norm_stats)
            if ds.norm_stats is not None:
                field_stats["seismic_mean"] = ds.norm_stats.get(
                    "seismic_mean", self.norm_stats["seismic_mean"]
                )
                field_stats["seismic_std"] = ds.norm_stats.get(
                    "seismic_std", self.norm_stats["seismic_std"]
                )
            ds.norm_stats = field_stats

        self.samples: List[Tuple[str, int]] = []
        for field, ds in self.field_datasets.items():
            for i in range(len(ds)):
                self.samples.append((field, i))

        self.train_wells = {f: ds.train_wells for f, ds in self.field_datasets.items()}
        self.val_wells = {f: ds.val_wells for f, ds in self.field_datasets.items()}

        print(
            f"CombinedMultimodalDataset ({mode}): "
            f"{len(self.samples)} samples from {list(self.field_datasets.keys())}"
        )

    @classmethod
    def build_train_val(
        cls,
        project_root: str,
        fields: Optional[List[str]] = None,
        seismic_patch_size: Tuple[int, int, int] = (32, 32, 32),
        well_seq_len: int = 128,
        well_curves: Optional[List[str]] = None,
        require_verified_geometry: bool = True,
    ) -> Tuple["CombinedMultimodalDataset", "CombinedMultimodalDataset"]:
        fields = list(fields or DEFAULT_FIELDS)
        train_parts = {}
        per_field_stats = []

        for field in fields:
            ds = FieldDataset(
                project_root=project_root,
                field=field,
                mode="pretrain",
                seismic_patch_size=seismic_patch_size,
                well_seq_len=well_seq_len,
                well_curves=well_curves,
                require_verified_geometry=require_verified_geometry,
            )
            train_parts[field] = ds
            per_field_stats.append(ds.norm_stats)

        curves = well_curves or next(iter(train_parts.values())).well_curves
        norm_stats = merge_norm_stats(per_field_stats, curves)

        train_ds = cls(
            project_root=project_root,
            fields=fields,
            mode="pretrain",
            seismic_patch_size=seismic_patch_size,
            well_seq_len=well_seq_len,
            well_curves=well_curves,
            norm_stats=norm_stats,
            field_datasets=train_parts,
            require_verified_geometry=require_verified_geometry,
        )

        val_parts = {}
        for field in fields:
            ref = train_parts[field]
            # ref.norm_stats now carries pooled curve stats + this field's OWN
            # seismic mean/std (set during train_ds construction). Reuse it so
            # val seismic is normalized on the same per-field scale as train.
            val_parts[field] = FieldDataset(
                project_root=project_root,
                field=field,
                mode="test",
                seismic_patch_size=seismic_patch_size,
                well_seq_len=well_seq_len,
                well_curves=well_curves,
                train_wells=ref.train_wells,
                val_wells=ref.val_wells,
                norm_stats=ref.norm_stats,
                require_verified_geometry=require_verified_geometry,
            )

        val_ds = cls(
            project_root=project_root,
            fields=fields,
            mode="test",
            seismic_patch_size=seismic_patch_size,
            well_seq_len=well_seq_len,
            well_curves=well_curves,
            norm_stats=norm_stats,
            field_datasets=val_parts,
            require_verified_geometry=require_verified_geometry,
        )
        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        field, local_idx = self.samples[idx]
        return self.field_datasets[field][local_idx]

    def summary(self) -> str:
        lines = ["Multi-field dataset summary:"]
        for field, ds in self.field_datasets.items():
            lines.append(
                f"  {field}: {len(ds.train_wells)} train wells, "
                f"{len(ds.val_wells)} val wells, {len(ds)} {self.mode} samples"
            )
        lines.append(f"  Total: {len(self)} samples")
        return "\n".join(lines)
