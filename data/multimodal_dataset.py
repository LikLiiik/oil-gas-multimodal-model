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


def merge_norm_stats(stats_list: List[Dict], well_curves: List[str]) -> Dict:
    """Merge per-field normalization stats (weighted by sample count proxy)."""
    merged: Dict = {"seismic_mean": 0.0, "seismic_std": 1.0}
    seis_means, seis_stds, weights = [], [], []

    for stats in stats_list:
        w = 1.0
        weights.append(w)
        seis_means.append(stats.get("seismic_mean", 0.0))
        seis_stds.append(stats.get("seismic_std", 1.0))

    if weights:
        w_arr = np.array(weights, dtype=np.float64)
        merged["seismic_mean"] = float(np.average(seis_means, weights=w_arr))
        merged["seismic_std"] = float(np.average(seis_stds, weights=w_arr)) + 1e-8

    for curve in well_curves:
        means, stds, cw = [], [], []
        for stats in stats_list:
            key_m, key_s = f"{curve}_mean", f"{curve}_std"
            if key_m in stats and stats.get(key_s, 1.0) > 1e-6:
                means.append(stats[key_m])
                stds.append(stats[key_s])
                cw.append(1.0)
        if means:
            merged[f"{curve}_mean"] = float(np.mean(means))
            merged[f"{curve}_std"] = float(np.mean(stds)) + 1e-8
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
        for ds in self.field_datasets.values():
            ds.norm_stats = self.norm_stats

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
            val_parts[field] = FieldDataset(
                project_root=project_root,
                field=field,
                mode="test",
                seismic_patch_size=seismic_patch_size,
                well_seq_len=well_seq_len,
                well_curves=well_curves,
                train_wells=ref.train_wells,
                val_wells=ref.val_wells,
                norm_stats=norm_stats,
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
