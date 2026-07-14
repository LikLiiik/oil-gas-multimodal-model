"""
Seismic-only patch dataset for Stage-1 MSM.

Samples patches directly from SEG-Y volumes (no well matching required),
reusing already-loaded FieldDataset seismic handles and norm stats.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .field_dataset import FieldDataset


class FieldSeismicPatchDataset(Dataset):
    """Random / fixed seismic patches from one field volume."""

    def __init__(
        self,
        field_ds: FieldDataset,
        num_patches: int = 2000,
        seed: int = 0,
        deterministic: bool = False,
        min_std: float = 1e-3,
        max_retries: int = 8,
    ):
        super().__init__()
        self.field = field_ds.field
        self.seismic = field_ds.seismic
        self.norm_stats = field_ds.norm_stats
        self.patch_size = tuple(field_ds.seismic_patch_size)
        self.min_std = min_std
        self.max_retries = max_retries
        self.deterministic = deterministic
        self.rng = np.random.RandomState(seed)

        self.centers = self._build_centers(num_patches, seed)
        print(
            f"[{self.field}] SeismicPatchDataset: {len(self.centers)} patches "
            f"(patch={self.patch_size}, deterministic={deterministic})"
        )

    def _build_centers(self, num_patches: int, seed: int) -> List[Tuple[int, int, int]]:
        """Pick (inline, xline, depth_start) from existing traces."""
        p_d, p_h, p_w = self.patch_size
        n_z = int(self.seismic.num_samples)
        keys = list(self.seismic.trace_index.keys())
        if not keys:
            raise RuntimeError(f"[{self.field}] No seismic traces in index")

        # Subsample trace keys so RMOTC-scale indexes stay manageable.
        rng = np.random.RandomState(seed)
        if len(keys) > max(num_patches * 20, 20000):
            idx = rng.choice(len(keys), size=max(num_patches * 20, 20000), replace=False)
            keys = [keys[i] for i in idx]

        centers: List[Tuple[int, int, int]] = []
        # Prefer traces that leave room for a full spatial patch.
        usable = []
        for il, xl in keys:
            if (
                il - p_h // 2 >= self.seismic.inline_min
                and il + (p_h - p_h // 2) - 1 <= self.seismic.inline_max
                and xl - p_w // 2 >= self.seismic.xline_min
                and xl + (p_w - p_w // 2) - 1 <= self.seismic.xline_max
            ):
                usable.append((il, xl))
        if not usable:
            usable = keys

        for i in range(num_patches):
            il, xl = usable[i % len(usable)] if self.deterministic else usable[
                int(rng.randint(0, len(usable)))
            ]
            if n_z <= p_d:
                d0 = 0
            elif self.deterministic:
                d0 = (i * 17) % max(1, n_z - p_d)
            else:
                d0 = int(rng.randint(0, n_z - p_d + 1))
            centers.append((int(il), int(xl), int(d0)))
        return centers

    def __len__(self) -> int:
        return len(self.centers)

    def _read_patch(self, il_c: int, xl_c: int, d0: int) -> Optional[np.ndarray]:
        p_d, p_h, p_w = self.patch_size
        il_start = int(il_c - p_h // 2)
        xl_start = int(xl_c - p_w // 2)
        il_end = il_start + p_h
        xl_end = xl_start + p_w
        try:
            volume = self.seismic.read_volume(
                il_range=(il_start, il_end), xl_range=(xl_start, xl_end)
            )
        except Exception:
            return None
        if volume is None or volume.size == 0:
            return None

        # Pad if survey edge clipped the window.
        if volume.shape[0] < p_h or volume.shape[1] < p_w:
            volume = np.pad(
                volume,
                (
                    (0, max(0, p_h - volume.shape[0])),
                    (0, max(0, p_w - volume.shape[1])),
                    (0, 0),
                ),
                mode="constant",
            )
        volume = volume[:p_h, :p_w, :]
        d0 = max(0, min(volume.shape[2] - p_d, d0))
        patch = volume[:, :, d0 : d0 + p_d]
        if patch.shape[2] < p_d:
            patch = np.pad(
                patch, ((0, 0), (0, 0), (0, p_d - patch.shape[2])), mode="constant"
            )
        if float(np.nanstd(patch)) < self.min_std:
            return None
        return patch.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        p_d, p_h, p_w = self.patch_size
        base_il, base_xl, base_d = self.centers[idx]
        patch = None
        for attempt in range(self.max_retries):
            if attempt == 0:
                il_c, xl_c, d0 = base_il, base_xl, base_d
            else:
                # Jitter around the indexed center on failure.
                il_c = base_il + int(self.rng.randint(-16, 17))
                xl_c = base_xl + int(self.rng.randint(-16, 17))
                d0 = max(0, base_d + int(self.rng.randint(-8, 9)))
                il_c = min(max(il_c, self.seismic.inline_min), self.seismic.inline_max)
                xl_c = min(max(xl_c, self.seismic.xline_min), self.seismic.xline_max)
            patch = self._read_patch(il_c, xl_c, d0)
            if patch is not None:
                break

        if patch is None:
            # Should be rare; return a unit-noise placeholder marked invalid.
            patch = self.rng.randn(p_h, p_w, p_d).astype(np.float32) * 1e-6
            valid = False
        else:
            valid = True

        patch = (patch - self.norm_stats["seismic_mean"]) / self.norm_stats["seismic_std"]
        # (H, W, D) -> (D, H, W)
        patch = np.transpose(patch, (2, 0, 1))
        seis = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        return {
            "seismic": seis,
            "seismic_valid": torch.tensor(valid, dtype=torch.bool),
            "field": self.field,
        }


class CombinedSeismicPatchDataset(Dataset):
    """Concatenates per-field seismic patch datasets."""

    def __init__(self, parts: Sequence[FieldSeismicPatchDataset]):
        super().__init__()
        self.parts = list(parts)
        self.index: List[Tuple[int, int]] = []
        for pi, part in enumerate(self.parts):
            for i in range(len(part)):
                self.index.append((pi, i))
        fields = [p.field for p in self.parts]
        print(
            f"CombinedSeismicPatchDataset: {len(self.index)} patches from {fields}"
        )

    @classmethod
    def from_field_datasets(
        cls,
        field_datasets: Dict[str, FieldDataset],
        patches_per_field: int = 2000,
        seed: int = 0,
        deterministic: bool = False,
    ) -> "CombinedSeismicPatchDataset":
        parts = []
        for i, (field, ds) in enumerate(field_datasets.items()):
            parts.append(
                FieldSeismicPatchDataset(
                    ds,
                    num_patches=patches_per_field,
                    seed=seed + i * 17,
                    deterministic=deterministic,
                )
            )
        return cls(parts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        pi, local = self.index[idx]
        return self.parts[pi][local]
