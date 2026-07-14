"""
Single-field multimodal dataset (Volve / RMOTC / Penobscot).

Uses verified geometry only — no estimated well positions or assumed-vertical RMOTC wells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .field_geometry import (
    build_field_survey_geometry,
    build_field_trajectories,
    is_field_geometry_verified,
    resolve_field_paths,
)
from .volve_dataset import LASWellLoader, SEGYLoader, parse_well_header_from_las


class FieldDataset(Dataset):
    """Multimodal seismic + well-log dataset for one prepared field."""

    def __init__(
        self,
        project_root: str,
        field: str,
        mode: str = "pretrain",
        seismic_patch_size: Tuple[int, int, int] = (32, 32, 32),
        well_seq_len: int = 128,
        well_curves: Optional[List[str]] = None,
        train_wells: Optional[List[str]] = None,
        val_wells: Optional[List[str]] = None,
        norm_stats: Optional[Dict] = None,
        require_verified_geometry: bool = True,
    ):
        super().__init__()
        self.project_root = Path(project_root)
        self.field = field
        self.mode = mode
        self.seismic_patch_size = seismic_patch_size
        self.well_seq_len = well_seq_len
        self.require_verified_geometry = require_verified_geometry

        paths = resolve_field_paths(self.project_root, field)
        self.prepared_dir = paths["prepared_dir"]
        self.segy_path = paths["segy_path"]
        self.deviations_dir = paths["deviations_dir"]

        with open(self.prepared_dir / "well_metadata.json", encoding="utf-8") as f:
            self.well_metadata = json.load(f)
        dev_inv_path = self.prepared_dir / "deviation_inventory.json"
        self.deviation_inventory = (
            json.load(open(dev_inv_path, encoding="utf-8")) if dev_inv_path.exists() else {}
        )

        try:
            from .prepare_volve_data import STANDARD_CURVES

            default_curves = list(STANDARD_CURVES)
        except ImportError:
            default_curves = ["GR", "SP", "CAL", "RD", "MLL", "MSFL", "NPHI", "RHOB", "DT"]
        self.well_curves = well_curves or default_curves

        print(f"[{field}] Loading seismic from {self.segy_path} ...")
        self.seismic = SEGYLoader(str(self.segy_path))
        print(f"[{field}]   Shape: {self.seismic.shape}")

        self.las_loader = LASWellLoader()
        self.well_data = self._load_all_wells()
        print(f"[{field}] Loaded {len(self.well_data)} wells with LAS data")

        self.trajectories = build_field_trajectories(
            field, self.well_metadata, self.deviations_dir
        )
        dev_loaded = sum(
            1
            for w in self.well_data
            if w in self.trajectories and self.trajectories[w].has_deviation
        )
        print(f"[{field}] Trajectories: {len(self.trajectories)} wells, {dev_loaded} with deviation")

        self.survey_geometry = build_field_survey_geometry(
            field, self.project_root, self.seismic
        )
        self.depth_axis = getattr(self.survey_geometry, "depth_axis", "depth")
        self.sample_interval_s = getattr(self.survey_geometry, "sample_interval_s", 0.004)

        if self.require_verified_geometry:
            self.well_data = self._filter_geometry_verified_wells(self.well_data)

        well_names = sorted(self.well_data.keys())
        n_train = max(1, int(len(well_names) * 0.8)) if well_names else 0

        if train_wells is not None:
            self.train_wells = [w for w in train_wells if w in self.well_data]
            self.val_wells = [w for w in (val_wells or []) if w in self.well_data]
        else:
            self.train_wells = well_names[:n_train]
            self.val_wells = well_names[n_train:]

        print(f"[{field}] Train wells ({len(self.train_wells)}): {self.train_wells}")
        print(f"[{field}] Val wells ({len(self.val_wells)}): {self.val_wells}")

        self.samples = self._build_sample_index()
        print(f"[{field}] Total samples ({mode}): {len(self.samples)}")

        self.norm_stats = norm_stats if norm_stats is not None else self._compute_norm_stats()

    def _load_all_wells(self) -> Dict:
        well_data = {}
        for well_name, meta in sorted(self.well_metadata.items()):
            las_path = meta.get("las_path")
            if not las_path or not Path(las_path).exists():
                continue
            try:
                data = self.las_loader.read(str(las_path))
            except Exception as exc:
                print(f"  [{self.field}] Warning: failed to read {las_path}: {exc}")
                continue
            if "depth" not in data or len(data["depth"]) < self.well_seq_len * 2:
                continue
            data["well_name"] = well_name
            data["las_path"] = las_path
            header_meta = parse_well_header_from_las(str(las_path))
            for key in ("latitude", "longitude", "kb_elevation_m"):
                if header_meta.get(key) is not None:
                    data[key] = header_meta[key]
            for key in ("latitude", "longitude", "kb_elevation_m"):
                if key not in data and meta.get(key) is not None:
                    data[key] = meta[key]
            well_data[well_name] = data
        return well_data

    def _filter_geometry_verified_wells(self, well_data: Dict) -> Dict:
        verified: Dict = {}
        excluded: List[str] = []
        for well_name, data in well_data.items():
            has_dev = (
                well_name in self.trajectories
                and self.trajectories[well_name].has_deviation
            )
            if is_field_geometry_verified(
                self.field,
                well_name,
                self.well_metadata,
                self.deviation_inventory,
                has_dev,
            ):
                verified[well_name] = data
            else:
                excluded.append(well_name)

        if excluded:
            print(
                f"[{self.field}] Excluded {len(excluded)} wells (non-verified geometry)"
            )
        print(f"[{self.field}] Geometry-verified wells: {len(verified)}")
        return verified

    def _build_sample_index(self) -> List[Dict]:
        samples = []
        wells = self.val_wells if self.mode in ("test", "val") else self.train_wells
        stride = max(1, self.well_seq_len // 2)
        for well_name in wells:
            if well_name not in self.well_data:
                continue
            depth = self.well_data[well_name]["depth"]
            n = len(depth)
            for start in range(0, n - self.well_seq_len, stride):
                samples.append(
                    {
                        "well_name": well_name,
                        "depth_start": start,
                        "depth_end": start + self.well_seq_len,
                    }
                )
        return samples

    def _compute_norm_stats(self) -> Dict:
        stats: Dict = {}
        seismic_values = []
        for sample in self.samples[: min(500, len(self.samples))]:
            patch = self._extract_seismic_patch(
                sample["well_name"], sample["depth_start"], sample["depth_end"]
            )
            if patch is not None:
                seismic_values.append(patch.flatten())

        stats["n_samples"] = float(len(self.samples))
        if seismic_values:
            all_seis = np.concatenate(seismic_values)
            stats["seismic_mean"] = float(np.nanmean(all_seis))
            stats["seismic_std"] = float(np.nanstd(all_seis)) + 1e-8
        else:
            stats["seismic_mean"] = 0.0
            stats["seismic_std"] = 1.0

        for curve_name in self.well_curves:
            values = []
            for well_name in self.train_wells:
                if well_name in self.well_data and curve_name in self.well_data[well_name]:
                    v = self.well_data[well_name][curve_name]
                    values.append(v[~np.isnan(v)])
            if values:
                all_vals = np.concatenate(values)
                stats[f"{curve_name}_mean"] = float(np.nanmean(all_vals))
                stats[f"{curve_name}_std"] = float(np.nanstd(all_vals)) + 1e-8
            else:
                stats[f"{curve_name}_mean"] = 0.0
                stats[f"{curve_name}_std"] = 1.0
        return stats

    def _extract_seismic_patch(
        self, well_name: str, depth_start: int, depth_end: int
    ) -> Optional[np.ndarray]:
        p_d, p_h, p_w = self.seismic_patch_size
        well_depth = self.well_data.get(well_name, {}).get("depth")
        if well_depth is None or depth_start >= len(well_depth):
            return None
        md_center = float(well_depth[(depth_start + depth_end) // 2])

        traj = self.trajectories.get(well_name)
        if traj is None:
            il_c = (self.seismic.inline_min + self.seismic.inline_max) // 2
            xl_c = (self.seismic.xline_min + self.seismic.xline_max) // 2
        else:
            easting, northing, tvdss = traj.get_position_at_md(md_center)
            il_c, xl_c = self.survey_geometry.utm_to_ilxl(easting, northing)

        il_start = max(self.seismic.inline_min, int(il_c - p_h // 2))
        il_end = min(self.seismic.inline_max + 1, il_start + p_h)
        xl_start = max(self.seismic.xline_min, int(xl_c - p_w // 2))
        xl_end = min(self.seismic.xline_max + 1, xl_start + p_w)

        try:
            volume = self.seismic.read_volume(
                il_range=(il_start, il_end), xl_range=(xl_start, xl_end)
            )
        except Exception:
            return None

        if volume is None or volume.size == 0:
            return None

        actual_h, actual_w = il_end - il_start, xl_end - xl_start
        if actual_h < p_h or actual_w < p_w:
            volume = np.pad(
                volume,
                ((0, max(0, p_h - actual_h)), (0, max(0, p_w - actual_w)), (0, 0)),
                mode="constant",
            )

        if self.depth_axis == "time":
            center_sample = volume.shape[2] // 2
        elif traj is not None:
            _, _, tvdss = traj.get_position_at_md(md_center)
            center_sample = int(tvdss / max(self.sample_interval_s * 1000, 1.0))
        else:
            center_sample = volume.shape[2] // 2

        d_start = max(0, min(volume.shape[2] - p_d, center_sample - p_d // 2))
        volume = volume[:p_h, :p_w, d_start : d_start + p_d]
        if volume.shape[2] < p_d:
            volume = np.pad(volume, ((0, 0), (0, 0), (0, p_d - volume.shape[2])), mode="constant")
        return volume

    def _extract_well_sequence(self, well_name: str, start_idx: int) -> Optional[Dict]:
        if well_name not in self.well_data:
            return None
        data = self.well_data[well_name]
        end_idx = start_idx + self.well_seq_len
        if end_idx > len(data["depth"]):
            return None
        result = {"depth": data["depth"][start_idx:end_idx]}
        for curve_name in self.well_curves:
            if curve_name in data:
                result[curve_name] = data[curve_name][start_idx:end_idx]
            else:
                result[curve_name] = np.full(self.well_seq_len, np.nan)
        for label_name in ["VSH", "PHIF", "SW", "KLOGH", "SAND_FLAG", "COAL_FLAG"]:
            if label_name in data:
                result[label_name] = data[label_name][start_idx:end_idx]
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        well_name = sample["well_name"]
        depth_start = sample["depth_start"]
        depth_end = sample["depth_end"]

        seis_patch = self._extract_seismic_patch(well_name, depth_start, depth_end)
        # Reject missing or near-constant patches (zero-fill / heavy pad → trivial MSM).
        seismic_valid = (
            seis_patch is not None and float(np.nanstd(seis_patch)) > 1e-6
        )
        if not seismic_valid:
            # Keep tensor shape for collation; MSM/CMCL must skip via seismic_valid.
            seis_patch = np.zeros(
                (self.seismic_patch_size[1], self.seismic_patch_size[2], self.seismic_patch_size[0]),
                dtype=np.float32,
            )

        seis_patch = (seis_patch - self.norm_stats["seismic_mean"]) / self.norm_stats["seismic_std"]
        seis_patch = np.transpose(seis_patch, (2, 0, 1))
        seis_tensor = torch.from_numpy(seis_patch.copy()).float().unsqueeze(0)

        well_seq = self._extract_well_sequence(well_name, depth_start)
        well_curves_arr, curve_mask_arr, value_mask_arr = [], [], []
        well_mask = np.zeros(self.well_seq_len, dtype=np.float32)

        for curve_name in self.well_curves:
            if well_seq is not None and curve_name in well_seq:
                raw = well_seq[curve_name]
                valid = ~np.isnan(raw)
                value_mask_arr.append(valid.astype(np.float32))
                if np.any(valid):
                    curve_mask_arr.append(1.0)
                    mean = self.norm_stats.get(f"{curve_name}_mean", 0)
                    std = self.norm_stats.get(f"{curve_name}_std", 1)
                    vals = np.where(valid, (raw - mean) / std, 0.0)
                    well_mask = np.maximum(
                        well_mask, valid.astype(np.float32)
                    )
                else:
                    curve_mask_arr.append(0.0)
                    vals = np.zeros(self.well_seq_len, dtype=np.float32)
            else:
                curve_mask_arr.append(0.0)
                value_mask_arr.append(
                    np.zeros(self.well_seq_len, dtype=np.float32)
                )
                vals = np.zeros(self.well_seq_len, dtype=np.float32)
            well_curves_arr.append(vals)

        label_specs = {
            "sand_flag": "SAND_FLAG",
            "coal_flag": "COAL_FLAG",
            "porosity": "PHIF",
            "water_saturation": "SW",
            "vshale": "VSH",
            "permeability": "KLOGH",
        }
        labels = {}
        for label_key, las_key in label_specs.items():
            if well_seq is not None and las_key in well_seq:
                labels[label_key] = torch.from_numpy(
                    np.nan_to_num(well_seq[las_key], nan=0).copy()
                ).float()
            else:
                labels[label_key] = torch.zeros(self.well_seq_len, dtype=torch.float32)

        return {
            "seismic": seis_tensor,
            "seismic_valid": torch.tensor(seismic_valid, dtype=torch.bool),
            "well_log": torch.from_numpy(np.stack(well_curves_arr)).float(),
            "well_mask": torch.from_numpy(well_mask).float(),
            "curve_mask": torch.tensor(curve_mask_arr, dtype=torch.float32),
            "well_value_mask": torch.from_numpy(
                np.stack(value_mask_arr)
            ).float(),
            "labels": labels,
            "well_name": well_name,
            "field": self.field,
            "depth_start": depth_start,
        }
