"""
Per-field survey geometry and well trajectory builders.

Supports Volve, RMOTC (Teapot Dome), and Penobscot with verified geometry only.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .cbvs_io import CBVSSurveySpec, ilxl_to_xy, parse_penobscot_survey
from .well_seismic_tie import (
    DEFAULT_VOLVE_GEOMETRY,
    SeismicSurveyGeometry,
    WellTrajectory,
    VOLVE_WELL_COORDS,
    build_well_trajectories,
)


class AffinePlaneGeometry:
    """Map field XY (easting/northing) directly to inline/crossline."""

    def __init__(
        self,
        il_min: int,
        il_max: int,
        xl_min: int,
        xl_max: int,
        il_from_xy: np.ndarray,
        xl_from_xy: np.ndarray,
        depth_axis: str = "time",
        sample_interval_s: float = 0.002,
    ):
        self.il_min = il_min
        self.il_max = il_max
        self.xl_min = xl_min
        self.xl_max = xl_max
        self.il_min_bound = il_min
        self.il_max_bound = il_max
        self.xl_min_bound = xl_min
        self.xl_max_bound = xl_max
        self._il_coef = il_from_xy
        self._xl_coef = xl_from_xy
        self.depth_axis = depth_axis
        self.sample_interval_s = sample_interval_s

    def xy_to_ilxl(self, easting: float, northing: float) -> Tuple[float, float]:
        v = np.array([easting, northing, 1.0], dtype=np.float64)
        il = float(v @ self._il_coef)
        xl = float(v @ self._xl_coef)
        return il, xl

    def utm_to_ilxl(self, easting: float, northing: float) -> Tuple[float, float]:
        return self.xy_to_ilxl(easting, northing)

    def latlon_to_ilxl(self, lat: float, lon: float) -> Tuple[float, float]:
        from .well_seismic_tie import latlon_to_utm

        _, e, n = latlon_to_utm(lat, lon)
        return self.xy_to_ilxl(e, n)


class PenobscotSurveyGeometry:
    """IL/XL mapping from OpendTect survey set points (verified, not estimated)."""

    def __init__(self, spec: CBVSSurveySpec):
        self.il_min = spec.inl_start
        self.il_max = spec.inl_stop
        self.xl_min = spec.crl_start
        self.xl_max = spec.crl_stop
        self.il_min_bound = spec.inl_start
        self.il_max_bound = spec.inl_stop
        self.xl_min_bound = spec.crl_start
        self.xl_max_bound = spec.crl_stop
        self.depth_axis = "time"
        self.sample_interval_s = spec.z_step
        self._spec = spec

        pts = spec.coord_points or []
        if len(pts) < 3:
            raise ValueError("Penobscot survey requires at least 3 set points")
        a = np.array([[p[0], p[1], 1.0] for p in pts], dtype=np.float64)
        bx = np.array([p[2] for p in pts], dtype=np.float64)
        by = np.array([p[3] for p in pts], dtype=np.float64)
        self._xy_to_il, _, _, _ = np.linalg.lstsq(a, np.array([p[0] for p in pts], dtype=np.float64), rcond=None)
        self._xy_to_xl, _, _, _ = np.linalg.lstsq(a, np.array([p[1] for p in pts], dtype=np.float64), rcond=None)
        # Forward: il,xl -> x,y (for reference)
        self._ilxl_to_x, _, _, _ = np.linalg.lstsq(a, bx, rcond=None)
        self._ilxl_to_y, _, _, _ = np.linalg.lstsq(a, by, rcond=None)
        # Inverse: x,y -> il,xl
        ab = np.array([[p[2], p[3], 1.0] for p in pts], dtype=np.float64)
        self._utm_to_il, _, _, _ = np.linalg.lstsq(ab, np.array([p[0] for p in pts], dtype=np.float64), rcond=None)
        self._utm_to_xl, _, _, _ = np.linalg.lstsq(ab, np.array([p[1] for p in pts], dtype=np.float64), rcond=None)

    def xy_to_ilxl(self, easting: float, northing: float) -> Tuple[float, float]:
        v = np.array([easting, northing, 1.0], dtype=np.float64)
        return float(v @ self._utm_to_il), float(v @ self._utm_to_xl)

    def utm_to_ilxl(self, easting: float, northing: float) -> Tuple[float, float]:
        return self.xy_to_ilxl(easting, northing)

    def latlon_to_ilxl(self, lat: float, lon: float) -> Tuple[float, float]:
        from .well_seismic_tie import latlon_to_utm

        _, e, n = latlon_to_utm(lat, lon)
        return self.xy_to_ilxl(e, n)


def is_field_geometry_verified(
    field: str,
    well_name: str,
    well_metadata: Dict,
    deviation_inventory: Dict,
    trajectory_has_deviation: bool,
) -> bool:
    """Strict geometry check — no estimated coords or assumed-vertical RMOTC wells."""
    meta = well_metadata.get(well_name, {})
    dev = deviation_inventory.get(well_name, {})

    if field == "volve":
        from .prepare_volve_data import is_geometry_verified

        return is_geometry_verified(
            well_name, well_metadata, deviation_inventory, trajectory_has_deviation
        )

    if field == "rmotc":
        if dev.get("source") != "rmotc_directional":
            return False
        if meta.get("surface_x") is None or meta.get("surface_y") is None:
            return False
        if meta.get("coord_source") != "rmotc_well_headers":
            return False
        return trajectory_has_deviation

    if field == "penobscot":
        if meta.get("coord_source") != "penobscot_survey":
            return False
        if meta.get("latitude") is None or meta.get("longitude") is None:
            return False
        if meta.get("surface_x") is None or meta.get("surface_y") is None:
            return False
        return True

    return False


def calibrate_rmotc_geometry(segy_path: Path, segy_loader) -> AffinePlaneGeometry:
    """Fit IL/XL from SEG-Y source X/Y headers (Wyoming State Plane, scale /10)."""
    sample_step = max(1, segy_loader.num_traces // 8000)
    ils, xls, es, ns = [], [], [], []

    with open(segy_path, "rb") as f:
        f.seek(3600)
        for i in range(segy_loader.num_traces):
            th = f.read(240)
            if i % sample_step == 0:
                il = struct.unpack(">i", th[188:192])[0]
                xl = struct.unpack(">i", th[192:196])[0]
                e = struct.unpack(">i", th[72:76])[0] / 10.0
                n = struct.unpack(">i", th[76:80])[0] / 10.0
                ils.append(il)
                xls.append(xl)
                es.append(e)
                ns.append(n)
            f.seek(segy_loader.num_samples * 4, 1)

    arr_e = np.array(es, dtype=np.float64)
    arr_n = np.array(ns, dtype=np.float64)
    arr_il = np.array(ils, dtype=np.float64)
    arr_xl = np.array(xls, dtype=np.float64)
    design = np.column_stack([arr_e, arr_n, np.ones_like(arr_e)])
    il_coef, _, _, _ = np.linalg.lstsq(design, arr_il, rcond=None)
    xl_coef, _, _, _ = np.linalg.lstsq(design, arr_xl, rcond=None)

    interval_us = 2000
    return AffinePlaneGeometry(
        il_min=segy_loader.inline_min,
        il_max=segy_loader.inline_max,
        xl_min=segy_loader.xline_min,
        xl_max=segy_loader.xline_max,
        il_from_xy=il_coef,
        xl_from_xy=xl_coef,
        depth_axis="time",
        sample_interval_s=interval_us / 1e6,
    )


def build_penobscot_geometry(project_root: Path) -> PenobscotSurveyGeometry:
    survey_path = project_root / "data" / "penobscot" / "raw" / "Penobscot" / ".survey"
    if not survey_path.exists():
        raw = project_root / "data" / "penobscot" / "prepared" / "survey.json"
        if raw.exists():
            text = json.loads(raw.read_text(encoding="utf-8")).get("text_excerpt", "")
        else:
            raise FileNotFoundError("Penobscot .survey file not found")
    else:
        text = survey_path.read_text(encoding="utf-8", errors="replace")
    spec = parse_penobscot_survey(text)
    return PenobscotSurveyGeometry(spec)


def build_field_survey_geometry(field: str, project_root: Path, segy_loader) -> object:
    if field == "volve":
        geom = DEFAULT_VOLVE_GEOMETRY
        geom.depth_axis = "depth"
        geom.sample_interval_s = 0.004
        return geom
    if field == "rmotc":
        layout_path = project_root / "data" / "rmotc" / "prepared" / "data_layout.json"
        segy_path = Path(json.loads(layout_path.read_text())["segy_3d"])
        return calibrate_rmotc_geometry(segy_path, segy_loader)
    if field == "penobscot":
        return build_penobscot_geometry(project_root)
    raise ValueError(f"Unknown field: {field}")


def _deviation_csv_path(
    field: str, well_name: str, meta: Dict, deviations_dir: Path
) -> Optional[Path]:
    if field == "rmotc":
        api = meta.get("api")
        if api:
            p = deviations_dir / f"{api}.csv"
            if p.exists():
                return p
    p = deviations_dir / f"{well_name}.csv"
    return p if p.exists() else None


def build_field_trajectories(
    field: str,
    well_metadata: Dict,
    deviations_dir: Path,
) -> Dict[str, WellTrajectory]:
    if field == "volve":
        trajectories = build_well_trajectories(VOLVE_WELL_COORDS, well_metadata=well_metadata)
        if deviations_dir.exists():
            for well_name, traj in trajectories.items():
                csv_path = deviations_dir / f"{well_name}.csv"
                if csv_path.exists():
                    traj.load_deviation_csv(str(csv_path))
        return trajectories

    trajectories: Dict[str, WellTrajectory] = {}
    for well_name, meta in well_metadata.items():
        kb = meta.get("kb_elevation_m") or 0.0
        gl = meta.get("ground_elevation_m", 0.0)

        if field == "rmotc":
            sx, sy = meta.get("surface_x"), meta.get("surface_y")
            if sx is None or sy is None:
                continue
            traj = WellTrajectory(
                well_name=well_name,
                kb_elevation=kb,
                ground_elevation=gl,
                surface_e=float(sx),
                surface_n=float(sy),
            )
        elif field == "penobscot":
            lat, lon = meta.get("latitude"), meta.get("longitude")
            if lat is None or lon is None:
                continue
            traj = WellTrajectory(
                well_name=well_name,
                surface_lat=float(lat),
                surface_lon=float(lon),
                kb_elevation=kb,
                ground_elevation=gl,
            )
        else:
            continue

        csv_path = _deviation_csv_path(field, well_name, meta, deviations_dir)
        if csv_path is not None:
            traj.load_deviation_csv(str(csv_path))
        trajectories[well_name] = traj

    return trajectories


def resolve_field_paths(project_root: Path, field: str) -> Dict[str, Path]:
    root = Path(project_root)
    if field == "volve":
        from .prepare_volve_data import resolve_project_paths

        paths = resolve_project_paths(root)
        layout = json.loads((paths["prepared_dir"] / "data_layout.json").read_text(encoding="utf-8"))
        return {
            "prepared_dir": paths["prepared_dir"],
            "segy_path": Path(layout.get("segy_path") or paths["seismic_dir"] / "volve.segy"),
            "deviations_dir": paths["deviations_dir"],
        }
    if field == "rmotc":
        prepared = root / "data" / "rmotc" / "prepared"
        layout = json.loads((prepared / "data_layout.json").read_text(encoding="utf-8"))
        return {
            "prepared_dir": prepared,
            "segy_path": Path(layout["segy_3d"]),
            "deviations_dir": prepared / "deviations",
        }
    if field == "penobscot":
        prepared = root / "data" / "penobscot" / "prepared"
        layout = json.loads((prepared / "data_layout.json").read_text(encoding="utf-8"))
        return {
            "prepared_dir": prepared,
            "segy_path": Path(layout["segy_3d"]),
            "deviations_dir": prepared / "deviations",
        }
    raise ValueError(f"Unknown field: {field}")
