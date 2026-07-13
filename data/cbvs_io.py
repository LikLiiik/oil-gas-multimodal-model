"""
Minimal OpendTect CBVS reader for Penobscot-style 3D post-stack volumes.

CBVS is a proprietary OpendTect format. This module implements a validated
int16 payload reader for single-file, inline-sorted volumes where:

    file_size ≈ header_trailer_bytes + n_inlines * n_crosslines * n_samples * 2

The reader probes a small header window, validates the inferred cube against
expected survey dimensions, and can export IEEE SEG-Y via segyio.

Reference survey (Penobscot): IL 1000–1600, XL 1000–1481, Z 0–6 s @ 4 ms.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import segyio
    from segyio import BinField, TraceField
except ImportError as e:
    raise ImportError("cbvs_io requires segyio") from e


@dataclass
class CBVSSurveySpec:
    """Expected 3D geometry for validation."""

    inl_start: int
    inl_stop: int
    inl_step: int
    crl_start: int
    crl_stop: int
    crl_step: int
    z_start: float
    z_stop: float
    z_step: float
    coord_points: Optional[List[Tuple[int, int, float, float]]] = None

    @property
    def n_inlines(self) -> int:
        return (self.inl_stop - self.inl_start) // self.inl_step + 1

    @property
    def n_crosslines(self) -> int:
        return (self.crl_stop - self.crl_start) // self.crl_step + 1

    @property
    def n_samples(self) -> int:
        return int(round((self.z_stop - self.z_start) / self.z_step)) + 1

    @property
    def n_traces(self) -> int:
        return self.n_inlines * self.n_crosslines

    @property
    def payload_bytes(self) -> int:
        return self.n_traces * self.n_samples * 2


def parse_penobscot_survey(text: str) -> CBVSSurveySpec:
    """Parse OpendTect ``.survey`` text into a CBVSSurveySpec."""

    def _range_line(label: str) -> Tuple[int, int, int]:
        m = re.search(rf"{label}:\s*(\d+)`(\d+)`(\d+)", text)
        if not m:
            raise ValueError(f"Could not parse {label} from survey file")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def _z_range() -> Tuple[float, float, float]:
        m = re.search(r"Z range:\s*([0-9.]+)`([0-9.]+)`([0-9.]+)`T", text)
        if not m:
            raise ValueError("Could not parse Z range from survey file")
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    inl_start, inl_stop, inl_step = _range_line("In-line range")
    crl_start, crl_stop, crl_step = _range_line("Cross-line range")
    z_start, z_stop, z_step = _z_range()

    points: List[Tuple[int, int, float, float]] = []
    for m in re.finditer(
        r"Set Point\.\d+:\s*(\d+)/(\d+)`\(([-0-9.]+),([-0-9.]+)\)", text
    ):
        points.append(
            (int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)))
        )

    return CBVSSurveySpec(
        inl_start=inl_start,
        inl_stop=inl_stop,
        inl_step=inl_step,
        crl_start=crl_start,
        crl_stop=crl_stop,
        crl_step=crl_step,
        z_start=z_start,
        z_stop=z_stop,
        z_step=z_step,
        coord_points=points or None,
    )


def _score_int16_payload(arr: np.ndarray) -> float:
    """Higher is better: non-flat, non-empty seismic-like amplitudes."""
    if arr.size == 0:
        return -np.inf
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -np.inf
    std = float(np.std(finite))
    if std < 1e-3:
        return -np.inf
    uniq_ratio = len(np.unique(finite[: min(50000, finite.size)])) / min(
        50000, finite.size
    )
    nonzero_ratio = float(np.mean(finite != 0))
    return std * uniq_ratio * (0.5 + nonzero_ratio)


def find_int16_payload_offset(
    cbvs_path: Path,
    spec: CBVSSurveySpec,
) -> Tuple[int, float]:
    """
    Locate int16 payload start. Penobscot CBVS uses payload-at-tail layout:
        file_size = header_trailer_bytes + n_traces * n_samples * 2
    """
    file_size = cbvs_path.stat().st_size
    expected = spec.payload_bytes
    count = spec.n_traces * spec.n_samples

    if expected <= 0 or expected > file_size:
        raise ValueError(f"Invalid payload size {expected} for file size {file_size}")

    candidates = []
    tail_offset = file_size - expected
    if tail_offset >= 0:
        candidates.append(tail_offset)

    # Fallback: small fixed header sizes seen in OpendTect CBVS exports
    for hdr in (0, 1024, 4096, 3715820):
        if hdr not in candidates and hdr + expected <= file_size:
            candidates.append(hdr)

    best_offset, best_score = 0, -np.inf
    for offset in candidates:
        arr = np.fromfile(cbvs_path, dtype=np.int16, offset=offset, count=count)
        if arr.size != count:
            continue
        score = _score_int16_payload(arr)
        if score > best_score:
            best_score, best_offset = score, offset

    if not np.isfinite(best_score):
        raise RuntimeError(f"Could not locate int16 CBVS payload in {cbvs_path}")

    return best_offset, best_score


def read_cbvs_int16_volume(
    cbvs_path: Path,
    spec: CBVSSurveySpec,
    offset: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Read CBVS as inline-major int16 array shaped (n_inl, n_crl, n_samples).

    Returns (volume, info_dict).
    """
    cbvs_path = Path(cbvs_path)
    if offset is None:
        offset, score = find_int16_payload_offset(cbvs_path, spec)
    else:
        score = np.nan

    count = spec.n_traces * spec.n_samples
    raw = np.fromfile(cbvs_path, dtype=np.int16, offset=offset, count=count)
    if raw.size != count:
        raise RuntimeError(
            f"Expected {count} int16 samples, got {raw.size} at offset {offset}"
        )

    volume = raw.reshape(spec.n_inlines, spec.n_crosslines, spec.n_samples)

    # Basic sanity checks
    finite = volume[np.isfinite(volume)]
    if finite.size == 0 or float(np.std(finite)) < 1e-3:
        raise RuntimeError("CBVS payload looks flat/empty after reshape")

    info = {
        "offset": offset,
        "score": float(score) if np.isfinite(score) else None,
        "dtype": "int16",
        "shape": list(volume.shape),
        "amplitude_std": float(np.std(finite)),
        "amplitude_min": float(np.min(finite)),
        "amplitude_max": float(np.max(finite)),
    }
    return volume, info


def ilxl_to_xy(
    inl: int,
    crl: int,
    spec: CBVSSurveySpec,
) -> Tuple[float, float]:
    """Affine map inline/crossline to UTM XY using survey set points."""
    pts = spec.coord_points or []
    if len(pts) < 3:
        raise ValueError("Need at least 3 survey set points for XY mapping")

    a = np.array([[p[0], p[1], 1.0] for p in pts], dtype=np.float64)
    bx = np.array([p[2] for p in pts], dtype=np.float64)
    by = np.array([p[3] for p in pts], dtype=np.float64)
    cx, _, _, _ = np.linalg.lstsq(a, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(a, by, rcond=None)
    x = cx[0] * inl + cx[1] * crl + cx[2]
    y = cy[0] * inl + cy[1] * crl + cy[2]
    return float(x), float(y)


def cbvs_to_segy(
    cbvs_path: Path,
    segy_path: Path,
    spec: CBVSSurveySpec,
    scale: float = 1.0,
    validate: bool = True,
) -> Dict:
    """
    Convert CBVS int16 volume to IEEE float SEG-Y (inline-sorted traces).

    Uses vectorized header construction and bulk trace write for speed.
    """
    cbvs_path = Path(cbvs_path)
    segy_path = Path(segy_path)
    segy_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Reading CBVS: {cbvs_path.name} ...")
    volume, read_info = read_cbvs_int16_volume(cbvs_path, spec)
    n_inl, n_crl, n_samp = volume.shape
    n_tr = n_inl * n_crl

    sample_interval_us = int(round(spec.z_step * 1e6))
    samples = np.arange(n_samp, dtype=np.int32)

    spec_obj = segyio.spec()
    spec_obj.format = 5
    spec_obj.sorting = 2
    spec_obj.samples = samples.tolist()
    spec_obj.tracecount = n_tr
    spec_obj.t0 = int(round(spec.z_start * 1e6))

    print(f"  Building {n_tr} traces ({n_inl}x{n_crl}x{n_samp}) ...")
    traces = volume.reshape(n_tr, n_samp).astype(np.float32) * scale

    trace_idx = np.arange(n_tr, dtype=np.int32)
    i_inl = trace_idx // n_crl
    i_crl = trace_idx % n_crl
    inlines = spec.inl_start + i_inl * spec.inl_step
    crosslines = spec.crl_start + i_crl * spec.crl_step

    pts = spec.coord_points or []
    a = np.array([[p[0], p[1], 1.0] for p in pts], dtype=np.float64)
    bx = np.array([p[2] for p in pts], dtype=np.float64)
    by = np.array([p[3] for p in pts], dtype=np.float64)
    cx, _, _, _ = np.linalg.lstsq(a, bx, rcond=None)
    cy, _, _, _ = np.linalg.lstsq(a, by, rcond=None)
    xs = (cx[0] * inlines + cx[1] * crosslines + cx[2]).astype(np.int32)
    ys = (cy[0] * inlines + cy[1] * crosslines + cy[2]).astype(np.int32)

    print(f"  Writing SEG-Y: {segy_path.name} ...")
    with segyio.create(str(segy_path), spec_obj) as f:
        f.bin[BinField.Interval] = sample_interval_us
        f.bin[BinField.Samples] = n_samp
        f.bin[BinField.Format] = 5

        for i in range(n_tr):
            h = f.header[i]
            h[TraceField.INLINE_3D] = int(inlines[i])
            h[TraceField.CROSSLINE_3D] = int(crosslines[i])
            h[TraceField.CDP_X] = int(xs[i])
            h[TraceField.CDP_Y] = int(ys[i])
            h[TraceField.SourceX] = int(xs[i])
            h[TraceField.SourceY] = int(ys[i])
            h[TraceField.GroupX] = int(xs[i])
            h[TraceField.GroupY] = int(ys[i])
            if i > 0 and i % 50000 == 0:
                print(f"    headers {i}/{n_tr} ...")

        f.trace.raw[:] = np.ascontiguousarray(traces)

    meta = {
        "source_cbvs": str(cbvs_path),
        "output_segy": str(segy_path),
        "scale_applied": scale,
        **read_info,
    }

    if validate:
        print("  Validating SEG-Y ...")
        meta["validation"] = validate_segy_cube(segy_path, spec)

    return meta


def validate_segy_cube(segy_path: Path, spec: CBVSSurveySpec) -> Dict:
    """Validate written SEG-Y against expected trace/sample counts and ranges."""
    segy_path = Path(segy_path)
    with segyio.open(str(segy_path), "r", ignore_geometry=True) as f:
        n_tr = f.tracecount
        n_samp = len(f.samples)
        if n_tr != spec.n_traces:
            raise RuntimeError(f"SEG-Y trace count {n_tr} != expected {spec.n_traces}")
        if n_samp != spec.n_samples:
            raise RuntimeError(f"SEG-Y samples {n_samp} != expected {spec.n_samples}")

        inlines = {f.header[i][TraceField.INLINE_3D] for i in range(min(n_tr, 5000))}
        crosslines = {f.header[i][TraceField.CROSSLINE_3D] for i in range(min(n_tr, 5000))}

        sample0 = f.trace[0].astype(np.float64)
        sample_mid = f.trace[n_tr // 2].astype(np.float64)

        return {
            "ok": True,
            "tracecount": n_tr,
            "nsamples": n_samp,
            "sample_interval_us": int(f.bin[BinField.Interval]),
            "inline_min": int(min(inlines)),
            "inline_max": int(max(inlines)),
            "crossline_min": int(min(crosslines)),
            "crossline_max": int(max(crosslines)),
            "trace0_std": float(np.std(sample0)),
            "trace_mid_std": float(np.std(sample_mid)),
        }
