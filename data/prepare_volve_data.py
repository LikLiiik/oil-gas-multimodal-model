"""
Prepare Volve dataset from locally available files.

Scans existing seismic SEG-Y, well LAS logs, and LWD MWD inclination curves,
then writes standardized metadata and deviation surveys under data/prepared/.

Usage:
    python -m data.prepare_volve_data
    python scripts/prepare_data.py --project_root /path/to/oil-gas-multimodal-model
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

STANDARD_CURVES = ["GR", "SP", "CAL", "RD", "MLL", "MSFL", "NPHI", "RHOB", "DT"]

# Nine conventional log slots. RD accepts RT as fallback when loading LAS data.
# MLL/MSFL use wireline-style mnemonics only — NOT LWD ARC curves (RACELM etc.).
CURVE_ALIASES = {
    "GR": ["GR", "GR_1", "GRC", "SGR", "GAMMA", "GR_CAL", "NBGRCFM", "LFP_GR", "GRM1"],
    "SP": ["SP", "SP_1", "SPR"],
    "CAL": ["CALI", "CAL", "HCAL", "CALIPER", "BS", "LFP_CALI", "C1", "C2"],
    "RD": ["RD", "LLD", "ILD", "RLLD", "RILD", "AT90", "AT90M", "RLA5", "RLA1", "RDEEP", "LFP_RT"],
    "RT_FALLBACK": ["RT"],  # merged into RD slot when RD absent; not a separate channel
    "MLL": ["MLL", "LLM", "ILM", "RLLM", "RILM", "RMS", "ILS", "LLS", "RLLS", "RLM", "RLL2",
            "AT10", "AT20", "RLA2", "RLA3", "RLA4", "RS", "RSHAL", "RSHALOW", "LFP_RLM"],
    "MSFL": ["MSFL", "MCFL", "RMC", "RMCFL", "RMLL", "RXO", "RMSFL", "RLL3", "RLL4", "RMCF",
             "RFLU", "RFL", "LFP_RXO"],
    "NPHI": ["NPHI", "NPHI_1", "TNPH", "NEUTRON", "NEU", "LFP_NPHI", "CN", "CNL", "NPOR"],
    "RHOB": ["RHOB", "RHOZ", "RHO8", "DEN", "DENSITY", "LFP_RHOB", "ZDL", "RHO"],
    "DT": ["DT", "DTCO", "AC", "SONIC", "DTC", "LFP_DT", "DTP", "ITT"],
}

# Explicitly excluded from resistivity slots (different physics / not equivalent):
# RACELM, RPCELM, RACEHM, RPCEHM — LWD ARC frequency/attenuation components
# RM — mud resistivity
EXCLUDED_RESISTIVITY_ALIASES = frozenset({
    "RACELM", "RPCELM", "RACEHM", "RPCEHM", "RM", "RPCEHM", "ARC_GR_RT", "ARC_GR_UNC_RT",
})

# Train/val wells must have verified surface coords and real deviation (not hardcoded / vertical assumed).
VERIFIED_COORD_SOURCES = frozenset({"witsml_wellinfo", "edm_cd_well", "las_header"})
VERIFIED_DEVIATION_SOURCES = frozenset({"witsml", "edm_definitive", "sodir", "lwd_mwd"})


def is_geometry_verified(
    well_name: str,
    well_metadata: Dict,
    deviation_inventory: Dict,
    trajectory_has_deviation: bool = False,
) -> bool:
    """True when well has verified lat/lon and a real deviation survey loaded."""
    meta = well_metadata.get(well_name, {})
    if meta.get("geometry_verified") is False:
        return False
    if meta.get("geometry_verified") is True:
        return bool(trajectory_has_deviation)

    lat, lon = meta.get("latitude"), meta.get("longitude")
    if lat is None or lon is None:
        return False
    if meta.get("coord_source") not in VERIFIED_COORD_SOURCES:
        return False

    dev_src = deviation_inventory.get(well_name, {}).get("source")
    if dev_src not in VERIFIED_DEVIATION_SOURCES:
        return False
    return bool(trajectory_has_deviation)


def annotate_geometry_quality(
    well_metadata: Dict,
    deviation_inventory: Dict,
) -> None:
    """Set geometry_verified and exclusion reason on each well (in place)."""
    for well_name, meta in well_metadata.items():
        dev = deviation_inventory.get(well_name, {})
        has_dev = dev.get("source") in VERIFIED_DEVIATION_SOURCES
        verified = is_geometry_verified(
            well_name, well_metadata, deviation_inventory, has_dev
        )
        meta["geometry_verified"] = verified
        if verified:
            meta.pop("geometry_excluded_reason", None)
            continue
        reasons = []
        if meta.get("latitude") is None or meta.get("longitude") is None:
            reasons.append("no_verified_latlon")
        elif meta.get("coord_source") not in VERIFIED_COORD_SOURCES:
            reasons.append(f"coord_source={meta.get('coord_source')}")
        if dev.get("source") not in VERIFIED_DEVIATION_SOURCES:
            reasons.append(f"deviation={dev.get('source', 'none')}")
        meta["geometry_excluded_reason"] = ",".join(reasons) if reasons else "unknown"


SODIR_WELLBORE_IDS = {
    "15_9-19 A": 3372,
    "15_9-19 B": 3373,
    "15_9-19 S": 3374,
    "15_9-F-1": 3804,
    "15_9-F-1 A": 5893,
    "15_9-F-1 B": 5894,
    "15_9-F-1 C": 5895,
    "15_9-F-4": 5516,
    "15_9-F-5": 5517,
    "15_9-F-7": 5518,
    "15_9-F-9": 5948,
    "15_9-F-9 A": 7117,
    "15_9-F-10": 6203,
    "15_9-F-11": 6204,
    "15_9-F-11 A": 7083,
    "15_9-F-11 B": 7118,
    "15_9-F-11 T2": 7410,
    "15_9-F-12": 6303,
    "15_9-F-14": 6698,
    "15_9-F-15": 6699,
    "15_9-F-15 A": 7084,
    "15_9-F-15 B": 7119,
    "15_9-F-15 C": 7276,
    "15_9-F-15 D": 7580,
}


def resolve_project_paths(project_root: Optional[Path] = None) -> Dict[str, Path]:
    """Resolve seismic, well log, and prepared output directories."""
    root = Path(project_root or Path(__file__).resolve().parents[1])

    seismic_dir = root / "seismic"
    segy_files = list(seismic_dir.glob("*.segy")) + list(seismic_dir.glob("*.SEGY"))
    if not segy_files:
        segy_files = list(root.glob("*.segy")) + list(root.glob("*.SEGY"))

    well_dir = root / "data" / "Volve_Well_logs_pr_WELL" / "Well_logs_pr_WELL"
    if not well_dir.exists():
        alt = root / "Volve_Well_logs_pr_WELL" / "Well_logs_pr_WELL"
        well_dir = alt if alt.exists() else well_dir

    prepared_dir = root / "data" / "prepared"
    deviations_dir = prepared_dir / "deviations"

    witsml_dir = root / "data" / "Volve_WITSML_Realtime_drilling_data" / "WITSML Realtime drilling data"
    edm_path = (
        root / "data" / "Volve_Well_technical_data" / "Well_technical_data"
        / "EDM.XML" / "Volve F.edm.xml"
    )

    return {
        "project_root": root,
        "seismic_dir": seismic_dir,
        "segy_path": segy_files[0] if segy_files else None,
        "well_dir": well_dir,
        "prepared_dir": prepared_dir,
        "deviations_dir": deviations_dir,
        "legacy_deviations_dir": root / "data" / "volve_deviations",
        "witsml_dir": witsml_dir,
        "edm_path": edm_path,
    }


# ---------------------------------------------------------------------------
# LAS helpers
# ---------------------------------------------------------------------------

def _read_las_text(path: Path) -> str:
    return path.read_bytes().decode("latin-1", errors="replace")


def _parse_header_value(line: str) -> Optional[str]:
    if ":" in line:
        val_part = line.split(":", 1)[0]
    else:
        val_part = line
    tokens = val_part.split()
    if len(tokens) >= 2:
        return tokens[-1]
    return tokens[0] if tokens else None


def _parse_float(val: Optional[str], null: float = -999.25) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if abs(f - null) < 1e-3 or abs(f + 999.25) < 1e-3:
            return None
        return f
    except ValueError:
        return None


def _dms_from_line(line: str, default_hemi: str) -> Optional[float]:
    m = re.search(r"(\d+)[°\s]+(\d+)['\s]+([\d.]+)[\"']?\s*([NSEW])?", line)
    if not m:
        m = re.search(r"(\d+)\s+(\d+)'\s*([\d.]+)", line)
        if not m:
            return None
        hemi = default_hemi
        deg, minute, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
    else:
        deg, minute, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
        hemi = m.group(4) or default_hemi

    decimal = deg + minute / 60.0 + sec / 3600.0
    if hemi.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def parse_las_well_header(las_path: Path) -> Dict:
    """Parse ~W section for coordinates and elevation datums."""
    text = _read_las_text(las_path)
    header: Dict[str, str] = {}
    in_well = False

    for line in text.split("\n"):
        if line.startswith("~W"):
            in_well = True
            continue
        if line.startswith("~") and not line.startswith("~W"):
            if in_well:
                break
            continue
        if not in_well or "." not in line or line.startswith("#"):
            continue

        mnem = line.split(".")[0].strip().upper().rstrip(".")
        val = _parse_header_value(line)
        if val is not None:
            header[mnem] = val

    result: Dict = {"las_path": str(las_path), "header_raw": header}

    # Coordinates
    for key, hemi in [("LATI", "N"), ("LAT", "N"), ("LONG", "E"), ("LON", "E")]:
        if key in header:
            latlon = _dms_from_line(header[key], hemi)
            if latlon is not None:
                if key.startswith("LAT") and abs(latlon) > 10:
                    result["latitude"] = latlon
                elif not key.startswith("LAT") and abs(latlon) > 10:
                    result["longitude"] = latlon

    # Depth reference / elevations (meters above MSL unless noted)
    elev_type = header.get("ELEV_TYPE", header.get("ELEV_TYPE.", "")).upper()
    for key in ("APD", "EDF", "EKB", "ELEV", "EGL", "STRT"):
        if key in header:
            f = _parse_float(header[key])
            if f is not None:
                result[key.lower()] = f

    # Resolve KB (depth reference above MSL)
    kb = None
    if elev_type == "KB" and result.get("elev") is not None:
        kb = result["elev"]
    elif result.get("apd") is not None:
        kb = result["apd"]
    elif result.get("edf") is not None:
        kb = result["edf"]
    elif result.get("ekb") is not None:
        kb = result["ekb"]

    gl = result.get("egl")
    if gl is None and kb is not None and abs(kb - 54.9) < 1.0:
        gl = -91.0  # Volve subsea template default

    result["kb_elevation_m"] = kb
    result["ground_elevation_m"] = gl
    result["log_start_md"] = result.get("strt")

    # Infer subsea template KB from log start depth (RT to seafloor ≈ 145.9m)
    strt = result.get("strt")
    if kb is None and strt is not None and 140.0 <= strt <= 200.0:
        result["kb_elevation_m"] = 54.9
        if gl is None:
            result["ground_elevation_m"] = -91.0
    elif kb is None and gl is not None and gl < -50:
        result["kb_elevation_m"] = 54.9

    if "WELL" in header:
        result["well_name_las"] = header["WELL"].replace("/", "_").replace(" ", "_")

    return result


def list_las_curves(las_path: Path) -> List[str]:
    curves = []
    text = _read_las_text(las_path)
    in_curve = False
    for line in text.split("\n"):
        if line.startswith("~C"):
            in_curve = True
            continue
        if line.startswith("~") and in_curve:
            break
        if in_curve and "." in line and not line.startswith("#"):
            mnem = line.split(".")[0].strip().upper()
            if mnem not in ("DEPT", "DEPTH", "TIME", "DATE"):
                curves.append(mnem)

    if curves:
        return curves

    # Some Volve LWD files list curve mnemonics on the ~A header line only
    for line in text.split("\n"):
        if line.startswith("~A"):
            parts = line[2:].split()
            for mnem in parts:
                m = mnem.upper()
                if m not in ("DEPT", "DEPTH", "TIME", "DATE"):
                    curves.append(m)
            break
    return curves


def map_standard_curves(raw_curves: List[str]) -> List[str]:
    raw_u = {c.upper() for c in raw_curves}
    found = []
    for std in STANDARD_CURVES:
        aliases = CURVE_ALIASES.get(std, [])
        if any(a.upper() in raw_u for a in aliases):
            found.append(std)
            continue
        if std == "RD" and any(a.upper() in raw_u for a in CURVE_ALIASES.get("RT_FALLBACK", [])):
            found.append(std)
    return found


def apply_rd_rt_fallback(curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fill RD slot from RT when RD is missing (RT is not a separate channel)."""
    rd = curves.get("RD")
    rt = curves.get("RT")
    rd_valid = rd is not None and np.any(np.isfinite(rd))
    rt_valid = rt is not None and np.any(np.isfinite(rt))
    if not rd_valid and rt_valid:
        curves["RD"] = rt.copy()
    if "RT" in curves:
        del curves["RT"]
    return curves


def score_las_file(path: Path) -> Tuple[int, List[str], Dict]:
    """Score LAS by number of standard curves; higher is better."""
    header = parse_las_well_header(path)
    curves = list_las_curves(path)
    std = map_standard_curves(curves)
    score = len(std)

    name = path.name.upper()
    if "INPUT" in name:
        score += 20
    elif "LFP" in name:
        score += 15
    elif "CPI" in name:
        score += 5
    elif "OUTPUT" in name:
        score -= 5

    if header.get("latitude") is not None:
        score += 2
    if header.get("kb_elevation_m") is not None:
        score += 2

    return score, std, header


def find_best_las_for_well(well_dir: Path) -> Optional[Path]:
    """Find the best LAS file for curves + header across all well subfolders."""
    if not well_dir.is_dir():
        return None

    candidates: List[Path] = []
    patterns = [
        "05.PETROPHYSICAL INTERPRETATION/WLC_PETRO_COMPUTED_INPUT*.LAS",
        "05.PETROPHYSICAL INTERPRETATION/WLC_PETRO_COMPUTED_INPUT*.las",
        "05.PETROPHYSICAL INTERPRETATION/*CPI*.las",
        "05.PETROPHYSICAL INTERPRETATION/*CPI*.LAS",
        "06.LFP/*LFP*.las",
        "06.LFP/*LFP*.LAS",
        "04.COMPOSITE/*.las",
        "04.COMPOSITE/*.LAS",
        "04.COMPOSITE/**/*.las",
        "04.COMPOSITE/**/*.LAS",
        "05.PETROPHYSICAL INTERPRETATION/**/*.las",
        "05.PETROPHYSICAL INTERPRETATION/**/*.LAS",
        "05.PETROPHYSICAL INTERPRETATION/*.las",
        "05.PETROPHYSICAL INTERPRETATION/*.LAS",
        "02.LWD_EWL/**/*.LAS",
        "02.LWD_EWL/**/*.las",
        "02.LWD_EWL/*.LAS",
        "02.LWD_EWL/*.las",
    ]
    for pat in patterns:
        candidates.extend(well_dir.glob(pat))

    if not candidates:
        candidates = list(well_dir.rglob("*.las")) + list(well_dir.rglob("*.LAS"))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    if not unique:
        return None

    best = max(unique, key=lambda p: score_las_file(p)[0])
    best_score, best_curves, _ = score_las_file(best)
    if best_score <= 0 and not best_curves:
        return None
    return best


def discover_wells(well_root: Path) -> List[str]:
    wells = []
    if not well_root.exists():
        return wells
    for d in sorted(well_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if find_best_las_for_well(d) is not None:
            wells.append(d.name)
    return wells


# ---------------------------------------------------------------------------
# Deviation extraction
# ---------------------------------------------------------------------------

def extract_lwd_deviation(well_dir: Path, md_step: float = 10.0) -> Optional[Dict]:
    """Extract MD+INCL from LWD/MWD LAS files (real drilling survey)."""
    lwd_dir = well_dir / "02.LWD_EWL"
    if not lwd_dir.exists():
        return None

    las_files = list(lwd_dir.glob("*.LAS")) + list(lwd_dir.glob("*.las"))
    if not las_files:
        return None

    best_pts: List[Tuple[float, float]] = []
    best_meta = {}

    for las_path in las_files:
        text = _read_las_text(las_path)
        if "~A" not in text:
            continue

        header = None
        rows = []
        for line in text.split("\n"):
            if line.startswith("~A"):
                header = [h for h in line[2:].split() if h]
                continue
            if header and line.strip() and not line.startswith("#") and not line.startswith("~"):
                parts = line.split()
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts)))

        if not rows:
            continue

        incl_col = next((c for c in header if "INCL" in c.upper()), None)
        dept_col = next((c for c in header if c in ("DEPT", "DEPTH")), None)
        if not incl_col or not dept_col:
            continue

        pts = []
        for r in rows:
            try:
                md = float(r[dept_col])
                inc = float(r[incl_col])
            except ValueError:
                continue
            if md <= 0 or md > 8000:
                continue
            if inc < 0 or inc > 90 or abs(inc + 999.25) < 1e-3:
                continue
            pts.append((md, inc))

        if len(pts) > len(best_pts):
            best_pts = pts
            best_meta = {"source_file": las_path.name, "incl_column": incl_col}

    if len(best_pts) < 10:
        return None

    best_pts.sort(key=lambda x: x[0])

    # Downsample to md_step
    sampled = [best_pts[0]]
    last_md = best_pts[0][0]
    for md, inc in best_pts[1:]:
        if md - last_md >= md_step:
            sampled.append((md, inc))
            last_md = md
    if sampled[-1][0] != best_pts[-1][0]:
        sampled.append(best_pts[-1])

    return {
        "source": "lwd_mwd",
        "points": len(sampled),
        "max_inclination": max(p[1] for p in sampled),
        "md_range": [sampled[0][0], sampled[-1][0]],
        **best_meta,
        "survey": sampled,
    }


def write_deviation_csv(survey: List[Tuple[float, float]], out_path: Path, source: str):
    """Write MD,INCL,AZI,TVD,NS,EW CSV. AZI=0 when unknown (inclination-only MWD)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MD", "INCL", "AZI", "TVD", "NS", "EW", "SOURCE"])
        for md, inc in survey:
            inc_rad = np.radians(inc)
            tvd = md * np.cos(inc_rad)  # approximate for vertical start
            ns = md * np.sin(inc_rad)  # simplified; AZI unknown
            w.writerow([f"{md:.2f}", f"{inc:.2f}", "0.00", f"{tvd:.2f}", f"{ns:.2f}", "0.00", source])


# Survey point: (MD, INCL, AZI, TVD, NS, EW) — TVD/NS/EW may be None
SurveyPoint = Tuple[float, float, float, Optional[float], Optional[float], Optional[float]]

FT_TO_M = 0.3048


def normalize_well_name(name: str) -> str:
    """Map WITSML/EDM well names to LAS folder convention (15_9-F-*)."""
    name = re.sub(r"^NO\s+", "", name.strip())
    name = name.replace("/", "_")
    name = re.sub(r"\s*-\s*Main Wellbore\s*$", "", name, flags=re.I)
    name = re.sub(r"\s*-\s*Original Hole\s*$", "", name, flags=re.I)
    m = re.match(r"^(15_9-F-\d+)([A-Z])$", name.replace(" ", ""))
    if m:
        name = f"{m.group(1)} {m.group(2)}"
    return name.strip()


def _well_name_from_witsml(name_well: str, name_wellbore: str) -> str:
    if name_wellbore:
        wb = normalize_well_name(name_wellbore)
        if re.match(r"15_9-", wb):
            return wb
    return normalize_well_name(name_well)


def _parse_witsml_trajectory_file(path: Path) -> Optional[Tuple[str, List[SurveyPoint], bool]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    nw = re.search(r"<nameWell>([^<]+)</nameWell>", text)
    if not nw:
        return None
    nwb = re.search(r"<nameWellbore>([^<]+)</nameWellbore>", text)
    well = _well_name_from_witsml(nw.group(1), nwb.group(1) if nwb else "")
    tname = re.search(r"<trajectory[^>]*>.*?<name>([^<]+)</name>", text, re.DOTALL)
    is_actual = bool(tname and "Actual Traj" in tname.group(1))

    points: List[SurveyPoint] = []
    for block in re.findall(r"<trajectoryStation[^>]*>(.*?)</trajectoryStation>", text, re.DOTALL):
        def _get(tag: str) -> Optional[float]:
            m = re.search(rf"<{tag} uom=\"[^\"]*\">([^<]+)</{tag}>", block)
            if not m:
                return None
            try:
                return float(m.group(1))
            except ValueError:
                return None

        md, inc = _get("md"), _get("incl")
        if md is None or inc is None:
            continue
        points.append((md, inc, _get("azi") or 0.0, _get("tvd"), _get("dispNs"), _get("dispEw")))

    if not points:
        return None
    return well, points, is_actual


def _merge_survey_points(
    raw: List[Tuple[SurveyPoint, bool]],
) -> List[SurveyPoint]:
    """Merge stations from multiple trajectory files; dedupe by MD."""
    by_md: Dict[float, Tuple[SurveyPoint, bool]] = {}
    for pt, is_actual in raw:
        key = round(pt[0], 2)

        def _score(p: SurveyPoint, actual: bool) -> Tuple[int, int]:
            completeness = sum(1 for x in p[3:] if x is not None)
            return (1 if actual else 0, completeness)

        old = by_md.get(key)
        if old is None or _score(pt, is_actual) > _score(old[0], old[1]):
            by_md[key] = (pt, is_actual)
    return [p for p, _ in sorted(by_md.values(), key=lambda x: x[0][0])]


def collect_witsml_trajectories(witsml_dir: Path) -> Dict[str, List[SurveyPoint]]:
    if not witsml_dir.exists():
        return {}

    raw_by_well: Dict[str, List[Tuple[SurveyPoint, bool]]] = {}
    for traj_path in witsml_dir.glob("*/*/trajectory/*.xml"):
        parsed = _parse_witsml_trajectory_file(traj_path)
        if parsed is None:
            continue
        well, points, is_actual = parsed
        raw_by_well.setdefault(well, []).extend((p, is_actual) for p in points)

    return {well: _merge_survey_points(raw) for well, raw in raw_by_well.items()}


def collect_edm_trajectories(edm_path: Path) -> Dict[str, List[SurveyPoint]]:
    """Parse EDM CD_DEFINITIVE_SURVEY (ACTUAL, definitive) — units converted ft -> m."""
    if not edm_path.exists():
        return {}

    wellbores: Dict[str, str] = {}
    best_header: Dict[str, Tuple[str, float]] = {}  # wellbore_id -> (header_id, bh_md)

    with open(edm_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("<CD_WELLBORE ") and "wellbore_name=" in line:
                m = re.search(r'wellbore_id="([^"]+)".*wellbore_name="([^"]+)"', line)
                if m:
                    wellbores[m.group(1)] = m.group(2)
            elif line.startswith("<CD_DEFINITIVE_SURVEY_HEADER"):
                hid_m = re.search(r'def_survey_header_id="([^"]+)"', line)
                wb_m = re.search(r'wellbore_id="([^"]+)"', line)
                phase_m = re.search(r'phase="([^"]+)"', line)
                isdef_m = re.search(r'is_definitive="([^"]+)"', line)
                bh_m = re.search(r'bh_md="([^"]+)"', line)
                if not (hid_m and wb_m and phase_m and isdef_m):
                    continue
                if phase_m.group(1) != "ACTUAL" or isdef_m.group(1) != "Y":
                    continue
                wb_id = wb_m.group(1)
                bh_md = float(bh_m.group(1)) if bh_m else 0.0
                prev = best_header.get(wb_id)
                if prev is None or bh_md > prev[1]:
                    best_header[wb_id] = (hid_m.group(1), bh_md)

    header_to_well = {
        hid: normalize_well_name(wellbores[wb_id])
        for wb_id, (hid, _) in best_header.items()
        if wb_id in wellbores
    }
    if not header_to_well:
        return {}

    surveys: Dict[str, List[SurveyPoint]] = {}
    with open(edm_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("<CD_DEFINITIVE_SURVEY_STATION"):
                continue
            hid_m = re.search(r'def_survey_header_id="([^"]+)"', line)
            if not hid_m or hid_m.group(1) not in header_to_well:
                continue
            well = header_to_well[hid_m.group(1)]

            def _attr(attr: str) -> Optional[float]:
                m = re.search(rf'{attr}="([^"]+)"', line)
                if not m or m.group(1) in ("", "null"):
                    return None
                try:
                    return float(m.group(1))
                except ValueError:
                    return None

            md, inc = _attr("md"), _attr("inclination")
            if md is None or inc is None:
                continue
            tvd = _attr("tvd")
            ns, ew = _attr("offset_north"), _attr("offset_east")
            surveys.setdefault(well, []).append((
                md * FT_TO_M,
                inc,
                _attr("azimuth") or 0.0,
                (tvd if tvd is not None else md) * FT_TO_M,
                (ns or 0.0) * FT_TO_M,
                (ew or 0.0) * FT_TO_M,
            ))

    return {well: sorted(pts, key=lambda p: p[0]) for well, pts in surveys.items()}


def _survey_stats(survey: List[SurveyPoint]) -> Dict:
    if not survey:
        return {"points": 0, "max_inclination": 0.0, "md_range": [0.0, 0.0]}
    return {
        "points": len(survey),
        "max_inclination": max(p[1] for p in survey),
        "md_range": [survey[0][0], survey[-1][0]],
    }


def _pick_best_survey(
    witsml: Optional[List[SurveyPoint]],
    edm: Optional[List[SurveyPoint]],
) -> Tuple[Optional[List[SurveyPoint]], Optional[str]]:
    """Prefer EDM definitive survey when available; otherwise WITSML."""
    if edm and len(edm) >= 10:
        return edm, "edm_definitive"
    if witsml and len(witsml) >= 10:
        return witsml, "witsml"
    if edm:
        return edm, "edm_definitive"
    if witsml:
        return witsml, "witsml"
    return None, None


def write_survey_csv(survey: List[SurveyPoint], out_path: Path, source: str):
    """Write full MD,INCL,AZI,TVD,NS,EW CSV from parsed survey stations."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MD", "INCL", "AZI", "TVD", "NS", "EW", "SOURCE"])
        for md, inc, azi, tvd, ns, ew in survey:
            if tvd is None:
                tvd = md * np.cos(np.radians(inc))
            if ns is None:
                ns = md * np.sin(np.radians(inc))
            if ew is None:
                ew = 0.0
            w.writerow([
                f"{md:.2f}", f"{inc:.2f}", f"{azi:.2f}",
                f"{tvd:.2f}", f"{ns:.2f}", f"{ew:.2f}", source,
            ])


def collect_witsml_well_info(witsml_dir: Path) -> Dict[str, Dict]:
    """Parse WITSML _wellInfo XML for lat/lon, KB, water depth."""
    if not witsml_dir.exists():
        return {}

    info: Dict[str, Dict] = {}
    for path in witsml_dir.glob("*/*_wellInfo/*.xml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        name_m = re.search(r"<well[^>]*>.*?<name>([^<]+)</name>", text, re.DOTALL)
        if not name_m:
            continue
        well = normalize_well_name(name_m.group(1))

        def _f(tag: str) -> Optional[float]:
            m = re.search(rf"<{tag} uom=\"[^\"]*\">([^<]+)</{tag}>", text)
            if not m:
                return None
            try:
                return float(m.group(1))
            except ValueError:
                return None

        entry: Dict = {"coord_source": "witsml_wellinfo", "wellinfo_file": str(path.name)}
        lat, lon = _f("latitude"), _f("longitude")
        kb, wd = _f("wellheadElevation"), _f("waterDepth")
        if _valid_latlon(lat, lon):
            entry["latitude"] = lat
            entry["longitude"] = lon
        if kb is not None:
            entry["kb_elevation_m"] = kb
            entry["kb_source"] = "witsml_wellinfo"
        if wd is not None:
            entry["water_depth_m"] = wd
            entry["ground_elevation_m"] = -wd

        prev = info.get(well)
        if prev is None or len(entry) > len(prev):
            info[well] = entry
    return info


def collect_edm_well_info(edm_path: Path) -> Dict[str, Dict]:
    """Parse EDM CD_WELL for lat/lon and water depth (fallback for coordinates)."""
    if not edm_path.exists():
        return {}

    info: Dict[str, Dict] = {}
    with open(edm_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("<CD_WELL ") or "well_legal_name=" not in line:
                continue
            legal_m = re.search(r'well_legal_name="([^"]+)"', line)
            if not legal_m:
                continue
            well = normalize_well_name(legal_m.group(1))
            entry: Dict = {"coord_source": "edm_cd_well"}

            lat_m = re.search(r'geo_latitude="([^"]+)"', line)
            lon_m = re.search(r'geo_longitude="([^"]+)"', line)
            wd_m = re.search(r'water_depth="([^"]+)"', line)
            if lat_m:
                entry["latitude"] = float(lat_m.group(1))
            if lon_m:
                entry["longitude"] = float(lon_m.group(1))
            if wd_m:
                wd = float(wd_m.group(1)) * FT_TO_M
                entry["water_depth_m"] = wd
                entry["ground_elevation_m"] = -wd

            info[well] = entry
    return info


def _valid_latlon(lat: Optional[float], lon: Optional[float]) -> bool:
    return (
        lat is not None
        and lon is not None
        and abs(lat) > 10.0
        and abs(lon) > 0.01
    )


def _parent_well_name(well_name: str) -> Optional[str]:
    m = re.match(r"^(15_9-F-\d+)", well_name)
    return m.group(1) if m else None


def merge_surface_info(
    meta: Dict,
    well_name: str,
    witsml_info: Dict[str, Dict],
    edm_info: Dict[str, Dict],
) -> Dict:
    """Overlay WITSML/EDM surface location and KB onto LAS-derived metadata."""
    wi = witsml_info.get(well_name, {})
    ei = edm_info.get(well_name, {})
    parent = _parent_well_name(well_name)
    wi_parent = witsml_info.get(parent, {}) if parent else {}
    ei_parent = edm_info.get(parent, {}) if parent else {}

    for src in (wi, ei, wi_parent, ei_parent):
        if _valid_latlon(src.get("latitude"), src.get("longitude")):
            meta["latitude"] = src["latitude"]
            meta["longitude"] = src["longitude"]
            meta["coord_source"] = src.get("coord_source", "merged")
            break

    for src in (wi, wi_parent):
        if src.get("kb_elevation_m") is not None:
            meta["kb_elevation_m"] = src["kb_elevation_m"]
            meta["kb_source"] = src.get("kb_source", "witsml_wellinfo")
            break

    if meta.get("kb_elevation_m") is None and meta.get("strt") is not None:
        strt = meta["strt"]
        if 140.0 <= strt <= 200.0:
            meta["kb_elevation_m"] = 54.9
            meta["kb_source"] = "inferred_volve_template"
        elif well_name.startswith("15_9-19"):
            meta["kb_elevation_m"] = 25.0
            meta["kb_source"] = "inferred_19_series"

    for src in (wi, ei, wi_parent, ei_parent):
        if src.get("water_depth_m") is not None:
            meta["water_depth_m"] = src["water_depth_m"]
            break

    for src in (wi, ei, wi_parent, ei_parent):
        if src.get("ground_elevation_m") is not None:
            meta["ground_elevation_m"] = src["ground_elevation_m"]
            break

    if meta.get("ground_elevation_m") is None:
        if meta.get("water_depth_m") is not None:
            meta["ground_elevation_m"] = -meta["water_depth_m"]
        elif meta.get("kb_elevation_m") == 54.9:
            meta["ground_elevation_m"] = -91.0

    if meta.get("kb_elevation_m") is None and meta.get("water_depth_m") is not None:
        meta["kb_elevation_m"] = 54.9
        meta["kb_source"] = "inferred_volve_template"

    return meta


def try_download_sodir(well_name: str, wellbore_id: int, out_path: Path) -> bool:
    """Try NPD/SODIR official deviation survey download."""
    try:
        import requests
    except ImportError:
        return False

    urls = [
        f"https://factpages.sodir.no/ReportServer?/FactPages/geometries/geometries&wellboreId={wellbore_id}&rs:Format=CSV",
        f"https://factpages.npd.no/ReportServer?/FactPages/geometries/geometries&wellboreId={wellbore_id}&rs:Format=CSV",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200 and len(resp.text) > 200 and "MD" in resp.text.upper():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                text = resp.text
                if "SOURCE" not in text.split("\n")[0]:
                    lines = text.strip().split("\n")
                    header = lines[0].strip() + ",SOURCE"
                    body = [header]
                    for line in lines[1:]:
                        if line.strip():
                            body.append(line.strip() + ",sodir")
                    text = "\n".join(body) + "\n"
                out_path.write_text(text, encoding="utf-8")
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------

def prepare_volve_data(
    project_root: Optional[Path] = None,
    try_sodir: bool = True,
    backup_synthetic: bool = True,
) -> Dict:
    paths = resolve_project_paths(project_root)
    root = paths["project_root"]
    well_root = paths["well_dir"]
    prepared = paths["prepared_dir"]
    dev_out = paths["deviations_dir"]
    prepared.mkdir(parents=True, exist_ok=True)
    dev_out.mkdir(parents=True, exist_ok=True)

    # Backup old synthetic deviations
    legacy = paths["legacy_deviations_dir"]
    if backup_synthetic and legacy.exists():
        backup = root / "data" / "volve_deviations_synthetic_backup"
        if not backup.exists():
            shutil.copytree(legacy, backup)
            print(f"Backed up legacy deviations -> {backup}")

    wells = discover_wells(well_root)
    well_metadata = {}
    curve_inventory = {}
    deviation_info = {}

    print(f"Project root: {root}")
    print(f"Seismic: {paths['segy_path']}")
    print(f"Wells dir: {well_root}")
    print(f"Discovered {len(wells)} wells with usable LAS\n")

    # Pre-scan real trajectory and surface location sources
    witsml_surveys = collect_witsml_trajectories(paths["witsml_dir"])
    edm_surveys = collect_edm_trajectories(paths["edm_path"])
    witsml_info = collect_witsml_well_info(paths["witsml_dir"])
    edm_info = collect_edm_well_info(paths["edm_path"])
    if witsml_surveys:
        print(f"WITSML trajectories: {len(witsml_surveys)} wells")
    if edm_surveys:
        print(f"EDM definitive surveys: {len(edm_surveys)} wells")
    if witsml_info:
        print(f"WITSML wellInfo: {len(witsml_info)} wells (lat/lon/KB)")
    if edm_info:
        print(f"EDM well headers: {len(edm_info)} wells")
    if witsml_surveys or edm_surveys or witsml_info:
        print()

    for well_name in wells:
        well_dir = well_root / well_name
        las_path = find_best_las_for_well(well_dir)
        if las_path is None:
            continue

        score, std_curves, header = score_las_file(las_path)
        meta = {
            "well_name": well_name,
            "best_las": str(las_path.relative_to(root)),
            "standard_curves": std_curves,
            "n_standard_curves": len(std_curves),
            "las_score": score,
            **{k: v for k, v in header.items() if k != "header_raw"},
        }
        meta = merge_surface_info(meta, well_name, witsml_info, edm_info)
        well_metadata[well_name] = meta
        curve_inventory[well_name] = {
            "curves": std_curves,
            "missing": [c for c in STANDARD_CURVES if c not in std_curves],
            "las_file": las_path.name,
        }

        # Deviation: EDM/WITSML real survey first, then SODIR, then LWD
        csv_path = dev_out / f"{well_name}.csv"
        dev_source = None

        survey, survey_src = _pick_best_survey(
            witsml_surveys.get(well_name),
            edm_surveys.get(well_name),
        )
        if survey is not None:
            write_survey_csv(survey, csv_path, survey_src)
            stats = _survey_stats(survey)
            deviation_info[well_name] = {
                "source": survey_src,
                **stats,
                "has_azimuth": any(p[2] != 0 for p in survey),
                "has_tvd_ns_ew": any(p[3] is not None for p in survey),
            }
            dev_source = survey_src

        if dev_source is None and try_sodir and well_name in SODIR_WELLBORE_IDS:
            ok = try_download_sodir(well_name, SODIR_WELLBORE_IDS[well_name], csv_path)
            if ok:
                dev_source = "sodir"
                n_rows = len(csv_path.read_text().strip().split("\n")) - 1
                deviation_info[well_name] = {"source": "sodir", "points": n_rows}

        if dev_source is None:
            lwd = extract_lwd_deviation(well_dir)
            if lwd is not None:
                write_deviation_csv(lwd["survey"], csv_path, "lwd_mwd")
                deviation_info[well_name] = {
                    "source": "lwd_mwd",
                    "points": lwd["points"],
                    "max_inclination": lwd["max_inclination"],
                    "md_range": lwd["md_range"],
                    "file": lwd.get("source_file"),
                    "note": "AZI=0 (MWD logs lack azimuth); inclination is real",
                }
                dev_source = "lwd_mwd"

        if dev_source is None:
            deviation_info[well_name] = {"source": "none", "note": "vertical assumed"}

        curves_str = ",".join(std_curves) if std_curves else "none"
        kb = meta.get("kb_elevation_m", "?")
        print(f"  {well_name:18s} curves={len(std_curves)}/9  KB={kb}m  dev={deviation_info[well_name]['source']}")

    annotate_geometry_quality(well_metadata, deviation_info)
    n_verified = sum(1 for m in well_metadata.values() if m.get("geometry_verified"))

    # Write outputs
    meta_path = prepared / "well_metadata.json"
    inv_path = prepared / "curve_inventory.json"
    dev_path = prepared / "deviation_inventory.json"
    layout_path = prepared / "data_layout.json"

    layout = {
        "project_root": str(root),
        "segy_path": str(paths["segy_path"]) if paths["segy_path"] else None,
        "well_dir": str(well_root),
        "prepared_dir": str(prepared),
        "deviations_dir": str(dev_out),
        "witsml_dir": str(paths["witsml_dir"]) if paths["witsml_dir"].exists() else None,
        "edm_path": str(paths["edm_path"]) if paths["edm_path"].exists() else None,
        "data_dir_for_training": str(root),
    }

    for p, obj in [
        (meta_path, well_metadata),
        (inv_path, curve_inventory),
        (dev_path, deviation_info),
        (layout_path, layout),
    ]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "n_wells": len(wells),
        "n_wells_with_7plus_curves": sum(1 for w in curve_inventory.values() if len(w["curves"]) >= 7),
        "n_wells_with_deviation": sum(1 for d in deviation_info.values() if d["source"] != "none"),
        "n_wells_geometry_verified": n_verified,
        "segy_available": paths["segy_path"] is not None,
        "paths": layout,
    }
    manifest_path = prepared / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nPrepared data written to {prepared}")
    print(f"  Wells: {manifest['n_wells']}")
    print(f"  >=7 curves: {manifest['n_wells_with_7plus_curves']}")
    print(f"  With deviation: {manifest['n_wells_with_deviation']}")

    return manifest


def load_prepared_metadata(project_root: Optional[Path] = None) -> Optional[Dict]:
    paths = resolve_project_paths(project_root)
    meta_path = paths["prepared_dir"] / "well_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_prepared_layout(project_root: Optional[Path] = None) -> Optional[Dict]:
    paths = resolve_project_paths(project_root)
    layout_path = paths["prepared_dir"] / "data_layout.json"
    if not layout_path.exists():
        return None
    with open(layout_path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    prepare_volve_data()
