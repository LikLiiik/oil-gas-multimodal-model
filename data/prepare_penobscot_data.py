"""
Prepare Penobscot 3D dataset (OpendTect project zip).

Extracts Penobscot.zip, parses survey/well metadata, copies LAS logs,
converts 3D post-stack CBVS -> SEG-Y with validation.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .cbvs_io import cbvs_to_segy, parse_penobscot_survey
from .prepare_volve_data import STANDARD_CURVES, annotate_geometry_quality, score_las_file

# Well surface coords from Penobscot .survey (authoritative, NAD27 UTM 20N)
PENO_WELLS = {
    "B-41": {"latitude": 44 + 10 / 60 + 2.44 / 3600, "longitude": -(60 + 6 / 60 + 32.72 / 3600)},
    "L-30": {"latitude": 44 + 9 / 60 + 43.55 / 3600, "longitude": -(60 + 4 / 60 + 9.33 / 3600)},
}


def extract_penobscot_zip(zip_path: Path, dest_dir: Path, force: bool = False) -> Path:
    proj = dest_dir / "Penobscot"
    if proj.exists() and not force:
        print(f"Penobscot already extracted at {proj}")
        return proj
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {dest_dir} (this may take several minutes) ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Extraction complete.")
    return proj


def parse_well_file(well_path: Path) -> Dict:
    text = well_path.read_text(encoding="utf-8", errors="replace")
    out: Dict = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                x, y, kb = float(parts[0]), float(parts[1]), float(parts[3])
                if x > 100000 and y > 1_000_000:
                    out["surface_x"] = x
                    out["surface_y"] = y
                    out["kb_elevation_m"] = kb
            except ValueError:
                pass
    name = well_path.stem
    if name in PENO_WELLS:
        out.update(PENO_WELLS[name])
    out["coord_source"] = "penobscot_survey"
    out["well_name"] = name
    return out


def copy_horizons(proj_dir: Path, horizons_dir: Path) -> List[str]:
    src = proj_dir / "Surfaces"
    horizons_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(src.glob("*.hor")):
        if "^" in p.name:
            continue
        dst = horizons_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        copied.append(p.stem)
    return copied


def write_vertical_deviation(out_path: Path, td_m: float = 4000.0, step: float = 10.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MD", "INCL", "AZI", "TVD", "NS", "EW", "SOURCE"])
        md = 0.0
        while md <= td_m:
            w.writerow([f"{md:.2f}", "0.00", "0.00", f"{md:.2f}", "0.00", "0.00", "vertical_assumed"])
            md += step


def prepare_penobscot_data(
    project_root: Optional[Path] = None,
    zip_path: Optional[Path] = None,
    force_extract: bool = False,
    skip_segy: bool = False,
) -> Dict:
    root = Path(project_root or Path(__file__).resolve().parents[1])
    zip_path = Path(zip_path or root / "data" / "Penobscot.zip")
    raw_base = root / "data" / "penobscot" / "raw"
    prepared = root / "data" / "penobscot" / "prepared"
    seismic_dir = root / "data" / "penobscot" / "seismic"
    wells_dir = root / "data" / "penobscot" / "wells"
    dev_dir = prepared / "deviations"
    prepared.mkdir(parents=True, exist_ok=True)

    proj_dir = extract_penobscot_zip(zip_path, raw_base, force=force_extract)
    survey_text = (proj_dir / ".survey").read_text(encoding="utf-8", errors="replace")
    spec = parse_penobscot_survey(survey_text)

    well_metadata, curve_inventory, deviation_info = {}, {}, {}
    las_dir = proj_dir / "Rawdata" / "WellLogs"
    for las_path in sorted(las_dir.glob("*.las")):
        m = re.search(r"\b(B-\d+|L-\d+)\b", las_path.stem, re.I)
        if not m:
            continue
        wname = m.group(1).upper()

        score, std_curves, las_hdr = score_las_file(las_path)
        well_file = proj_dir / "WellInfo" / f"{wname}.well"
        wi = parse_well_file(well_file) if well_file.exists() else {}

        dst_las = wells_dir / "las" / f"{wname}.las"
        dst_las.parent.mkdir(parents=True, exist_ok=True)
        if not dst_las.exists():
            shutil.copy2(las_path, dst_las)

        well_metadata[wname] = {
            "well_name": wname,
            "las_path": str(dst_las),
            "standard_curves": std_curves,
            "n_standard_curves": len(std_curves),
            "las_score": score,
            "latitude": wi.get("latitude") or las_hdr.get("latitude"),
            "longitude": wi.get("longitude") or las_hdr.get("longitude"),
            "surface_x": wi.get("surface_x"),
            "surface_y": wi.get("surface_y"),
            "kb_elevation_m": wi.get("kb_elevation_m") or las_hdr.get("kb_elevation_m"),
            "coord_source": wi.get("coord_source", "penobscot_survey"),
            "dataset": "penobscot",
            "crs": "NAD27 UTM 20N",
        }
        curve_inventory[wname] = {
            "curves": std_curves,
            "missing": [c for c in STANDARD_CURVES if c not in std_curves],
            "las_file": las_path.name,
        }
        csv_path = dev_dir / f"{wname}.csv"
        write_vertical_deviation(csv_path)
        deviation_info[wname] = {
            "source": "penobscot_survey",
            "points": int(4000 / 10) + 1,
            "note": "Vertical trajectory; surface XY/lat-lon from OpendTect survey",
        }

    annotate_geometry_quality(well_metadata, deviation_info)
    for wname, meta in well_metadata.items():
        if meta.get("latitude") and meta.get("longitude") and meta.get("surface_x"):
            meta["geometry_verified"] = True
            meta.pop("geometry_excluded_reason", None)

    horizons = copy_horizons(proj_dir, prepared / "horizons")

    layout = {
        "dataset": "penobscot",
        "raw_dir": str(proj_dir),
        "prepared_dir": str(prepared),
        "seismic_dir": str(seismic_dir),
        "segy_3d": None,
        "survey": {
            "inline": [spec.inl_start, spec.inl_stop, spec.inl_step],
            "crossline": [spec.crl_start, spec.crl_stop, spec.crl_step],
            "z": [spec.z_start, spec.z_stop, spec.z_step],
        },
        "horizons": horizons,
    }

    # Save metadata before slow SEG-Y conversion
    for p, obj in [
        (prepared / "well_metadata.json", well_metadata),
        (prepared / "curve_inventory.json", curve_inventory),
        (prepared / "deviation_inventory.json", deviation_info),
        (prepared / "data_layout.json", layout),
        (prepared / "survey.json", {"text_excerpt": survey_text[:2000], "spec": layout["survey"]}),
    ]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    cbvs_meta = {}
    segy_path = seismic_dir / "pstm_stack_agc.segy"
    if not skip_segy and not segy_path.exists():
        cbvs_src = proj_dir / "Seismics" / "1-PSTM_stack_agc.cbvs"
        if cbvs_src.exists():
            print(f"Converting CBVS -> SEG-Y ({spec.n_inlines}x{spec.n_crosslines}x{spec.n_samples}) ...")
            cbvs_meta = cbvs_to_segy(cbvs_src, segy_path, spec, scale=1.0, validate=True)
            print(f"SEG-Y written: {segy_path}")
        else:
            raise FileNotFoundError(f"Missing stack CBVS: {cbvs_src}")
    elif segy_path.exists():
        print(f"SEG-Y already exists: {segy_path}")

    layout["segy_3d"] = str(segy_path) if segy_path.exists() else None
    with open(prepared / "data_layout.json", "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "penobscot",
        "n_wells": len(well_metadata),
        "n_geometry_verified": sum(1 for m in well_metadata.values() if m.get("geometry_verified")),
        "horizons": horizons,
        "cbvs_conversion": cbvs_meta,
        "paths": layout,
    }
    with open(prepared / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Penobscot: {len(well_metadata)} wells, horizons={len(horizons)}, segy={segy_path.exists()}")
    return manifest


if __name__ == "__main__":
    prepare_penobscot_data()
