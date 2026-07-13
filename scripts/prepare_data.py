#!/usr/bin/env python3
"""
Prepare Volve dataset from locally available files.

Run once before training:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --project_root /path/to/oil-gas-multimodal-model
"""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import without loading data/__init__.py (avoids torch dependency)
_spec = importlib.util.spec_from_file_location(
    "prepare_volve_data",
    PROJECT_ROOT / "data" / "prepare_volve_data.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
prepare_volve_data = _mod.prepare_volve_data


def main():
    parser = argparse.ArgumentParser(description="Prepare Volve dataset metadata")
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Project root (contains seismic/ and data/)",
    )
    parser.add_argument(
        "--no-sodir",
        action="store_true",
        help="Skip NPD/SODIR deviation download, use LWD only",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup legacy synthetic deviation CSVs",
    )
    args = parser.parse_args()

    manifest = prepare_volve_data(
        project_root=Path(args.project_root),
        try_sodir=not args.no_sodir,
        backup_synthetic=not args.no_backup,
    )

    print("\n=== Summary ===")
    print(f"Wells ready: {manifest['n_wells']}")
    print(f"With >=7 curves: {manifest['n_wells_with_7plus_curves']}")
    print(f"With deviation: {manifest['n_wells_with_deviation']}")
    print(f"Output: {manifest['paths']['prepared_dir']}")


if __name__ == "__main__":
    main()
