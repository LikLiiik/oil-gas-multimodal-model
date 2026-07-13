#!/usr/bin/env python3
"""Prepare RMOTC and/or Penobscot external datasets."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Prepare external oil/gas datasets")
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--dataset", choices=["rmotc", "penobscot", "all"], default="all")
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--skip_penobscot_segy", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root)

    if args.dataset in ("rmotc", "all"):
        from data.prepare_rmotc_data import prepare_rmotc_data
        print("=" * 60)
        print("Preparing RMOTC (Teapot Dome)")
        print("=" * 60)
        prepare_rmotc_data(project_root=root, force_extract=args.force_extract)

    if args.dataset in ("penobscot", "all"):
        from data.prepare_penobscot_data import prepare_penobscot_data
        print("=" * 60)
        print("Preparing Penobscot 3D")
        print("=" * 60)
        prepare_penobscot_data(
            project_root=root,
            force_extract=args.force_extract,
            skip_segy=args.skip_penobscot_segy,
        )


if __name__ == "__main__":
    main()
