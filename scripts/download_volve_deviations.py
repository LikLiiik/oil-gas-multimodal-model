"""
Download Volve Well Deviation Surveys from NPD/SODIR FactPages

NPD wellbore IDs for all Volve wells. Run this script to download
deviation survey CSV files for each well.

Usage:
    python scripts/download_volve_deviations.py

Output:
    data/volve_deviations/{well_name}.csv  —  MD, Inclination, Azimuth, TVD, X, Y

Reference URLs (open in browser if download fails):
    https://factpages.sodir.no/ReportServer?/FactPages/geometries/geometries&wellboreId=5893&rs:Format=CSV
"""

import os
import sys
import time
from pathlib import Path

# Wellbore IDs for Volve field wells (from NPD/SODIR FactPages)
VOLVE_WELLBORE_IDS = {
    "15_9-19 A":   3372,
    "15_9-19 B":   3373,   # sidetrack
    "15_9-19 S":   3374,   # sidetrack
    "15_9-F-1":    3804,   # discovery well
    "15_9-F-1 A":  5893,   # sidetrack
    "15_9-F-1 B":  5894,   # sidetrack
    "15_9-F-1 C":  5895,   # sidetrack
    "15_9-F-4":    5516,
    "15_9-F-5":    5517,
    "15_9-F-7":    5518,
    "15_9-F-9":    5948,
    "15_9-F-9 A":  7117,   # sidetrack
    "15_9-F-10":   6203,
    "15_9-F-11":   6204,
    "15_9-F-11 A": 7083,   # sidetrack
    "15_9-F-11 B": 7118,   # sidetrack
    "15_9-F-11 T2": 7410,  # sidetrack
    "15_9-F-12":   6303,
    "15_9-F-14":   6698,
    "15_9-F-15":   6699,
    "15_9-F-15 A": 7084,   # sidetrack
    "15_9-F-15 B": 7119,   # sidetrack
    "15_9-F-15 C": 7276,   # sidetrack
    "15_9-F-15 D": 7580,   # sidetrack
}

def download_with_requests(well_name, wellbore_id, output_dir):
    """Download deviation survey using requests library."""
    import requests

    url = (
        "https://factpages.sodir.no/ReportServer"
        f"?/FactPages/geometries/geometries"
        f"&wellboreId={wellbore_id}"
        f"&rs:Format=CSV"
    )

    print(f"  Downloading {well_name} (ID={wellbore_id})...", end=" ", flush=True)

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.text) > 200:
            output_path = output_dir / f"{well_name}.csv"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            # Parse to count rows
            lines = resp.text.strip().split("\n")
            n_rows = len(lines) - 1  # minus header
            print(f"OK ({n_rows} survey points)")
            return True
        else:
            print(f"FAIL (HTTP {resp.status_code}, {len(resp.text)} bytes)")
            return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def download_with_curl(well_name, wellbore_id, output_dir):
    """Fallback: download using curl."""
    import subprocess

    url = (
        "https://factpages.npd.no/ReportServer"
        f"?/FactPages/geometries/geometries"
        f"&wellboreId={wellbore_id}"
        f"&rs:Format=CSV"
    )

    output_path = output_dir / f"{well_name}.csv"

    print(f"  Trying curl for {well_name} (ID={wellbore_id})...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ["curl", "-s", "-L", "--connect-timeout", "15", "-o", str(output_path), url],
            capture_output=True, text=True, timeout=20,
        )
        if output_path.exists() and output_path.stat().st_size > 200:
            with open(output_path) as f:
                n = len(f.readlines()) - 1
            print(f"OK ({n} survey points)")
            return True
        else:
            print("FAIL (empty or small file)")
            return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def create_manual_fallback(output_dir):
    """
    Create approximate deviation survey files for known Volve wells.

    For wells where we can't download, use a near-vertical approximation.
    Volve wells are typically deviated <5° in the reservoir section,
    with sidetracks kicking off from the main bore at ~2000-3000m MD.

    These are APPROXIMATE and should be replaced with real data.
    """
    import csv

    # For wells where we know the approximate kick-off:
    # 15/9-F-1: near-vertical discovery well
    # 15/9-F-1 A/B/C: sidetracks kicking off at ~3200m
    # 15/9-F-11: deviated producer, kicks at ~2200m
    # 15/9-F-11 A/B/T2: sidetracks

    well_configs = {
        "15_9-F-1": {
            "md_range": (0, 3800),
            "step": 50,
            "max_inc": 3.0,    # near vertical
            "max_azi": 135.0,
            "kick_md": 3500,
        },
        "15_9-F-1 A": {
            "md_range": (0, 4200),
            "step": 50,
            "max_inc": 45.0,   # high angle sidetrack
            "max_azi": 135.0,
            "kick_md": 3200,
        },
        "15_9-F-1 B": {
            "md_range": (0, 4000),
            "step": 50,
            "max_inc": 40.0,
            "max_azi": 225.0,
            "kick_md": 3200,
        },
        "15_9-F-1 C": {
            "md_range": (0, 4500),
            "step": 50,
            "max_inc": 50.0,
            "max_azi": 135.0,
            "kick_md": 3200,
        },
        "15_9-F-11": {
            "md_range": (0, 3800),
            "step": 50,
            "max_inc": 60.0,   # highly deviated producer
            "max_azi": 315.0,
            "kick_md": 2200,
        },
        "15_9-F-11 A": {
            "md_range": (0, 4200),
            "step": 50,
            "max_inc": 55.0,
            "max_azi": 315.0,
            "kick_md": 2800,
        },
        "15_9-F-11 B": {
            "md_range": (0, 5000),
            "step": 50,
            "max_inc": 50.0,
            "max_azi": 225.0,
            "kick_md": 2800,
        },
        "15_9-F-11 T2": {
            "md_range": (0, 4800),
            "step": 50,
            "max_inc": 45.0,
            "max_azi": 45.0,
            "kick_md": 2800,
        },
        "15_9-F-15 D": {
            "md_range": (0, 5000),
            "step": 50,
            "max_inc": 65.0,
            "max_azi": 180.0,
            "kick_md": 2400,
        },
    }

    import math

    for well_name, cfg in well_configs.items():
        output_path = output_dir / f"{well_name}.csv"
        if output_path.exists():
            continue  # Don't overwrite real downloads

        print(f"  Creating approximate deviation for {well_name}...")

        md_start, md_end = cfg["md_range"]
        step = cfg["step"]
        n = (md_end - md_start) // step + 1

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["MD", "INCL", "AZI", "TVD", "NS", "EW", "DLS"])

            for i in range(n):
                md = md_start + i * step
                if md <= cfg["kick_md"]:
                    inc = 0.0
                    azi = 0.0
                else:
                    frac = min(1.0, (md - cfg["kick_md"]) / 500.0)
                    inc = cfg["max_inc"] * frac
                    azi = cfg["max_azi"]

                # Approx TVD (simple integration)
                tvd = md_start  # simplified
                if md <= cfg["kick_md"]:
                    tvd = md
                else:
                    tvd = cfg["kick_md"] + (md - cfg["kick_md"]) * math.cos(math.radians(inc))

                writer.writerow([
                    f"{md:.2f}",
                    f"{inc:.2f}",
                    f"{azi:.2f}",
                    f"{tvd:.2f}",
                    "0.00",
                    "0.00",
                    "0.00",
                ])


def main():
    output_dir = Path(__file__).parent.parent / "data" / "volve_deviations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Downloading {len(VOLVE_WELLBORE_IDS)} wells...\n")

    # Try requests first, then curl
    downloaded = 0
    for well_name, wid in sorted(VOLVE_WELLBORE_IDS.items()):
        # Skip if already exists
        if (output_dir / f"{well_name}.csv").exists():
            print(f"  {well_name} (ID={wid}): already downloaded, skipping")
            downloaded += 1
            continue

        success = download_with_requests(well_name, wid, output_dir)
        if not success:
            success = download_with_curl(well_name, wid, output_dir)

        if success:
            downloaded += 1
        time.sleep(0.5)  # Rate limit

    print(f"\nDownloaded: {downed}/{len(VOLVE_WELLBORE_IDS)}")

    # If nothing downloaded, create approximate fallback data
    if downloaded == 0:
        print("\nNo real data downloaded. Creating approximate fallback data...")
        create_manual_fallback(output_dir)
        print("WARNING: These are APPROXIMATE trajectories! Replace with real data when possible.")

    print("\nDone!")


if __name__ == "__main__":
    main()
