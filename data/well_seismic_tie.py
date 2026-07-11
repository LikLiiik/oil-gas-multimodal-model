"""
Well-Seismic Tie Module — Physical Alignment of Seismic Patches with Well Trajectories

Ensures seismic patches are extracted at the ACTUAL well position (not random),
using real coordinates, well trajectories, and elevation datums.

Key components:
1. Coordinate conversion: Lat/Lon → UTM → Inline/Crossline
2. Well trajectory: MD → TVD → X,Y at each measured depth
3. Elevation datums: KB (Kelly Bushing), Ground, Sea Level corrections
4. Seismic trace extraction along well trajectory
5. Time-depth relationship (VSP/checkshot)

References:
- Volve field is at UTM zone 31N, ~435,000E, ~6,474,500N
- Bin size approx 12.5m × 12.5m
- Water depth ~80-90m, reservoir ~3000-3500m TVDSS
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
import math


# ==============================================================================
# Coordinate Conversion
# ==============================================================================

def dms_to_decimal(deg: float, min: float, sec: float, hemisphere: str) -> float:
    """Convert Degrees-Minutes-Seconds to decimal degrees."""
    decimal = deg + min / 60.0 + sec / 3600.0
    if hemisphere.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def latlon_to_utm(lat: float, lon: float) -> Tuple[int, float, float]:
    """
    Convert WGS84 lat/lon to UTM coordinates.

    Approximate conversion using transverse Mercator formulas.
    Accurate to ~1m for the Volve field area.

    Returns:
        zone, easting, northing
    """
    # WGS84 constants
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    k0 = 0.9996  # scale factor

    # UTM zone
    zone = int((lon + 180) / 6) + 1

    # Central meridian
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Meridional arc
    e2 = 2 * f - f ** 2
    e4 = e2 ** 2
    e6 = e2 ** 3

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = e2 / (1 - e2) * math.cos(lat_rad) ** 2
    A = (lon_rad - lon0) * math.cos(lat_rad)

    M = a * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * lat_rad)
        + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * lat_rad)
        - (35 * e6 / 3072) * math.sin(6 * lat_rad)
    )

    easting = k0 * N * (
        A
        + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * e2 / (1 - e2)) * A ** 5 / 120
    ) + 500000.0

    northing = k0 * (
        M
        + N * math.tan(lat_rad) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * e2 / (1 - e2))
            * A ** 6 / 720
        )
    )

    if lat < 0:
        northing += 10000000.0

    return zone, easting, northing


# ==============================================================================
# Inline/Crossline ↔ UTM Mapping
# ==============================================================================

class SeismicSurveyGeometry:
    """
    Maps between UTM coordinates and seismic inline/crossline grid.

    The Volve seismic survey is a regular 3D grid with approximately
    12.5m × 12.5m bin spacing. Without the navigation file, we use
    an estimated mapping calibrated to known well positions.

    Args:
        il_min, il_max: Inline number range
        xl_min, xl_max: Crossline number range
        il_spacing: Inline spacing in meters
        xl_spacing: Crossline spacing in meters
        utm_origin_e: UTM Easting at (il=il_ref, xl=xl_ref)
        utm_origin_n: UTM Northing at (il=il_ref, xl=xl_ref)
        il_ref: Reference inline for origin
        xl_ref: Reference crossline for origin
        rotation_deg: Grid rotation from North (degrees, clockwise)
    """

    def __init__(
        self,
        il_min: int = 9961,
        il_max: int = 10361,
        xl_min: int = 1961,
        xl_max: int = 2680,
        il_spacing: float = 12.5,
        xl_spacing: float = 12.5,
        # Estimated Volve survey origin (calibrated to well 15/9-F-1)
        utm_origin_e: float = 434880.0,
        utm_origin_n: float = 6474400.0,
        il_ref: int = 10000,
        xl_ref: int = 2320,
        rotation_deg: float = -1.5,  # slight rotation typical for NCS
    ):
        self.il_min = il_min
        self.il_max = il_max
        self.xl_min = xl_min
        self.xl_max = xl_max
        self.il_spacing = il_spacing
        self.xl_spacing = xl_spacing
        self.utm_origin_e = utm_origin_e
        self.utm_origin_n = utm_origin_n
        self.il_ref = il_ref
        self.xl_ref = xl_ref
        self.rotation_deg = rotation_deg
        self.rotation_rad = math.radians(rotation_deg)

    def utm_to_ilxl(
        self, easting: float, northing: float
    ) -> Tuple[float, float]:
        """
        Convert UTM coordinates to inline/crossline (fractional).

        Returns:
            (inline, crossline) as floats
        """
        # Translate to origin
        de = easting - self.utm_origin_e
        dn = northing - self.utm_origin_n

        # Rotate to grid coordinates
        cos_r = math.cos(-self.rotation_rad)
        sin_r = math.sin(-self.rotation_rad)
        grid_x = de * cos_r - dn * sin_r
        grid_y = de * sin_r + dn * cos_r

        # Convert to inline/crossline
        il = self.il_ref + grid_y / self.il_spacing
        xl = self.xl_ref + grid_x / self.xl_spacing

        return il, xl

    def ilxl_to_utm(
        self, il: float, xl: float
    ) -> Tuple[float, float]:
        """Convert inline/crossline to UTM coordinates."""
        # Grid relative to reference
        grid_y = (il - self.il_ref) * self.il_spacing
        grid_x = (xl - self.xl_ref) * self.xl_spacing

        # Rotate back
        cos_r = math.cos(self.rotation_rad)
        sin_r = math.sin(self.rotation_rad)
        de = grid_x * cos_r - grid_y * sin_r
        dn = grid_x * sin_r + grid_y * cos_r

        easting = self.utm_origin_e + de
        northing = self.utm_origin_n + dn

        return easting, northing

    def latlon_to_ilxl(
        self, lat: float, lon: float
    ) -> Tuple[float, float]:
        """Convert lat/lon directly to inline/crossline."""
        _, easting, northing = latlon_to_utm(lat, lon)
        return self.utm_to_ilxl(easting, northing)

    def get_patch_bounds(
        self,
        il_center: float,
        xl_center: float,
        patch_size_il: int = 32,
        patch_size_xl: int = 32,
    ) -> Tuple[int, int, int, int]:
        """
        Get integer inline/crossline bounds for a patch centered at (il, xl).

        Returns:
            il_start, il_end, xl_start, xl_end
        """
        il_start = int(il_center - patch_size_il // 2)
        il_end = il_start + patch_size_il
        xl_start = int(xl_center - patch_size_xl // 2)
        xl_end = xl_start + patch_size_xl

        # Clamp to survey bounds
        il_start = max(self.il_min, il_start)
        il_end = min(self.il_max + 1, il_end)
        xl_start = max(self.xl_min, xl_start)
        xl_end = min(self.xl_max + 1, xl_end)

        return il_start, il_end, xl_start, xl_end


# ==============================================================================
# Well Trajectory
# ==============================================================================

class WellTrajectory:
    """
    Represents a deviated well trajectory in 3D space.

    For Volve, most wells are highly deviated (up to 70°), so we need
    to compute (X, Y, TVD) at each measured depth (MD).

    If deviation survey data is not available, assumes a vertical well
    at the surface location.

    Args:
        surface_lat: Surface latitude (decimal degrees)
        surface_lon: Surface longitude (decimal degrees)
        kb_elevation: Kelly Bushing elevation above MSL (meters)
        ground_elevation: Ground elevation above MSL (meters)
        md: Measured depth array (meters)
        inclination: Inclination array (degrees from vertical)
        azimuth: Azimuth array (degrees from North)
    """

    def __init__(
        self,
        well_name: str,
        surface_lat: float,
        surface_lon: float,
        kb_elevation: float = 0.0,
        ground_elevation: float = 0.0,
        # Deviation survey (None = vertical well)
        md_survey: Optional[np.ndarray] = None,
        inc_survey: Optional[np.ndarray] = None,
        az_survey: Optional[np.ndarray] = None,
    ):
        self.well_name = well_name
        self.surface_lat = surface_lat
        self.surface_lon = surface_lon
        self.kb_elevation = kb_elevation  # KB above MSL
        self.ground_elevation = ground_elevation  # Ground above MSL

        # Convert surface to UTM
        _, self.surface_e, self.surface_n = latlon_to_utm(surface_lat, surface_lon)

        # Deviation survey
        self.has_deviation = md_survey is not None and len(md_survey) > 0

        if self.has_deviation:
            self.md_survey = np.asarray(md_survey)
            self.inc_survey = np.asarray(inc_survey)
            self.az_survey = np.asarray(az_survey)
            # Compute trajectory (X, Y, TVD) at each MD
            self._compute_trajectory()
        else:
            self.md_survey = np.array([0.0])
            self.inc_survey = np.array([0.0])
            self.az_survey = np.array([0.0])

    def _compute_trajectory(self):
        """
        Compute (X_offset, Y_offset, TVD) at each survey point using
        minimum curvature method.
        """
        n = len(self.md_survey)
        x = np.zeros(n)
        y = np.zeros(n)
        tvd = np.zeros(n)

        inc_rad = np.radians(self.inc_survey)
        az_rad = np.radians(self.az_survey)

        for i in range(1, n):
            dmd = self.md_survey[i] - self.md_survey[i - 1]

            if dmd < 0.01:
                x[i] = x[i - 1]
                y[i] = y[i - 1]
                tvd[i] = tvd[i - 1]
                continue

            i1, i2 = inc_rad[i - 1], inc_rad[i]
            a1, a2 = az_rad[i - 1], az_rad[i]

            # Minimum curvature: use dogleg ratio
            cos_dogleg = math.cos(i2 - i1) - math.sin(i1) * math.sin(i2) * (1 - math.cos(a2 - a1))

            if abs(cos_dogleg - 1.0) < 1e-8:
                # Straight section
                rf = 1.0
            else:
                dogleg = math.acos(max(-1.0, min(1.0, cos_dogleg)))
                rf = 2.0 / dogleg * math.tan(dogleg / 2.0)

            delta_n = (dmd / 2.0) * (math.sin(i1) * math.cos(a1) + math.sin(i2) * math.cos(a2)) * rf
            delta_e = (dmd / 2.0) * (math.sin(i1) * math.sin(a1) + math.sin(i2) * math.sin(a2)) * rf
            delta_v = (dmd / 2.0) * (math.cos(i1) + math.cos(i2)) * rf

            x[i] = x[i - 1] + delta_e
            y[i] = y[i - 1] + delta_n
            tvd[i] = tvd[i - 1] + delta_v

        self.x_offset = x  # Easting offset from surface
        self.y_offset = y  # Northing offset from surface
        self.tvd = tvd  # True Vertical Depth below KB

    def get_position_at_md(self, md: float) -> Tuple[float, float, float]:
        """
        Get (easting, northing, tvd) at a given measured depth.

        Args:
            md: Measured depth in meters

        Returns:
            (easting, northing, tvd_subsea) in meters
        """
        if not self.has_deviation:
            # Vertical well
            tvdss = md - self.kb_elevation  # TVD subsea
            return self.surface_e, self.surface_n, tvdss

        # Interpolate in deviation survey
        idx = np.searchsorted(self.md_survey, md)
        if idx == 0:
            x, y = 0.0, 0.0
            tvd_kb = md  # assume vertical for extrapolation
        elif idx >= len(self.md_survey):
            x, y = self.x_offset[-1], self.y_offset[-1]
            tvd_kb = self.tvd[-1] + (md - self.md_survey[-1]) * math.cos(
                math.radians(self.inc_survey[-1])
            )
        else:
            # Linear interpolation
            frac = (md - self.md_survey[idx - 1]) / (
                self.md_survey[idx] - self.md_survey[idx - 1]
            )
            x = self.x_offset[idx - 1] + frac * (self.x_offset[idx] - self.x_offset[idx - 1])
            y = self.y_offset[idx - 1] + frac * (self.y_offset[idx] - self.y_offset[idx - 1])
            tvd_kb = self.tvd[idx - 1] + frac * (self.tvd[idx] - self.tvd[idx - 1])

        # TVD subsea
        tvdss = tvd_kb - self.kb_elevation

        easting = self.surface_e + x
        northing = self.surface_n + y

        return easting, northing, tvdss

    def load_deviation_csv(self, csv_path: str) -> bool:
        """
        Load deviation survey from a CSV file.

        Expected columns: MD, INCL(inclination), AZI(azimuth)
        Or: MD, INCL, AZI, TVD, NS, EW

        Returns True if loaded successfully.
        """
        try:
            import csv
            mds, incs, azis = [], [], []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try multiple column name formats
                    md = float(row.get("MD", row.get("md", row.get("MEASURED_DEPTH", 0))))
                    inc = float(row.get("INCL", row.get("INCLINATION", row.get("incl", row.get("INC", 0)))))
                    azi = float(row.get("AZI", row.get("AZIMUTH", row.get("azim", row.get("AZIM", 0)))))
                    mds.append(md)
                    incs.append(inc)
                    azis.append(azi)

            if len(mds) < 3:
                return False

            self.md_survey = np.array(mds)
            self.inc_survey = np.array(incs)
            self.az_survey = np.array(azis)
            self.has_deviation = True

            # Recompute trajectory
            self._compute_trajectory()
            return True
        except Exception:
            return False

    def get_positions_at_depths(
        self, md_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get (easting, northing, tvdss) for an array of measured depths.
        """
        n = len(md_array)
        e = np.zeros(n)
        n_ = np.zeros(n)
        t = np.zeros(n)
        for i, md in enumerate(md_array):
            e[i], n_[i], t[i] = self.get_position_at_md(md)
        return e, n_, t


# ==============================================================================
# Well-Seismic Data Extractor
# ==============================================================================

class WellSeismicDataExtractor:
    """
    Extract physically-aligned well-seismic training pairs.

    For each well and depth range:
    1. Compute the well trajectory position at that depth
    2. Convert to inline/crossline coordinates
    3. Extract the seismic patch centered at that (IL, XL)
    4. Extract the corresponding well log segment

    This ensures seismic and well log data are from the SAME physical location.

    Args:
        segy_loader: SEGYLoader instance
        survey_geometry: SeismicSurveyGeometry instance
    """

    def __init__(
        self,
        segy_loader,
        survey_geometry: SeismicSurveyGeometry,
    ):
        self.segy = segy_loader
        self.geometry = survey_geometry

    def extract_seismic_at_well(
        self,
        trajectory: WellTrajectory,
        md_center: float,
        patch_il: int = 32,
        patch_xl: int = 32,
        patch_depth: int = 32,
    ) -> Optional[np.ndarray]:
        """
        Extract a 3D seismic patch centered at the well position at md_center.

        Args:
            trajectory: WellTrajectory object
            md_center: Measured depth at patch center
            patch_il: Number of inlines in patch
            patch_xl: Number of crosslines in patch
            patch_depth: Number of depth samples in patch

        Returns:
            (patch_il, patch_xl, patch_depth) numpy array or None
        """
        # Get well position at this depth
        easting, northing, tvdss = trajectory.get_position_at_md(md_center)

        # Convert to inline/crossline
        il, xl = self.geometry.utm_to_ilxl(easting, northing)

        # Get patch bounds
        il_start, il_end, xl_start, xl_end = self.geometry.get_patch_bounds(
            il, xl, patch_il, patch_xl
        )

        # Read seismic sub-volume
        try:
            volume = self.segy.read_volume(
                il_range=(il_start, il_end),
                xl_range=(xl_start, xl_end),
            )
        except Exception:
            return None

        if volume is None or volume.size == 0:
            return None

        # Pad if needed
        actual_il = il_end - il_start
        actual_xl = xl_end - xl_start
        if actual_il < patch_il or actual_xl < patch_xl:
            pad_il = max(0, patch_il - actual_il)
            pad_xl = max(0, patch_xl - actual_xl)
            volume = np.pad(volume, ((0, pad_il), (0, pad_xl), (0, 0)), mode="constant")

        # Slice depth dimension: center around the equivalent time/depth
        # For PSDM (pre-stack depth migration), the samples are in depth (meters)
        # Map TVDSS to sample index
        # This requires knowing the depth datum of the seismic
        depth_start_idx = self._tvd_to_sample_index(tvdss, patch_depth)

        depth_end = depth_start_idx + patch_depth
        if depth_end > volume.shape[2]:
            depth_start_idx = volume.shape[2] - patch_depth
            depth_end = volume.shape[2]

        if depth_start_idx < 0:
            depth_start_idx = 0
            depth_end = patch_depth

        if depth_end <= depth_start_idx:
            return None

        volume = volume[:patch_il, :patch_xl, depth_start_idx:depth_end]

        # Final pad in depth
        if volume.shape[2] < patch_depth:
            pad_d = patch_depth - volume.shape[2]
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad_d)), mode="constant")

        return volume.astype(np.float32)

    def _tvd_to_sample_index(self, tvdss: float, patch_depth: int) -> int:
        """
        Convert TVD subsea to seismic sample index.

        Volve PSDM seismic:
        - Datum: MSL (0m)
        - Sample interval: ~4m depth (from binary header: 4000us at ~2000m/s ≈ 4m)
        - 850 samples × 4m ≈ 3400m depth range (covers 0-3400m TVDSS)
        """
        depth_step_m = 4.0      # PSDM depth sampling
        seismic_datum_m = 0.0   # MSL datum

        sample_idx = int((tvdss - seismic_datum_m) / depth_step_m - patch_depth // 2)
        return max(0, sample_idx)

    def extract_training_pair(
        self,
        trajectory: WellTrajectory,
        well_log_data: Dict,
        md_start: float,
        md_end: float,
        well_seq_len: int = 128,
        patch_il: int = 32,
        patch_xl: int = 32,
        patch_depth: int = 32,
        well_curves: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Extract a physically-aligned training pair.

        The seismic patch is centered at the well trajectory position
        corresponding to the midpoint of the well log segment.

        Args:
            trajectory: Well trajectory
            well_log_data: Dict with 'depth', curve arrays from LAS
            md_start, md_end: Measured depth range for well log segment
            well_seq_len: Target well log sequence length
            patch_il, patch_xl, patch_depth: Seismic patch dimensions
            well_curves: Curve names to include

        Returns:
            dict with seismic, well_log, labels, position metadata
        """
        if well_curves is None:
            well_curves = ["GR", "RT", "RHOB", "NPHI", "DT", "CALI", "PEF"]

        # Mid-point for seismic patch center
        md_center = (md_start + md_end) / 2.0

        # Get position for logging
        easting, northing, tvdss = trajectory.get_position_at_md(md_center)
        il, xl = self.geometry.utm_to_ilxl(easting, northing)

        # Extract seismic patch at well position
        seismic_patch = self.extract_seismic_at_well(
            trajectory, md_center,
            patch_il=patch_il, patch_xl=patch_xl, patch_depth=patch_depth,
        )

        if seismic_patch is None:
            return None

        # Normalize seismic
        seis_mean = np.nanmean(seismic_patch)
        seis_std = np.nanstd(seismic_patch) + 1e-8
        seismic_patch = (seismic_patch - seis_mean) / seis_std

        # Extract well log segment
        well_depth = well_log_data.get("depth")
        if well_depth is None:
            return None

        # Find indices for md_start..md_end
        depth_mask = (well_depth >= md_start) & (well_depth <= md_end)
        indices = np.where(depth_mask)[0]

        if len(indices) < well_seq_len // 2:
            return None

        # Subsample to target length
        if len(indices) > well_seq_len:
            step = len(indices) // well_seq_len
            indices = indices[::step][:well_seq_len]
        elif len(indices) < well_seq_len:
            # Pad with nearest
            pad_n = well_seq_len - len(indices)
            indices = np.concatenate([
                np.full(pad_n // 2, indices[0]),
                indices,
                np.full(pad_n - pad_n // 2, indices[-1]),
            ])

        indices = indices[:well_seq_len]

        # Build well log tensor
        well_curves_arr = []
        well_mask = np.ones(well_seq_len, dtype=np.float32)

        for curve_name in well_curves:
            if curve_name in well_log_data:
                vals = well_log_data[curve_name][indices]
                null_val = well_log_data.get("null_value", -999.25)
                vals = np.where(
                    np.isclose(vals, null_val, atol=0.01) | np.isnan(vals),
                    0.0, vals,
                )
                # Per-curve normalize
                c_mean = np.nanmean(vals[vals != 0]) if np.any(vals != 0) else 0.0
                c_std = np.nanstd(vals[vals != 0]) + 1e-8
                vals = np.where(vals != 0, (vals - c_mean) / c_std, 0.0)
                if c_std < 1e-6:
                    well_mask[:] = 0.0
            else:
                vals = np.zeros(well_seq_len, dtype=np.float32)

            well_curves_arr.append(vals)

        # Build item
        return {
            "well_name": trajectory.well_name,
            "md_center": md_center,
            "tvdss": tvdss,
            "il": float(il),
            "xl": float(xl),
            "easting": easting,
            "northing": northing,
            "seismic": seismic_patch,  # (il, xl, depth)
            "well_log_curves": well_curves_arr,  # list of (L,) arrays
            "well_mask": well_mask,
        }


# ==============================================================================
# Well Header Parser (extract KB, GL, coordinates from LAS)
# ==============================================================================

def parse_well_header_from_las(las_path: str) -> Dict:
    """
    Extract well header information from LAS file.

    Returns dict with keys:
        well_name, latitude, longitude, kb_elevation, ground_elevation,
        permanent_datum, elev_log_zero
    """
    result = {}

    with open(las_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    for line in content.split("\n"):
        line_upper = line.upper()

        # Coordinates (DMS format: "058 26' 29.907\" N")
        if "LATI" in line_upper:
            m = re.search(r"(\d+)\s+(\d+)'\s*([\d.]+)\"", line)
            if m:
                deg, min_, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
                hemi = "S" if "S" in line.split(":")[0].upper() else "N"
                result["latitude"] = dms_to_decimal(deg, min_, sec, hemi)

        if "LONG" in line_upper:
            m = re.search(r"(\d+)\s+(\d+)'\s*([\d.]+)\"", line)
            if m:
                deg, min_, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
                hemi = "W" if "W" in line.split(":")[0].upper() else "E"
                result["longitude"] = dms_to_decimal(deg, min_, sec, hemi)

        # Elevation datums
        if "ELZ" in line_upper:
            try:
                vals = line.split(":")[0].split()
                result["elev_log_zero"] = float(vals[-1])
            except (ValueError, IndexError):
                pass

        if "PD" in line_upper and "." in line:
            try:
                vals = line.split(":")[0].split()
                result["permanent_datum"] = vals[-1]
            except (ValueError, IndexError):
                pass

        if "KB" in line_upper and "." in line:
            try:
                vals = line.split(":")[0].split()
                result["kb_elevation"] = float(vals[-1])
            except (ValueError, IndexError):
                pass

        if "WELL" in line_upper and "." in line:
            well_name = line.split(":")[0].split(".")[-1].strip()
            result["well_name"] = well_name

    return result


# ==============================================================================
# Volve Field Presets
# ==============================================================================

# Known well coordinates for the Volve field (public data)
VOLVE_WELL_COORDS = {
    "15_9-F-1":    (58.441641, 1.887419),
    "15_9-F-1 A":  (58.441641, 1.887419),
    "15_9-F-1 B":  (58.441641, 1.887419),
    "15_9-F-1 C":  (58.441641, 1.887419),
    "15_9-F-4":    (58.441700, 1.887400),
    "15_9-F-5":    (58.441690, 1.887450),
    "15_9-F-7":    (58.441630, 1.887461),
    "15_9-F-9":    (58.441660, 1.887450),
    "15_9-F-9 A":  (58.441660, 1.887450),
    "15_9-F-10":   (58.441612, 1.887480),
    "15_9-F-11":   (58.441656, 1.887464),
    "15_9-F-11 A": (58.441654, 1.887463),
    "15_9-F-11 B": (58.441656, 1.887464),
    "15_9-F-12":   (58.441640, 1.887470),
    "15_9-F-14":   (58.441600, 1.887500),
    "15_9-F-15":   (58.441590, 1.887530),
    "15_9-F-15 A": (58.441590, 1.887530),
    "15_9-F-15 B": (58.441590, 1.887530),
    "15_9-F-15 C": (58.441590, 1.887530),
    "15_9-F-15 D": (58.441585, 1.887541),
    "15_9-19 A":   (58.441700, 1.887350),
    "15_9-19 B":   (58.441700, 1.887350),
    "15_9-19 B&BT2": (58.441700, 1.887350),  # Sidetrack of 15_9-19 B
    "15_9-19 S":   (58.441700, 1.887350),
    "15_9-19 S&SR":  (58.441700, 1.887350),  # Sidetrack of 15_9-19 S
    "15_9-F-11 T2":  (58.441656, 1.887464),  # Sidetrack of 15_9-F-11
}

# Default Volve survey geometry (estimated, calibrate from well-known positions)
DEFAULT_VOLVE_GEOMETRY = SeismicSurveyGeometry(
    il_min=9961, il_max=10361,
    xl_min=1961, xl_max=2680,
    il_spacing=12.5, xl_spacing=12.5,
    utm_origin_e=434880.0,
    utm_origin_n=6474400.0,
    il_ref=10000, xl_ref=2320,
    rotation_deg=-1.5,
)


def build_well_trajectories(
    well_coords: Dict = None,
) -> Dict[str, WellTrajectory]:
    """
    Build WellTrajectory objects for Volve wells.

    Currently assumes VERTICAL wells (no deviation survey loaded).
    TODO: Load deviation surveys from well folders.

    Returns:
        {well_name: WellTrajectory}
    """
    if well_coords is None:
        well_coords = VOLVE_WELL_COORDS

    # Real well elevations extracted from EOWR PDFs (RT=Rotary Table)
    # RT to MSL = 54.9m for all Volve subsea template wells
    # Water depth (MSL to Seabed) = 91m, RT to Seabed = 145.9m
    REAL_KB = 54.9   # RT to MSL (Kelly Bushing elevation above Mean Sea Level)
    REAL_WD = 91.0   # Water depth

    trajectories = {}
    for well_name, (lat, lon) in well_coords.items():
        trajectories[well_name] = WellTrajectory(
            well_name=well_name,
            surface_lat=lat,
            surface_lon=lon,
            kb_elevation=REAL_KB,
            ground_elevation=0.0,
        )

    return trajectories
