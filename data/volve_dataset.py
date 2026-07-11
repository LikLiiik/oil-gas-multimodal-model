"""
Volve Field Dataset Pipeline

Loads the Equinor Volve field dataset for multi-modal geophysical pretraining:
- 3D Post-stack depth migration seismic (SEG-Y, IBM float32)
- 24 wells with petrophysical logs (LAS format)

Data sources:
  Seismic: ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy
  Well logs: Volve_Well_logs_pr_WELL/Well_logs_pr_WELL/

Key features:
1. IBM float32 -> IEEE float32 conversion for SEG-Y data
2. Well log parsing with curve standardization across wells
3. Seismic trace extraction at well positions
4. Multi-task label preparation (lithology, reservoir properties, faults)
5. Train/val/test split by wells

Usage:
    from data.volve_dataset import VolveDataset
    ds = VolveDataset(data_dir="E:/oilmodel")
    sample = ds[0]  # {seismic, well_log, labels}
"""

import os
import struct
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Well-seismic physical alignment
from .well_seismic_tie import (
    SeismicSurveyGeometry,
    WellTrajectory,
    WellSeismicDataExtractor,
    VOLVE_WELL_COORDS,
    DEFAULT_VOLVE_GEOMETRY,
    build_well_trajectories,
    parse_well_header_from_las,
)


# ==============================================================================
# IBM Float Conversion
# ==============================================================================

def ibm2ieee(ibm: np.ndarray) -> np.ndarray:
    """
    Convert IBM 360 floating point (SEG-Y format code 1) to IEEE float32.

    IBM float format: 1 sign bit, 7 exponent bits (base 16), 24 fraction bits
    IEEE float format: 1 sign bit, 8 exponent bits (base 2), 23 fraction bits

    Handles overflow by clipping to IEEE float32 range.
    """
    ibm = np.asarray(ibm, dtype=np.uint32)

    # Extract sign, exponent, fraction
    sign = (ibm >> 31) & 0x01
    exponent = ((ibm >> 24) & 0x7F).astype(np.int32) - 64  # bias 64
    fraction = (ibm & 0x00FFFFFF).astype(np.float64) / (2.0 ** 24)

    # Prevent overflow: 16^exponent must fit in float32 (max ~3.4e38)
    # 16^31 ≈ 5.4e37 (safe), 16^32 ≈ 8.6e38 (overflow), 16^33 ≈ 1.4e40
    # Clip exponent to [-64, 31] range
    exponent = np.clip(exponent, -64, 31)

    # Use float64 for intermediate computation to avoid overflow
    magnitude = np.power(16.0, exponent.astype(np.float64))
    result = (1.0 - 2.0 * sign.astype(np.float64)) * fraction * magnitude

    # Clip to float32 range
    result = np.clip(result, -3.4e38, 3.4e38)

    return result.astype(np.float32)


# ==============================================================================
# SEG-Y Volume Loader
# ==============================================================================

class SEGYLoader:
    """
    Load 3D post-stack seismic volume from SEG-Y file.

    Handles IBM float32 (format code 1) and IEEE float32 (format code 5).

    Args:
        segy_path: Path to SEG-Y file
        il_byte: Byte position of inline number in trace header (0-indexed)
        xl_byte: Byte position of crossline number in trace header
        num_samples: Number of samples per trace (auto-detected from binary header if 0)
        sample_format: 1=IBM float, 5=IEEE float (auto-detected if None)
    """

    def __init__(
        self,
        segy_path: str,
        il_byte: int = 188,  # SEG-Y Rev1: bytes 189-192
        xl_byte: int = 192,  # SEG-Y Rev1: bytes 193-196
        num_samples: int = 0,
        sample_format: Optional[int] = None,
    ):
        self.path = Path(segy_path)
        self.il_byte = il_byte
        self.xl_byte = xl_byte
        self.file_size = os.path.getsize(segy_path)

        with open(segy_path, "rb") as f:
            # Read binary header
            f.seek(3200)
            bh = f.read(400)
            self.sample_format = sample_format or struct.unpack(">h", bh[24:26])[0]
            self.num_samples = num_samples or struct.unpack(">h", bh[20:22])[0]
            sample_interval_us = struct.unpack(">h", bh[16:18])[0]

        self.sample_bytes = 4  # float32
        self.trace_size = 240 + self.num_samples * self.sample_bytes
        self.num_traces = (self.file_size - 3600) // self.trace_size

        # Scan trace headers for geometry
        self.inline_min, self.inline_max = None, None
        self.xline_min, self.xline_max = None, None
        self.trace_index = {}  # (inline, xline) -> file offset

        self._scan_geometry()

        self.shape = (
            self.inline_max - self.inline_min + 1,
            self.xline_max - self.xline_min + 1,
            self.num_samples,
        )

    def _scan_geometry(self):
        """Scan all trace headers to build inline/xline -> offset map."""
        inlines = set()
        xlines = set()

        with open(self.path, "rb") as f:
            f.seek(3600)
            for i in range(self.num_traces):
                offset = f.tell()
                th = f.read(240)
                il = struct.unpack(">i", th[self.il_byte:self.il_byte + 4])[0]
                xl = struct.unpack(">i", th[self.xl_byte:self.xl_byte + 4])[0]
                inlines.add(il)
                xlines.add(xl)
                self.trace_index[(il, xl)] = offset
                f.seek(self.num_samples * 4, 1)

        self.inline_min, self.inline_max = min(inlines), max(inlines)
        self.xline_min, self.xline_max = min(xlines), max(xlines)

    def read_volume(self, il_range=None, xl_range=None) -> np.ndarray:
        """
        Read full or partial 3D volume into numpy array.

        Args:
            il_range: (start, end) inline range, None for all
            xl_range: (start, end) crossline range, None for all

        Returns:
            (n_il, n_xl, n_samples) float32 volume
        """
        il_start = il_range[0] if il_range else self.inline_min
        il_end = il_range[1] if il_range else self.inline_max + 1
        xl_start = xl_range[0] if xl_range else self.xline_min
        xl_end = xl_range[1] if xl_range else self.xline_max + 1

        n_il = il_end - il_start
        n_xl = xl_end - xl_start

        volume = np.zeros((n_il, n_xl, self.num_samples), dtype=np.float32)

        with open(self.path, "rb") as f:
            for il_idx, il in enumerate(range(il_start, il_end)):
                for xl_idx, xl in enumerate(range(xl_start, xl_end)):
                    key = (il, xl)
                    if key not in self.trace_index:
                        continue

                    f.seek(self.trace_index[key] + 240)
                    raw = f.read(self.num_samples * 4)
                    trace = np.frombuffer(raw, dtype=np.uint32)

                    if self.sample_format == 1:
                        # Try IBM float first, but if values are extreme (>1e10),
                        # fall back to big-endian IEEE float32
                        trace_ibm = ibm2ieee(trace)
                        if np.nanmax(np.abs(trace_ibm)) < 1e10:
                            trace = trace_ibm
                        else:
                            # Likely big-endian IEEE float32 mislabeled as IBM
                            trace = np.frombuffer(raw, dtype='>f4').astype(np.float32)
                    elif self.sample_format == 5:
                        trace = np.frombuffer(raw, dtype='>f4').astype(np.float32)
                    else:
                        # Default: try big-endian IEEE first
                        trace = np.frombuffer(raw, dtype='>f4').astype(np.float32)

                    volume[il_idx, xl_idx, :] = trace

        return volume

    def read_trace(self, il: int, xl: int) -> Optional[np.ndarray]:
        """
        Read a single trace.

        Returns:
            1D numpy array of length num_samples, or None if not found
        """
        key = (il, xl)
        if key not in self.trace_index:
            return None

        with open(self.path, "rb") as f:
            f.seek(self.trace_index[key] + 240)
            raw = f.read(self.num_samples * 4)
            trace = np.frombuffer(raw, dtype=np.uint32)
            if self.sample_format == 1:
                trace_ibm = ibm2ieee(trace)
                if np.nanmax(np.abs(trace_ibm)) < 1e10:
                    trace = trace_ibm
                else:
                    trace = np.frombuffer(raw, dtype='>f4').astype(np.float32)
            else:
                trace = np.frombuffer(raw, dtype='>f4').astype(np.float32)

        return trace.astype(np.float32)

    def get_inline(self, il: int) -> np.ndarray:
        """Read all crosslines for a given inline."""
        n_xl = self.xline_max - self.xline_min + 1
        data = np.zeros((n_xl, self.num_samples), dtype=np.float32)
        for i, xl in enumerate(range(self.xline_min, self.xline_max + 1)):
            trace = self.read_trace(il, xl)
            if trace is not None:
                data[i, :] = trace
        return data

    def get_xline(self, xl: int) -> np.ndarray:
        """Read all inlines for a given crossline."""
        n_il = self.inline_max - self.inline_min + 1
        data = np.zeros((n_il, self.num_samples), dtype=np.float32)
        for i, il in enumerate(range(self.inline_min, self.inline_max + 1)):
            trace = self.read_trace(il, xl)
            if trace is not None:
                data[i, :] = trace
        return data


# ==============================================================================
# LAS Well Log Loader
# ==============================================================================

class LASWellLoader:
    """
    Parse and standardize LAS well log files.

    Reads LAS 2.0 files and provides consistent curve access across wells.
    Supports curve aliasing to standardize naming conventions.

    Args:
        curve_aliases: Dict mapping standard names to alternative curve mnemonics
    """

    # Standard curve aliases across the Volve dataset
    DEFAULT_ALIASES = {
        "GR": ["GR", "GR_1", "GRC", "SGR", "GAMMA"],
        "RT": ["RT", "RDEEP", "RLA5", "AT90", "AT90M", "ILD"],
        "RHOB": ["RHOB", "RHOZ", "RHO8", "DEN", "DENSITY"],
        "NPHI": ["NPHI", "NPHI_1", "TNPH", "NEUTRON", "NEU"],
        "DT": ["DT", "DTCO", "AC", "SONIC", "DTC"],
        "DTS": ["DTS", "DTSM", "DTST", "DTSH"],
        "CALI": ["CALI", "CAL", "HCAL", "CALIPER", "BS"],
        "DRHO": ["DRHO", "HDRA", "CORRECTION"],
        "PEF": ["PEF", "PEFZ", "PE", "PHOTOELECTRIC"],
        "SP": ["SP", "SP_1"],
        "ROP": ["ROP", "ROP_1"],
        # Petrophysical outputs (labels)
        "VSH": ["VSH", "VSH_1", "VSHALE", "VSH_GR", "VOLUME_OF_SHALE"],
        "PHIF": ["PHIF", "PHI", "POR", "POROSITY", "PHIT", "FINAL_POROSITY"],
        "SW": ["SW", "SW_1", "SWT", "WATER_SATURATION"],
        "KLOGH": ["KLOGH", "PERM", "PERMEABILITY", "K_LOGH"],
        "SAND_FLAG": ["SAND_FLAG", "SAND", "SAND_FLAG_1"],
        "COAL_FLAG": ["COAL_FLAG", "COAL"],
        "CARB_FLAG": ["CARB_FLAG", "CARBONATE_FLAG", "VCARB"],
        "BVW": ["BVW", "BOUND_VOLUME_WATER"],
        "FACIES": ["FACIES", "LITHOFACIES", "LITHOLOGY"],
    }

    def __init__(
        self,
        curve_aliases: Optional[Dict[str, List[str]]] = None,
    ):
        self.curve_aliases = curve_aliases or self.DEFAULT_ALIASES
        # Reverse mapping: alternative name -> standard name
        self._alias_map = {}
        for std_name, aliases in self.curve_aliases.items():
            for alias in aliases:
                self._alias_map[alias.upper()] = std_name

    def read(self, las_path: str) -> Dict[str, np.ndarray]:
        """
        Read a LAS file and return standardized curve dict.

        Args:
            las_path: Path to .las file

        Returns:
            dict with keys:
                - 'depth': (N,) depth values in meters
                - curve names: (N,) curve values (standardized names)
                - 'header': LAS well header info
                - 'null_value': float, the NULL sentinel value
        """
        with open(las_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        header = {}
        curve_info = []
        null_value = -999.25  # default
        in_ascii = False
        data_lines = []
        section = "well"  # Start in well section

        for line in lines:
            line = line.strip()

            # Detect sections
            if line.startswith("~W"):
                section = "well"
                continue
            elif line.startswith("~C"):
                section = "curve"
                continue
            elif line.startswith("~P"):
                section = "param"
                continue
            elif line.startswith("~A"):
                section = "ascii"
                in_ascii = True
                continue
            elif line.startswith("~") and line != "~A":
                in_ascii = False
                continue

            if in_ascii:
                if line:
                    data_lines.append(line)
                continue

            # Parse well header
            if section == "well" and "." in line and not line.startswith("#"):
                # LAS format: MNEM.UNIT    Value    : Description
                # First split on the first dot
                dot_pos = line.index(".")
                mnemonic = line[:dot_pos].strip()
                rest = line[dot_pos + 1:].strip()
                # rest = UNIT + spaces + Value + : + Description
                # Split on first ":" to separate value from description
                if ":" in rest:
                    unit_value, desc = rest.split(":", 1)
                    # Unit is before the first space in unit_value
                    unit_parts = unit_value.strip().split(None, 1)
                    if len(unit_parts) >= 1:
                        unit = unit_parts[0]
                        value = unit_parts[1] if len(unit_parts) > 1 else ""
                        header[mnemonic] = value.strip()
                else:
                    header[mnemonic] = rest.strip()

                if mnemonic == "NULL":
                    try:
                        null_value = float(header[mnemonic])
                    except ValueError:
                        pass

            # Parse curve info
            if section == "curve" and "." in line and not line.startswith("#"):
                # LAS format: MNEM.UNIT    API CODE    Description
                dot_pos = line.index(".")
                mnemonic = line[:dot_pos].strip()
                rest = line[dot_pos + 1:].strip()
                # Split unit from description
                parts = rest.split(":", 1)
                unit_desc = parts[0].strip()
                unit = unit_desc.split()[0] if unit_desc.split() else ""
                desc = parts[1].strip() if len(parts) > 1 else ""
                curve_info.append({
                    "mnemonic": mnemonic,
                    "unit": unit,
                    "description": desc,
                })

        # Parse numerical data
        n_curves = len(curve_info)
        if n_curves == 0:
            return {"depth": np.array([]), "header": header, "null_value": null_value}

        # Parse data values
        all_values = []
        for line in data_lines:
            parts = line.split()
            for p in parts:
                try:
                    all_values.append(float(p))
                except ValueError:
                    all_values.append(null_value)

        # Reshape to (n_samples, n_curves)
        n_total = len(all_values)
        n_rows = n_total // n_curves
        if n_rows * n_curves != n_total:
            # Truncate
            n_rows = n_total // n_curves

        data = np.array(all_values[:n_rows * n_curves]).reshape(n_rows, n_curves)

        # Build standardized output dict
        result = {"header": header, "null_value": null_value}

        for i, ci in enumerate(curve_info):
            mnemonic = ci["mnemonic"].upper()
            # Standardize name
            std_name = self._alias_map.get(mnemonic, mnemonic)

            if i < data.shape[1]:
                values = data[:, i]
                # Replace null values with NaN
                values = np.where(
                    np.isclose(values, null_value, atol=1e-3),
                    np.nan,
                    values,
                )
                result[std_name] = values

        # Ensure depth is first
        if "DEPTH" in result:
            result["depth"] = result.pop("DEPTH")
        elif "DEPT" in result:
            result["depth"] = result.pop("DEPT")

        return result

    def get_standard_curves(self, data: Dict) -> List[str]:
        """Return which standard curves are available in the data."""
        available = []
        for std_name in self.curve_aliases:
            if std_name in data:
                available.append(std_name)
        return available


# ==============================================================================
# Volve Dataset
# ==============================================================================

class VolveDataset(Dataset):
    """
    PyTorch Dataset for the Volve field.

    Provides paired (seismic_patch, well_log_sequence) samples with
    multi-task labels for pretraining and finetuning.

    Args:
        data_dir: Root directory containing SEG-Y and well log folders
        mode: 'pretrain' | 'finetune'
        task: 'fault_detection' | 'reservoir_prediction' | 'lithology'
        seismic_patch_size: (D, H, W) size of 3D seismic patches
        well_seq_len: Length of well log sequences
        well_curves: List of curve names to include in well log input
        train_wells: List of well names for training (None = auto-split)
        val_wells: List of well names for validation
        norm_stats: Precomputed normalization stats
    """

    def __init__(
        self,
        data_dir: str = r"E:\oilmodel",
        mode: str = "pretrain",
        task: Optional[str] = None,
        seismic_patch_size: Tuple[int, int, int] = (64, 64, 64),
        well_seq_len: int = 256,
        well_curves: Optional[List[str]] = None,
        train_wells: Optional[List[str]] = None,
        val_wells: Optional[List[str]] = None,
        norm_stats: Optional[Dict] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.task = task
        self.seismic_patch_size = seismic_patch_size
        self.well_seq_len = well_seq_len

        # Default well curves (standard geophysical logs)
        self.well_curves = well_curves or [
            "GR", "RT", "RHOB", "NPHI", "DT", "CALI", "PEF",
        ]

        # Load seismic
        segy_path = list(self.data_dir.glob("*.segy"))[0]
        print(f"Loading seismic from {segy_path}...")
        self.seismic = SEGYLoader(str(segy_path))
        print(f"  Shape: {self.seismic.shape}")

        # Find and load well logs
        self.well_dir = self.data_dir / "Volve_Well_logs_pr_WELL" / "Well_logs_pr_WELL"
        self.las_loader = LASWellLoader()

        self.wells = self._discover_wells()
        print(f"Found {len(self.wells)} wells with petrophysical data")

        # Load all well data
        self.well_data = {}
        self.well_trajectories = {}
        for well_name in self.wells:
            data = self._load_well_data(well_name)
            if data is not None:
                self.well_data[well_name] = data

        print(f"Loaded {len(self.well_data)} wells successfully")

        # ---- Build Well-Seismic Physical Alignment ----
        # Build well trajectories from known coordinates + deviation surveys
        self.trajectories = build_well_trajectories(VOLVE_WELL_COORDS)

        # Load deviation surveys from CSV files
        dev_dir = Path(__file__).parent / "volve_deviations"
        dev_loaded = 0
        if dev_dir.exists():
            for well_name in self.well_data.keys():
                csv_path = dev_dir / f"{well_name}.csv"
                if csv_path.exists() and well_name in self.trajectories:
                    traj = self.trajectories[well_name]
                    success = traj.load_deviation_csv(str(csv_path))
                    if success:
                        dev_loaded += 1
            if dev_loaded > 0:
                print(f"Loaded deviation surveys for {dev_loaded} wells")

        # Verify which wells have trajectory data
        wells_with_traj = set(self.trajectories.keys()) & set(self.well_data.keys())
        wells_with_dev = {w for w in wells_with_traj
                          if self.trajectories[w].has_deviation}
        if wells_with_traj:
            print(f"Wells: {len(wells_with_traj)} with coords, {len(wells_with_dev)} with deviation")

        # Build survey geometry and data extractor
        self.survey_geometry = DEFAULT_VOLVE_GEOMETRY
        self.extractor = WellSeismicDataExtractor(self.seismic, self.survey_geometry)

        # Log coordinate mapping for wells with deviation
        for wn in sorted(wells_with_dev)[:5]:
            traj = self.trajectories[wn]
            il, xl = self.survey_geometry.latlon_to_ilxl(traj.surface_lat, traj.surface_lon)
            # Show trajectory offsets
            if traj.has_deviation:
                max_off = np.sqrt(traj.x_offset[-1]**2 + traj.y_offset[-1]**2)
                print(f"  {wn}: ({traj.surface_lat:.6f},{traj.surface_lon:.6f})"
                      f" -> IL={il:.1f}, XL={xl:.1f}, 水平位移={max_off:.0f}m")
        # ---- End Well-Seismic Alignment ----

        # Train/val split
        well_names = sorted(self.well_data.keys())
        n_train = int(len(well_names) * 0.8)

        if train_wells is not None:
            self.train_wells = [w for w in train_wells if w in self.well_data]
            self.val_wells = [w for w in (val_wells or []) if w in self.well_data]
        else:
            self.train_wells = well_names[:n_train]
            self.val_wells = well_names[n_train:]

        print(f"Train wells: {self.train_wells}")
        print(f"Val wells: {self.val_wells}")

        # Build sample index
        self.samples = self._build_sample_index()
        print(f"Total samples: {len(self.samples)}")

        # Compute or load normalization stats
        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            self.norm_stats = self._compute_norm_stats()

    def _discover_wells(self) -> List[str]:
        """Find wells that have petrophysical interpretation data."""
        wells = []
        for well_dir in sorted(self.well_dir.iterdir()):
            if well_dir.is_dir() and not well_dir.name.startswith("."):
                # Check for petrophysical LAS files
                petro_dir = well_dir / "05.PETROPHYSICAL INTERPRETATION"
                if petro_dir.exists():
                    # Check for CPI or WLC files
                    has_data = False
                    for subdir in petro_dir.iterdir():
                        if subdir.is_dir():
                            las_files = list(subdir.glob("*.las"))
                            if las_files:
                                has_data = True
                                break
                    # Also check for WLC files directly
                    las_files = list(petro_dir.glob("*.las"))
                    if las_files:
                        has_data = True

                    if has_data:
                        wells.append(well_dir.name)

        return wells

    def _find_best_las(self, well_name: str) -> Optional[Path]:
        """
        Find the best LAS file for a well.
        Priority: WLC_PETRO_COMPUTED_INPUT > CPI > LFP > any LAS.
        """
        well_dir = self.well_dir / well_name / "05.PETROPHYSICAL INTERPRETATION"
        if not well_dir.exists():
            return None

        # Priority order
        patterns = [
            "WLC_PETRO_COMPUTED_INPUT*.LAS",
            "WLC_PETRO_COMPUTED_INPUT*.las",
            "*CPI*.las",
            "*CPI*.LAS",
            "*.las",
            "*.LAS",
        ]

        # Also check CPI subdirectory
        cpi_dir = well_dir / "CPI"
        for pattern in patterns:
            for search_dir in [well_dir, cpi_dir] if cpi_dir.exists() else [well_dir]:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def _load_well_data(self, well_name: str) -> Optional[Dict]:
        """Load and standardize well log data for a single well."""
        las_path = self._find_best_las(well_name)
        if las_path is None:
            return None

        try:
            data = self.las_loader.read(str(las_path))
        except Exception as e:
            print(f"  Warning: Failed to read {las_path}: {e}")
            return None

        if "depth" not in data or len(data["depth"]) < self.well_seq_len * 2:
            return None

        data["well_name"] = well_name
        data["las_path"] = str(las_path)

        # Extract well coordinates from header if available
        # Volve wells have LATI/LONG in DMS format
        if "LATI" in data["header"]:
            lat_str = data["header"]["LATI"]
            data["latitude"] = self._parse_dms(lat_str)
        if "LONG" in data["header"]:
            lon_str = data["header"]["LONG"]
            data["longitude"] = self._parse_dms(lon_str)

        return data

    def _parse_dms(self, dms_str: str) -> float:
        """Parse DMS coordinate string to decimal degrees."""
        try:
            # Format: "058 26' 29.907 N DMS"
            parts = dms_str.strip().split()
            if len(parts) >= 4:
                deg = float(parts[0])
                minute_part = parts[1].replace("'", "")
                min_val = float(minute_part)
                sec_part = parts[2].replace('"', '')
                sec_val = float(sec_part)
                hemisphere = parts[3]

                decimal = deg + min_val / 60.0 + sec_val / 3600.0
                if hemisphere in ("S", "W"):
                    decimal = -decimal
                return decimal
        except (ValueError, IndexError):
            pass
        return np.nan

    def _build_sample_index(self) -> List[Dict]:
        """Build a list of all possible (well, depth_start) sample indices."""
        samples = []
        wells_to_use = self.train_wells if self.mode != "test" else self.train_wells + self.val_wells

        for well_name in wells_to_use:
            if well_name not in self.well_data:
                continue
            data = self.well_data[well_name]
            depth = data["depth"]
            n = len(depth)

            # Slide window to create samples
            stride = self.well_seq_len // 2  # 50% overlap
            for start in range(0, n - self.well_seq_len, stride):
                samples.append({
                    "well_name": well_name,
                    "depth_start": start,
                    "depth_end": start + self.well_seq_len,
                })

        return samples

    def _compute_norm_stats(self) -> Dict:
        """Compute normalization statistics for seismic and well logs."""
        stats = {}

        # Seismic stats (from training wells' patches)
        seismic_values = []
        for sample in self.samples[:min(500, len(self.samples))]:
            patch = self._extract_seismic_patch(sample["well_name"], sample["depth_start"])
            if patch is not None:
                seismic_values.append(patch.flatten())

        if seismic_values:
            all_seis = np.concatenate(seismic_values)
            stats["seismic_mean"] = float(np.nanmean(all_seis))
            stats["seismic_std"] = float(np.nanstd(all_seis)) + 1e-8
        else:
            stats["seismic_mean"] = 0.0
            stats["seismic_std"] = 1.0

        # Well log stats (per curve)
        for curve_name in self.well_curves:
            values = []
            for well_name in self.train_wells:
                if well_name in self.well_data:
                    data = self.well_data[well_name]
                    if curve_name in data:
                        v = data[curve_name]
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
        self, well_name: str, md_center: float
    ) -> Optional[np.ndarray]:
        """
        Extract a 3D seismic patch at the ACTUAL well position.

        Uses the well trajectory to compute (easting, northing, tvdss) at
        the given measured depth, then converts to inline/crossline via
        the survey geometry to extract the correct seismic patch.

        This ensures seismic patches and well log segments are from the
        SAME physical location.

        Args:
            well_name: Well name
            md_center: Measured depth at patch center (meters)

        Returns:
            (D, H, W) seismic patch or None
        """
        p_d, p_h, p_w = self.seismic_patch_size

        # Get well trajectory
        traj = self.trajectories.get(well_name)
        if traj is None:
            # Fallback: use center of volume (no trajectory data)
            il_c = (self.seismic.inline_min + self.seismic.inline_max) // 2
            xl_c = (self.seismic.xline_min + self.seismic.xline_max) // 2
        else:
            # Use real well position
            easting, northing, tvdss = traj.get_position_at_md(md_center)
            il_c, xl_c = self.survey_geometry.utm_to_ilxl(easting, northing)

        # Get integer patch bounds
        il_start = int(il_c - p_h // 2)
        il_end = il_start + p_h
        xl_start = int(xl_c - p_w // 2)
        xl_end = xl_start + p_w

        # Clamp to survey extent
        il_start = max(self.seismic.inline_min, il_start)
        il_end = min(self.seismic.inline_max + 1, il_end)
        xl_start = max(self.seismic.xline_min, xl_start)
        xl_end = min(self.seismic.xline_max + 1, xl_end)

        # Read sub-volume
        try:
            volume = self.seismic.read_volume(
                il_range=(il_start, il_end),
                xl_range=(xl_start, xl_end),
            )
        except Exception:
            return None

        if volume is None or volume.size == 0:
            return None

        # Pad spatial dims if boundary-clipped
        actual_h = il_end - il_start
        actual_w = xl_end - xl_start
        if actual_h < p_h or actual_w < p_w:
            pad_h = max(0, p_h - actual_h)
            pad_w = max(0, p_w - actual_w)
            volume = np.pad(volume, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        # Depth slice centered around tvdss-mapped sample
        if traj is not None:
            _, _, tvdss = traj.get_position_at_md(md_center)
            # Approximate depth index: PSDM with ~4m depth step
            depth_step = 4.0
            center_sample = int(tvdss / depth_step)
        else:
            center_sample = volume.shape[2] // 2

        d_start = max(0, min(volume.shape[2] - p_d, center_sample - p_d // 2))
        volume = volume[:p_h, :p_w, d_start:d_start + p_d]

        if volume.shape[2] < p_d:
            pad_d = p_d - volume.shape[2]
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad_d)), mode="constant")

        return volume

    def _extract_well_sequence(
        self, well_name: str, start_idx: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract a standardized well log sequence."""
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

        # Labels (if available)
        for label_name in ["VSH", "PHIF", "SW", "KLOGH", "SAND_FLAG", "COAL_FLAG", "CARB_FLAG"]:
            if label_name in data:
                result[label_name] = data[label_name][start_idx:end_idx]

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - seismic: (1, D, H, W) normalized 3D patch
                - well_log: (C, L) normalized well log curves
                - well_mask: (L,) valid data mask
                - labels: task-specific labels dict
                - metadata: well_name, depth_range, etc.
        """
        sample = self.samples[idx]
        well_name = sample["well_name"]
        depth_start = sample["depth_start"]
        depth_end = sample["depth_end"]

        # MD center for seismic patch positioning (well trajectory mid-point)
        md_data = self.well_data.get(well_name, {})
        well_depth = md_data.get("depth")
        if well_depth is not None and depth_start < len(well_depth):
            md_center = float(well_depth[(depth_start + depth_end) // 2]
                            if (depth_start + depth_end) // 2 < len(well_depth)
                            else well_depth[depth_start])
        else:
            md_center = float(depth_start)

        # Seismic patch at ACTUAL well position
        seis_patch = self._extract_seismic_patch(well_name, md_center)
        if seis_patch is None:
            seis_patch = np.zeros(
                (self.seismic_patch_size[1], self.seismic_patch_size[2],
                 self.seismic_patch_size[0]), dtype=np.float32
            )

        # Normalize
        seis_patch = (seis_patch - self.norm_stats["seismic_mean"]) / self.norm_stats["seismic_std"]
        # Rearrange: (H, W, D) -> (D, H, W)
        seis_patch = np.transpose(seis_patch, (2, 0, 1))
        seis_tensor = torch.from_numpy(seis_patch.copy()).float().unsqueeze(0)

        # Well log sequence
        well_seq = self._extract_well_sequence(well_name, depth_start)
        well_curves_arr = []
        well_mask = np.ones(self.well_seq_len, dtype=np.float32)

        for curve_name in self.well_curves:
            if well_seq is not None and curve_name in well_seq:
                vals = well_seq[curve_name]
                mean = self.norm_stats.get(f"{curve_name}_mean", 0)
                std = self.norm_stats.get(f"{curve_name}_std", 1)
                vals = np.where(np.isnan(vals), 0, vals)
                vals = (vals - mean) / std
                well_mask = well_mask * (~np.isnan(well_seq[curve_name])).astype(np.float32) \
                    if well_seq is not None else well_mask
            else:
                vals = np.zeros(self.well_seq_len, dtype=np.float32)
                well_mask = np.zeros(self.well_seq_len, dtype=np.float32)

            well_curves_arr.append(vals)

        well_tensor = torch.from_numpy(np.stack(well_curves_arr)).float()
        mask_tensor = torch.from_numpy(well_mask).float()

        # Labels
        labels = {}
        if well_seq is not None:
            if "SAND_FLAG" in well_seq:
                labels["sand_flag"] = torch.from_numpy(
                    np.nan_to_num(well_seq["SAND_FLAG"], nan=0).copy()
                ).float()
            if "COAL_FLAG" in well_seq:
                labels["coal_flag"] = torch.from_numpy(
                    np.nan_to_num(well_seq["COAL_FLAG"], nan=0).copy()
                ).float()
            if "PHIF" in well_seq:
                labels["porosity"] = torch.from_numpy(
                    np.nan_to_num(well_seq["PHIF"], nan=0).copy()
                ).float()
            if "SW" in well_seq:
                labels["water_saturation"] = torch.from_numpy(
                    np.nan_to_num(well_seq["SW"], nan=0).copy()
                ).float()
            if "VSH" in well_seq:
                labels["vshale"] = torch.from_numpy(
                    np.nan_to_num(well_seq["VSH"], nan=0).copy()
                ).float()
            if "KLOGH" in well_seq:
                labels["permeability"] = torch.from_numpy(
                    np.nan_to_num(well_seq["KLOGH"], nan=0).copy()
                ).float()

        return {
            "seismic": seis_tensor,
            "well_log": well_tensor,
            "well_mask": mask_tensor,
            "labels": labels,
            "well_name": well_name,
            "depth_start": depth_start,
        }


# ==============================================================================
# Utility functions
# ==============================================================================

def create_volve_dataloaders(
    data_dir: str = r"E:\oilmodel",
    seismic_patch_size: Tuple[int, int, int] = (64, 64, 64),
    well_seq_len: int = 256,
    well_curves: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for the Volve dataset.

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    train_ds = VolveDataset(
        data_dir=data_dir,
        mode="pretrain",
        seismic_patch_size=seismic_patch_size,
        well_seq_len=well_seq_len,
        well_curves=well_curves,
    )

    val_ds = VolveDataset(
        data_dir=data_dir,
        mode="test",
        seismic_patch_size=seismic_patch_size,
        well_seq_len=well_seq_len,
        well_curves=well_curves,
        train_wells=train_ds.train_wells,
        val_wells=train_ds.val_wells,
        norm_stats=train_ds.norm_stats,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader
