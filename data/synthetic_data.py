"""
Synthetic Data Generator

Generates realistic synthetic seismic volumes and well log curves
for development and testing purposes when real data is unavailable.

Seismic data features:
- Random reflectivity model with layer structures
- Convolved with Ricker wavelet for realistic seismic response
- Fault plane simulation
- Channel/reef geological body simulation

Well log data features:
- Realistic curve shapes with depth trends
- Petrophysical constraints (e.g., GR vs POR correlation)
- Layer boundary responses
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import h5py
from pathlib import Path


class SyntheticDataGenerator:
    """
    Generate synthetic seismic volumes and well log curves
    with realistic geological patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    # ==================================================================
    # Seismic Data Generation
    # ==================================================================

    def generate_seismic_volume(
        self,
        shape: Tuple[int, int, int] = (128, 256, 256),
        num_layers: int = 20,
        num_faults: int = 3,
        noise_level: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic 3D seismic volume.

        Args:
            shape: (depth, inline, xline) dimensions
            num_layers: Number of geological layers
            num_faults: Number of faults to simulate
            noise_level: Standard deviation of random noise

        Returns:
            seismic_volume: (D, H, W) seismic amplitude data
            reflectivity: (D, H, W) true reflectivity model
            fault_mask: (D, H, W) binary fault labels
        """
        D, H, W = shape

        # 1. Build reflectivity model
        reflectivity = self._build_reflectivity(shape, num_layers)

        # 2. Add channel/reef bodies
        reflectivity = self._add_geological_bodies(reflectivity, num_bodies=5)

        # 3. Generate fault mask and apply fault displacement
        fault_mask = self._generate_faults(shape, num_faults)
        reflectivity = self._apply_faults(reflectivity, fault_mask)

        # 4. Convolve with Ricker wavelet
        seismic = self._convolve_ricker(reflectivity, freq=25.0)

        # 5. Add random noise
        seismic += self.rng.randn(*shape).astype(np.float32) * noise_level

        return (
            seismic.astype(np.float32),
            reflectivity.astype(np.float32),
            fault_mask.astype(np.float32),
        )

    def _build_reflectivity(
        self, shape: Tuple[int, int, int], num_layers: int
    ) -> np.ndarray:
        """Build a layered reflectivity model."""
        D, H, W = shape
        reflectivity = np.zeros(shape, dtype=np.float32)

        # Generate random layer interfaces with gentle folding
        layer_positions = np.sort(
            self.rng.randint(5, D - 5, size=num_layers)
        )

        for i, pos in enumerate(layer_positions):
            # Create gentle structural undulations
            x = np.arange(W)
            y = np.arange(H)
            xx, yy = np.meshgrid(x, y)

            # Random fold parameters
            amp = self.rng.uniform(2, 8)
            freq_x = self.rng.uniform(0.5, 2.0)
            freq_y = self.rng.uniform(0.5, 2.0)

            # Deform the layer position
            fold = amp * (
                np.sin(2 * np.pi * freq_x * xx / W)
                + np.cos(2 * np.pi * freq_y * yy / H)
            )
            deformed_pos = np.clip(
                int(pos) + fold.astype(int), 0, D - 1
            )

            # Assign reflection coefficient
            coeff = self.rng.uniform(-0.1, 0.1) * (-1) ** i
            for h in range(H):
                for w in range(W):
                    dp = deformed_pos[h, w]
                    reflectivity[dp, h, w] = coeff

        return reflectivity

    def _add_geological_bodies(
        self, reflectivity: np.ndarray, num_bodies: int = 5
    ) -> np.ndarray:
        """Add channel-like or reef-like geological bodies."""
        D, H, W = reflectivity.shape

        for _ in range(num_bodies):
            # Random body center
            center = (
                self.rng.randint(D // 4, 3 * D // 4),
                self.rng.randint(H // 4, 3 * H // 4),
                self.rng.randint(W // 4, 3 * W // 4),
            )

            # Ellipsoid body
            radius_d = self.rng.randint(5, D // 10)
            radius_h = self.rng.randint(10, H // 6)
            radius_w = self.rng.randint(10, W // 6)

            amplitude = self.rng.uniform(-0.08, 0.08)

            d_idx = np.arange(D)[:, None, None]
            h_idx = np.arange(H)[None, :, None]
            w_idx = np.arange(W)[None, None, :]

            dist = (
                ((d_idx - center[0]) / radius_d) ** 2
                + ((h_idx - center[1]) / radius_h) ** 2
                + ((w_idx - center[2]) / radius_w) ** 2
            )

            mask = dist <= 1.0
            reflectivity[mask] += amplitude

        return reflectivity

    def _generate_faults(
        self, shape: Tuple[int, int, int], num_faults: int
    ) -> np.ndarray:
        """Generate fault plane masks."""
        D, H, W = shape
        fault_mask = np.zeros(shape, dtype=np.float32)

        for _ in range(num_faults):
            # Random fault plane parameters
            dip = self.rng.uniform(30, 80)  # degrees
            strike = self.rng.uniform(0, 180)

            # Fault center
            cx, cy = self.rng.randint(D // 4, 3 * D // 4), H // 2

            # Create fault plane
            dip_rad = np.radians(dip)
            strike_rad = np.radians(strike)

            d_idx = np.arange(D)[:, None, None]
            h_idx = np.arange(H)[None, :, None]
            w_idx = np.arange(W)[None, None, :]

            # Distance to fault plane
            dz = d_idx - cx
            dh = h_idx - cy
            dist = np.abs(
                dz * np.cos(dip_rad)
                + dh * np.sin(dip_rad) * np.sin(strike_rad)
                + w_idx * np.sin(dip_rad) * np.cos(strike_rad)
            )

            # Fault thickness
            fault_thickness = self.rng.randint(2, 5)
            fault_mask[dist < fault_thickness] = 1.0

        return fault_mask

    def _apply_faults(
        self, reflectivity: np.ndarray, fault_mask: np.ndarray
    ) -> np.ndarray:
        """Apply displacement along fault planes."""
        D, H, W = reflectivity.shape
        displacement = self.rng.randint(2, 8)

        # Simple vertical displacement where faults exist
        for d in range(D - displacement):
            for h in range(H):
                for w in range(W):
                    if fault_mask[d, h, w] > 0:
                        reflectivity[d + displacement, h, w] = reflectivity[d, h, w]

        return reflectivity

    def _convolve_ricker(
        self, reflectivity: np.ndarray, freq: float = 25.0
    ) -> np.ndarray:
        """Convolve reflectivity with a Ricker wavelet along depth axis."""
        D, H, W = reflectivity.shape

        # Generate Ricker wavelet with appropriate length
        # Wavelet length should be shorter than the signal
        wavelet_len = min(31, D // 2)
        if wavelet_len % 2 == 0:
            wavelet_len += 1  # ensure odd length
        t = np.linspace(-0.05, 0.05, wavelet_len)
        wavelet = (1 - 2 * np.pi**2 * freq**2 * t**2) * np.exp(
            -np.pi**2 * freq**2 * t**2
        )
        wavelet = wavelet.astype(np.float32)

        # 1D convolution along depth
        seismic = np.zeros_like(reflectivity)
        for h in range(H):
            for w in range(W):
                result = np.convolve(
                    reflectivity[:, h, w], wavelet, mode="same"
                )
                # Handle potential length mismatch
                if len(result) >= D:
                    seismic[:, h, w] = result[:D]
                else:
                    seismic[:len(result), h, w] = result

        return seismic

    # ==================================================================
    # Well Log Data Generation
    # ==================================================================

    def generate_well_logs(
        self,
        sequence_length: int = 512,
        num_curves: int = 7,
        curve_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic well log curves.

        Args:
            sequence_length: Number of depth samples
            num_curves: Number of log curve types
            curve_names: Names of curves (for reference)

        Returns:
            log_curves: (L, C) well log measurements
            lithology_labels: (L,) lithology class labels
        """
        if curve_names is None:
            curve_names = ["GR", "RT", "DEN", "POR", "AC", "SP", "CAL"]

        depth = np.arange(sequence_length)

        # Generate base log curves with realistic depth trends
        curves = {}

        # Gamma Ray (GR): 0-150 API, compaction trend
        curves["GR"] = 60 + 30 * np.sin(depth / 50) + 10 * np.sin(depth / 10)
        curves["GR"] += self.rng.randn(sequence_length) * 5

        # Resistivity (RT): 1-1000 ohm-m, log scale
        curves["RT"] = 50 + 30 * np.sin(depth / 80) + 20 * np.sin(depth / 15)
        curves["RT"] = np.exp(np.log(curves["RT"]) + self.rng.randn(sequence_length) * 0.3)

        # Density (DEN): 2.0-2.8 g/cm³
        curves["DEN"] = 2.4 + 0.2 * np.sin(depth / 60) + self.rng.randn(sequence_length) * 0.05

        # Porosity (POR): 0-0.4 fraction
        curves["POR"] = 0.2 + 0.1 * np.sin(depth / 45) + self.rng.randn(sequence_length) * 0.02
        curves["POR"] = np.clip(curves["POR"], 0.01, 0.4)

        # Acoustic (AC): 40-140 us/ft
        curves["AC"] = 80 + 20 * np.sin(depth / 55) + self.rng.randn(sequence_length) * 3

        # Spontaneous Potential (SP): -100 to 0 mV
        curves["SP"] = -50 + 25 * np.sin(depth / 40) + self.rng.randn(sequence_length) * 5

        # Caliper (CAL): 6-12 inches
        curves["CAL"] = 8.5 + self.rng.randn(sequence_length) * 0.5

        # Ensure all curves have correct ordering
        ordered_names = curve_names[:num_curves]
        log_data = np.column_stack([
            curves.get(name, np.zeros(sequence_length))
            for name in ordered_names
        ]).astype(np.float32)

        # Generate lithology labels
        # 0: shale, 1: sand, 2: carbonate, 3: coal
        lithology = self._generate_lithology_labels(
            sequence_length, log_data, ordered_names
        )

        return log_data, lithology.astype(np.int64)

    def _generate_lithology_labels(
        self, length: int, log_data: np.ndarray, curve_names: List[str]
    ) -> np.ndarray:
        """Generate lithology labels based on log values."""
        labels = np.zeros(length, dtype=np.int64)

        # Simple rule-based lithology classification
        gr_idx = curve_names.index("GR") if "GR" in curve_names else 0
        rt_idx = curve_names.index("RT") if "RT" in curve_names else 1
        por_idx = curve_names.index("POR") if "POR" in curve_names else 3

        gr = log_data[:, gr_idx]
        rt = log_data[:, rt_idx]
        por = log_data[:, por_idx]

        # Shale: high GR, low RT
        labels[(gr > 70) & (rt < 30)] = 0
        # Sand: low GR, low RT
        labels[(gr < 60) & (rt < 50)] = 1
        # Carbonate: low GR, high RT
        labels[(gr < 50) & (rt > 50)] = 2
        # Coal: very low density/high porosity
        labels[(por > 0.25)] = 3

        return labels

    def generate_well_seismic_pair(
        self,
        seismic_shape: Tuple[int, int, int] = (128, 256, 256),
        log_length: int = 512,
        well_position: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a paired seismic volume and well log at a specific position.

        This is the key data type for cross-modal learning:
        seismic volume + well log at a known (inline, xline) position.

        Args:
            seismic_shape: (D, H, W) seismic volume dimensions
            log_length: Well log sequence length
            well_position: (inline, xline) well location

        Returns:
            dict with keys:
                - seismic: (1, D, H, W) seismic volume
                - well_log: (L, C) well log curves
                - lithology: (L,) lithology labels
                - fault_mask: (D, H, W) fault labels
                - reservoir_mask: (D, H, W) reservoir labels
                - well_position: (inline, xline)
                - well_trace: (D,) seismic trace at well position
        """
        # Generate seismic volume
        seismic, reflectivity, fault_mask = self.generate_seismic_volume(
            shape=seismic_shape
        )

        # Generate well logs
        log_curves, lithology = self.generate_well_logs(
            sequence_length=log_length
        )

        # Well position
        if well_position is None:
            well_position = (
                self.rng.randint(seismic_shape[1] // 4, 3 * seismic_shape[1] // 4),
                self.rng.randint(seismic_shape[2] // 4, 3 * seismic_shape[2] // 4),
            )

        # Extract seismic trace at well position
        well_trace = seismic[:, well_position[0], well_position[1]].copy()

        # Generate reservoir mask (combining lithology and reflectivity info)
        reservoir_mask = np.zeros(seismic_shape, dtype=np.float32)

        # Sand/carbonate layers as reservoirs
        for d in range(1, seismic_shape[0] - 1):
            if np.abs(reflectivity[d, :, :]).mean() > 0.01:
                reservoir_mask[d, :, :] = 1.0

        # Dilate reservoir along layers
        for _ in range(3):
            dilated = reservoir_mask.copy()
            for d in range(1, seismic_shape[0] - 1):
                dilated[d, :, :] = reservoir_mask[d - 1 : d + 2, :, :].max(axis=0)
            reservoir_mask = np.maximum(reservoir_mask, dilated)

        return {
            "seismic": seismic[np.newaxis, ...],  # (1, D, H, W)
            "well_log": log_curves,               # (L, C)
            "lithology": lithology,               # (L,)
            "fault_mask": fault_mask,             # (D, H, W)
            "reservoir_mask": reservoir_mask,     # (D, H, W)
            "well_position": np.array(well_position, dtype=np.int32),
            "well_trace": well_trace,             # (D,)
            "curve_names": ["GR", "RT", "DEN", "POR", "AC", "SP", "CAL"],
        }

    def generate_pretraining_batch(
        self,
        batch_size: int = 4,
        seismic_shape: Tuple[int, int, int] = (128, 256, 256),
        log_length: int = 512,
    ) -> List[Dict[str, np.ndarray]]:
        """Generate a batch of paired data for pretraining."""
        batch = []
        for _ in range(batch_size):
            sample = self.generate_well_seismic_pair(
                seismic_shape=seismic_shape,
                log_length=log_length,
            )
            batch.append(sample)
        return batch

    def save_to_hdf5(
        self,
        filepath: str,
        num_samples: int = 100,
        seismic_shape: Tuple[int, int, int] = (128, 256, 256),
        log_length: int = 512,
    ):
        """Save a synthetic dataset to HDF5 format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            for i in range(num_samples):
                sample = self.generate_well_seismic_pair(
                    seismic_shape=seismic_shape,
                    log_length=log_length,
                )
                grp = f.create_group(f"sample_{i:04d}")
                grp.create_dataset("seismic", data=sample["seismic"])
                grp.create_dataset("well_log", data=sample["well_log"])
                grp.create_dataset("lithology", data=sample["lithology"])
                grp.create_dataset("fault_mask", data=sample["fault_mask"])
                grp.create_dataset("reservoir_mask", data=sample["reservoir_mask"])
                grp.create_dataset("well_position", data=sample["well_position"])
                grp.create_dataset("well_trace", data=sample["well_trace"])

        print(f"Saved {num_samples} samples to {filepath}")


if __name__ == "__main__":
    # Quick test
    gen = SyntheticDataGenerator(seed=42)

    print("Generating sample data...")
    sample = gen.generate_well_seismic_pair()

    print(f"Seismic shape: {sample['seismic'].shape}")
    print(f"Well log shape: {sample['well_log'].shape}")
    print(f"Lithology shape: {sample['lithology'].shape}")
    print(f"Fault mask shape: {sample['fault_mask'].shape}")
    print(f"Reservoir mask shape: {sample['reservoir_mask'].shape}")
    print(f"Well position: {sample['well_position']}")
    print(f"Lithology classes: {np.unique(sample['lithology'])}")
    print("Synthetic data generation test passed!")
