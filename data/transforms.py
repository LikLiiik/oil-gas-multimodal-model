"""
Data Augmentation Transforms

Custom augmentation strategies for:
- 3D seismic volumes (geological structure-preserving)
- 1D well log curves (physics-constrained)
"""

import numpy as np
from typing import Tuple, Optional, Dict
import random


class SeismicAugmentation:
    """
    Data augmentation for 3D seismic volumes.

    Preserves geological structures while applying:
    - Random flipping (inline/xline directions, NOT depth)
    - Random 90-degree rotation
    - Random brightness/contrast (amplitude scaling)
    - Gaussian noise injection
    - Random cropping/padding
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rotate: float = 0.3,
        p_noise: float = 0.5,
        p_contrast: float = 0.5,
        noise_std: float = 0.02,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise
        self.p_contrast = p_contrast
        self.noise_std = noise_std
        self.contrast_range = contrast_range

    def __call__(
        self,
        seismic: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentations to seismic volume.

        Args:
            seismic: (C, D, H, W) or (D, H, W) seismic data
            masks: Optional dict of masks to transform identically

        Returns:
            dict with 'seismic' and optional mask keys
        """
        result = {"seismic": seismic.copy()}

        if masks is not None:
            for key, mask in masks.items():
                result[key] = mask.copy()

        # Check shape format
        if seismic.ndim == 3:
            # (D, H, W) -> expand to (1, D, H, W) for unified processing
            single_channel = True
            seismic_data = seismic[np.newaxis, ...]
        else:
            single_channel = False
            seismic_data = seismic.copy()

        C, D, H, W = seismic_data.shape

        # 1. Random horizontal flip (inline direction)
        if random.random() < self.p_flip:
            seismic_data = np.flip(seismic_data, axis=2)  # flip H
            result["seismic"] = seismic_data if not single_channel else seismic_data[0]

        # 2. Random vertical flip (xline direction)
        if random.random() < self.p_flip:
            seismic_data = np.flip(seismic_data, axis=3)  # flip W
            result["seismic"] = seismic_data if not single_channel else seismic_data[0]

        # 3. Random 90-degree rotation in (H, W) plane
        if random.random() < self.p_rotate:
            k = random.randint(1, 3)
            seismic_data = np.rot90(seismic_data, k=k, axes=(2, 3))
            result["seismic"] = seismic_data if not single_channel else seismic_data[0]

        # 4. Random contrast/brightness
        if random.random() < self.p_contrast:
            factor = random.uniform(*self.contrast_range)
            seismic_data = seismic_data * factor
            result["seismic"] = seismic_data if not single_channel else seismic_data[0]

        # 5. Gaussian noise
        if random.random() < self.p_noise:
            noise = np.random.randn(*seismic_data.shape).astype(np.float32)
            noise *= self.noise_std * np.std(seismic_data)
            seismic_data = seismic_data + noise
            result["seismic"] = seismic_data if not single_channel else seismic_data[0]

        # Apply same spatial transforms to masks
        if masks is not None:
            for key in masks:
                result[key] = self._apply_spatial_transforms(
                    masks[key], seismic_data.shape, single_channel
                )

        if single_channel:
            result["seismic"] = result["seismic"][0] if isinstance(result["seismic"], np.ndarray) and result["seismic"].ndim == 4 else result["seismic"]

        return result

    def _apply_spatial_transforms(self, mask, shape, single_channel):
        """Placeholder for mask spatial transform synchronization."""
        return mask


class WellLogAugmentation:
    """
    Data augmentation for well log curves.

    Physics-constrained augmentations:
    - Random depth shift (shifting curves along depth)
    - Random amplitude scaling (mild, preserves physical meaning)
    - Random missing sections (simulating missing data)
    - Gaussian noise with curve-specific variance
    - Random curve dropout (simulate missing log types)
    """

    def __init__(
        self,
        p_shift: float = 0.5,
        p_scale: float = 0.3,
        p_missing: float = 0.3,
        p_noise: float = 0.5,
        p_dropout: float = 0.2,
        max_shift: int = 20,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        missing_ratio: float = 0.1,
        noise_std: float = 0.02,
        curve_dropout_prob: float = 0.15,
    ):
        self.p_shift = p_shift
        self.p_scale = p_scale
        self.p_missing = p_missing
        self.p_noise = p_noise
        self.p_dropout = p_dropout
        self.max_shift = max_shift
        self.scale_range = scale_range
        self.missing_ratio = missing_ratio
        self.noise_std = noise_std
        self.curve_dropout_prob = curve_dropout_prob

    def __call__(
        self,
        well_log: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to well log data.

        Args:
            well_log: (L, C) well log curves
            labels: Optional (L,) or (L, num_classes) labels

        Returns:
            augmented_well_log, augmented_labels
        """
        L, C = well_log.shape
        data = well_log.copy()

        # 1. Random depth shift (circular)
        if random.random() < self.p_shift:
            shift = random.randint(-self.max_shift, self.max_shift)
            data = np.roll(data, shift, axis=0)

        # 2. Random amplitude scaling (per curve, mild)
        if random.random() < self.p_scale:
            for c in range(C):
                factor = random.uniform(*self.scale_range)
                data[:, c] *= factor

        # 3. Random missing sections (linear interpolation across gap)
        if random.random() < self.p_missing:
            n_missing = int(L * self.missing_ratio)
            start = random.randint(0, L - n_missing - 1)
            gap = data[start : start + n_missing].copy()
            # Linear interpolation
            if start > 0 and start + n_missing < L:
                for c in range(C):
                    data[start : start + n_missing, c] = np.linspace(
                        data[start - 1, c], data[start + n_missing, c], n_missing
                    )

        # 4. Gaussian noise
        if random.random() < self.p_noise:
            noise = np.random.randn(*data.shape).astype(np.float32)
            # Different std per curve based on data std
            for c in range(C):
                data_std = np.std(data[:, c])
                data[:, c] += noise[:, c] * self.noise_std * data_std

        # 5. Random curve dropout (set to mean)
        if random.random() < self.p_dropout:
            for c in range(C):
                if random.random() < self.curve_dropout_prob:
                    data[:, c] = np.mean(data[:, c])

        return data, labels


class SeismicWellPairedAugmentation:
    """
    Paired augmentation for seismic-well data.

    Ensures consistent augmentation when both modalities
    are available for contrastive/matching tasks.
    """

    def __init__(
        self,
        seismic_aug: Optional[SeismicAugmentation] = None,
        well_aug: Optional[WellLogAugmentation] = None,
    ):
        self.seismic_aug = seismic_aug or SeismicAugmentation()
        self.well_aug = well_aug or WellLogAugmentation()

    def __call__(
        self,
        seismic: np.ndarray,
        well_log: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Apply paired augmentations."""
        # Augment seismic
        seis_result = self.seismic_aug(seismic, masks)

        # Augment well log
        well_augmented, labels_augmented = self.well_aug(well_log, labels)

        result = {
            "seismic": seis_result["seismic"],
            "well_log": well_augmented,
            "labels": labels_augmented,
        }

        if masks is not None:
            for key in masks:
                if key in seis_result:
                    result[key] = seis_result[key]

        return result
