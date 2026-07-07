"""
PyTorch Dataset Classes

Provides dataset classes for:
- Pretraining: paired seismic-well data with masks
- Finetuning: task-specific datasets with labels

支持从HDF5文件或合成数据生成器加载数据。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import h5py

from .synthetic_data import SyntheticDataGenerator
from .transforms import SeismicAugmentation, WellLogAugmentation


class PretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining.

    Each sample contains:
    - seismic: (1, D, H, W) 3D seismic volume
    - well_log: (L, C) well log curves
    - well_position: (2,) well location in seismic grid
    - well_trace: (D,) seismic trace at well position (for CMCL)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 1000,
        seismic_shape: Tuple[int, int, int] = (128, 256, 256),
        log_length: int = 512,
        use_synthetic: bool = True,
        seismic_aug: Optional[SeismicAugmentation] = None,
        well_aug: Optional[WellLogAugmentation] = None,
        seed: int = 42,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.num_samples = num_samples
        self.seismic_shape = seismic_shape
        self.log_length = log_length
        self.use_synthetic = use_synthetic
        self.seismic_aug = seismic_aug
        self.well_aug = well_aug

        if use_synthetic:
            self.generator = SyntheticDataGenerator(seed=seed)
        elif data_path is not None and self.data_path.exists():
            self.h5file = h5py.File(data_path, "r")
            self.keys = list(self.h5file.keys())
            self.num_samples = len(self.keys)
        else:
            raise ValueError("Must provide either data_path or use_synthetic=True")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_synthetic:
            sample = self.generator.generate_well_seismic_pair(
                seismic_shape=self.seismic_shape,
                log_length=self.log_length,
            )
        else:
            grp = self.h5file[self.keys[idx]]
            sample = {
                "seismic": grp["seismic"][:],
                "well_log": grp["well_log"][:],
                "lithology": grp["lithology"][:],
                "fault_mask": grp["fault_mask"][:],
                "reservoir_mask": grp["reservoir_mask"][:],
                "well_position": grp["well_position"][:],
                "well_trace": grp["well_trace"][:],
            }

        # Apply augmentations
        if self.seismic_aug is not None:
            aug_result = self.seismic_aug(
                sample["seismic"],
                masks={
                    "fault_mask": sample.get("fault_mask"),
                    "reservoir_mask": sample.get("reservoir_mask"),
                },
            )
            seismic = aug_result["seismic"]
        else:
            seismic = sample["seismic"]

        if self.well_aug is not None and "well_log" in sample:
            well_log, _ = self.well_aug(sample["well_log"])
        else:
            well_log = sample["well_log"]

        # Convert to tensors
        result = {
            "seismic": torch.from_numpy(seismic).float(),
            "well_log": torch.from_numpy(well_log).float(),
            "well_position": torch.from_numpy(sample["well_position"]).long(),
            "well_trace": torch.from_numpy(sample["well_trace"]).float(),
        }

        if "fault_mask" in sample and sample["fault_mask"] is not None:
            result["fault_mask"] = torch.from_numpy(sample["fault_mask"]).float()
        if "reservoir_mask" in sample and sample["reservoir_mask"] is not None:
            result["reservoir_mask"] = torch.from_numpy(sample["reservoir_mask"]).float()
        if "lithology" in sample:
            result["lithology"] = torch.from_numpy(sample["lithology"]).long()

        return result


class FinetuneDataset(Dataset):
    """
    Dataset for downstream task finetuning.

    Supports tasks:
    - fault_detection: seismic + fault mask labels
    - reservoir_prediction: seismic + well_log + reservoir mask labels
    - lithology_classification: well_log (or seismic trace) + lithology labels

    Task-specific data format:
    - fault_detection: returns seismic, fault_mask
    - reservoir_prediction: returns seismic, well_log, reservoir_mask
    - lithology_classification: returns well_log (or seismic_trace), lithology
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 500,
        seismic_shape: Tuple[int, int, int] = (128, 256, 256),
        log_length: int = 512,
        task: str = "fault_detection",
        use_synthetic: bool = True,
        seismic_aug: Optional[SeismicAugmentation] = None,
        well_aug: Optional[WellLogAugmentation] = None,
        seed: int = 42,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.num_samples = num_samples
        self.seismic_shape = seismic_shape
        self.log_length = log_length
        self.task = task
        self.use_synthetic = use_synthetic
        self.seismic_aug = seismic_aug
        self.well_aug = well_aug

        assert task in [
            "fault_detection",
            "reservoir_prediction",
            "lithology_classification",
        ], f"Unknown task: {task}"

        if use_synthetic:
            self.generator = SyntheticDataGenerator(seed=seed)
        elif data_path is not None and self.data_path.exists():
            self.h5file = h5py.File(data_path, "r")
            self.keys = list(self.h5file.keys())
            self.num_samples = len(self.keys)
        else:
            raise ValueError("Must provide either data_path or use_synthetic=True")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_synthetic:
            sample = self.generator.generate_well_seismic_pair(
                seismic_shape=self.seismic_shape,
                log_length=self.log_length,
            )
        else:
            grp = self.h5file[self.keys[idx]]
            sample = {
                "seismic": grp["seismic"][:],
                "well_log": grp["well_log"][:],
                "lithology": grp["lithology"][:],
                "fault_mask": grp["fault_mask"][:],
                "reservoir_mask": grp["reservoir_mask"][:],
                "well_position": grp["well_position"][:],
                "well_trace": grp["well_trace"][:],
            }

        # Apply augmentations
        if self.seismic_aug is not None:
            masks = {}
            if self.task == "fault_detection":
                masks["fault_mask"] = sample["fault_mask"]
            elif self.task == "reservoir_prediction":
                masks["reservoir_mask"] = sample["reservoir_mask"]

            aug_result = self.seismic_aug(sample["seismic"], masks=masks if masks else None)
            seismic = aug_result["seismic"]
            for key in masks:
                sample[key] = aug_result.get(key, sample[key])
        else:
            seismic = sample["seismic"]

        if "well_log" in sample and self.well_aug is not None:
            well_log, _ = self.well_aug(sample["well_log"])
        else:
            well_log = sample.get("well_log")

        # Build task-specific output
        if self.task == "fault_detection":
            return {
                "seismic": torch.from_numpy(seismic).float(),
                "fault_mask": torch.from_numpy(sample["fault_mask"]).float().unsqueeze(0),
            }

        elif self.task == "reservoir_prediction":
            return {
                "seismic": torch.from_numpy(seismic).float(),
                "well_log": torch.from_numpy(well_log).float() if well_log is not None else torch.zeros(1),
                "reservoir_mask": torch.from_numpy(sample["reservoir_mask"]).float().unsqueeze(0),
                "well_position": torch.from_numpy(sample["well_position"]).long(),
            }

        elif self.task == "lithology_classification":
            return {
                "well_log": torch.from_numpy(well_log).float() if well_log is not None else torch.zeros(1),
                "well_trace": torch.from_numpy(sample["well_trace"]).float(),
                "lithology": torch.from_numpy(sample["lithology"]).long(),
            }


class SeismicWellDataset(Dataset):
    """
    Full paired seismic-well dataset supporting all tasks.

    This is the primary dataset class that handles real data
    from HDF5 archives or generates synthetic data.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_samples: int = 500,
        seismic_shape: Tuple[int, int, int] = (128, 256, 256),
        log_length: int = 512,
        use_synthetic: bool = True,
        phase: str = "pretrain",  # pretrain, train, val, test
        seed: int = 42,
    ):
        self.phase = phase
        self.seismic_shape = seismic_shape
        self.log_length = log_length
        self.use_synthetic = use_synthetic

        if use_synthetic:
            self.generator = SyntheticDataGenerator(seed=seed)
            self.num_samples = num_samples
        elif data_path:
            self.data_path = Path(data_path)
            self.h5file = h5py.File(data_path, "r")
            self.keys = list(self.h5file.keys())
            self.num_samples = len(self.keys)
        else:
            raise ValueError("Provide data_path or use_synthetic=True")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_synthetic:
            sample = self.generator.generate_well_seismic_pair(
                seismic_shape=self.seismic_shape,
                log_length=self.log_length,
            )
        else:
            grp = self.h5file[self.keys[idx]]
            sample = {
                "seismic": grp["seismic"][:],
                "well_log": grp["well_log"][:],
                "lithology": grp["lithology"][:],
                "fault_mask": grp["fault_mask"][:],
                "reservoir_mask": grp["reservoir_mask"][:],
                "well_position": grp["well_position"][:],
                "well_trace": grp["well_trace"][:],
            }

        # Convert all to tensors
        return {
            "seismic": torch.from_numpy(sample["seismic"]).float(),
            "well_log": torch.from_numpy(sample["well_log"]).float(),
            "well_position": torch.from_numpy(sample["well_position"]).long(),
            "well_trace": torch.from_numpy(sample["well_trace"]).float(),
            "fault_mask": torch.from_numpy(sample.get("fault_mask", np.zeros(self.seismic_shape))).float(),
            "reservoir_mask": torch.from_numpy(sample.get("reservoir_mask", np.zeros(self.seismic_shape))).float(),
            "lithology": torch.from_numpy(sample.get("lithology", np.zeros(self.log_length, dtype=np.int64))).long(),
        }


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
