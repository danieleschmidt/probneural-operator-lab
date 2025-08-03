"""Data loading utilities for neural operators."""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .datasets import PDEDataset, NavierStokesDataset, BurgersDataset, DarcyFlowDataset, HeatEquationDataset


class PDEDataLoader:
    """Enhanced DataLoader for PDE datasets with additional functionality."""
    
    def __init__(self, 
                 dataset: PDEDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = False):
        """Initialize PDE DataLoader.
        
        Args:
            dataset: PDE dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        """Get batch size."""
        return self.dataloader.batch_size
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for testing."""
        return next(iter(self.dataloader))
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return self.dataset.get_stats()
    
    def get_grid(self) -> Optional[torch.Tensor]:
        """Get spatial grid."""
        return self.dataset.get_grid()


def create_dataloaders(dataset_name: str,
                      data_path: Union[str, Path],
                      batch_size: int = 32,
                      split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                      num_workers: int = 0,
                      pin_memory: bool = False,
                      **dataset_kwargs) -> Dict[str, PDEDataLoader]:
    """Create train/val/test dataloaders for a PDE dataset.
    
    Args:
        dataset_name: Name of dataset ("navier_stokes", "burgers", "darcy", "heat")
        data_path: Path to dataset
        batch_size: Batch size for all loaders
        split_ratios: Ratios for (train, val, test) splits
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Dictionary with train/val/test dataloaders
    """
    # Dataset class mapping
    dataset_classes = {
        "navier_stokes": NavierStokesDataset,
        "burgers": BurgersDataset,
        "darcy": DarcyFlowDataset,
        "darcy_flow": DarcyFlowDataset,
        "heat": HeatEquationDataset,
        "heat_equation": HeatEquationDataset
    }
    
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_classes.keys())}")
    
    dataset_class = dataset_classes[dataset_name]
    
    # Check if pre-split data exists
    data_path = Path(data_path)
    
    # Try to load pre-split datasets
    dataloaders = {}
    for split in ["train", "val", "test"]:
        try:
            dataset = dataset_class(data_path, split=split, **dataset_kwargs)
            dataloader = PDEDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == "train")
            )
            dataloaders[split] = dataloader
            print(f"Loaded {split} dataset: {len(dataset)} samples")
            
        except Exception as e:
            print(f"Could not load {split} dataset: {e}")
    
    # If we have all splits, return them
    if len(dataloaders) == 3:
        return dataloaders
    
    # Otherwise, create a single dataset and split it
    print("Creating single dataset and splitting...")
    full_dataset = dataset_class(data_path, split="train", **dataset_kwargs)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(split_ratios[0] * total_size)
    val_size = int(split_ratios[1] * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    dataloaders = {
        "train": PDEDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ),
        "val": PDEDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        ),
        "test": PDEDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    }
    
    print(f"Split dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return dataloaders


def create_dataloader_from_tensors(inputs: torch.Tensor,
                                  outputs: torch.Tensor,
                                  batch_size: int = 32,
                                  shuffle: bool = True,
                                  **kwargs) -> DataLoader:
    """Create DataLoader from tensor data.
    
    Args:
        inputs: Input tensor
        outputs: Output tensor  
        batch_size: Batch size
        shuffle: Whether to shuffle
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import TensorDataset
    
    dataset = TensorDataset(inputs, outputs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "navier_stokes": {
            "description": "2D Navier-Stokes equation solutions",
            "spatial_dim": 2,
            "temporal": True,
            "equation": "∂u/∂t + (u·∇)u = -∇p + ν∇²u + f",
            "parameters": ["viscosity", "resolution", "time_steps"]
        },
        "burgers": {
            "description": "1D viscous Burgers equation solutions", 
            "spatial_dim": 1,
            "temporal": True,
            "equation": "∂u/∂t + u∂u/∂x = ν∂²u/∂x²",
            "parameters": ["viscosity", "resolution", "time_steps"]
        },
        "darcy": {
            "description": "2D Darcy flow equation solutions",
            "spatial_dim": 2,
            "temporal": False,
            "equation": "-∇·(κ(x)∇u) = f(x)",
            "parameters": ["resolution"]
        },
        "heat": {
            "description": "2D heat equation solutions",
            "spatial_dim": 2,
            "temporal": True,
            "equation": "∂u/∂t = α∇²u + f",
            "parameters": ["diffusivity", "resolution", "time_steps"]
        }
    }
    
    return info.get(dataset_name, {"description": "Unknown dataset"})


class DatasetRegistry:
    """Registry for available datasets."""
    
    _datasets = {
        "navier_stokes": NavierStokesDataset,
        "burgers": BurgersDataset,
        "darcy": DarcyFlowDataset,
        "heat": HeatEquationDataset
    }
    
    @classmethod
    def list_datasets(cls) -> list:
        """List available datasets."""
        return list(cls._datasets.keys())
    
    @classmethod
    def get_dataset_class(cls, name: str):
        """Get dataset class by name."""
        if name not in cls._datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {cls.list_datasets()}")
        return cls._datasets[name]
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class):
        """Register a new dataset class."""
        cls._datasets[name] = dataset_class