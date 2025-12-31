"""
PyTorch Data Loading Module for Vanilla GAN Signature Generation.

This module provides data loading utilities including:
- Custom SignatureDataset class
- DataLoader factory functions
- Data augmentation pipelines
- Train/validation split support
- Sample batch visualization utilities

Author: Vanilla GAN Signatures Project
Date: December 2024
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Union
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_IMAGE_SIZE: int = 64
DEFAULT_VAL_SPLIT: float = 0.1
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


class SignatureDataset(Dataset):
    """
    Custom PyTorch Dataset for loading signature images.
    
    Loads grayscale signature images from a directory and applies
    transforms for GAN training.
    
    Attributes:
        root_dir: Path to the directory containing images.
        transform: Optional transform to apply to images.
        image_paths: List of paths to valid image files.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = SUPPORTED_EXTENSIONS
    ):
        """
        Initialize the SignatureDataset.
        
        Args:
            root_dir: Path to directory containing signature images.
            transform: Optional torchvision transform to apply.
            extensions: Tuple of valid image file extensions.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        # Validate directory exists
        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")
        
        # Collect all image paths
        self.image_paths = self._collect_image_paths()
        
        if len(self.image_paths) == 0:
            logger.warning(f"No images found in {root_dir}")
        else:
            logger.info(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def _collect_image_paths(self) -> List[Path]:
        """
        Collect all valid image file paths from the root directory.
        
        Returns:
            List of Path objects pointing to image files.
        """
        image_paths = []
        
        for ext in self.extensions:
            # Case-insensitive matching
            image_paths.extend(self.root_dir.glob(f'*{ext}'))
            image_paths.extend(self.root_dir.glob(f'*{ext.upper()}'))
        
        # Sort for reproducibility
        image_paths = sorted(set(image_paths))
        
        return image_paths
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single image by index.
        
        Args:
            idx: Index of the image to retrieve.
            
        Returns:
            Transformed image as a torch.Tensor.
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        img_path = self.image_paths[idx]
        
        try:
            # Load image in grayscale mode
            image = Image.open(img_path).convert('L')
            
            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)
            else:
                # Default: convert to tensor
                image = transforms.ToTensor()(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    
    def get_image_path(self, idx: int) -> Path:
        """
        Get the file path for an image by index.
        
        Args:
            idx: Index of the image.
            
        Returns:
            Path to the image file.
        """
        return self.image_paths[idx]


def get_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    rotation_degrees: float = 5.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    horizontal_flip: bool = False,
    normalize_range: Tuple[float, float] = (-1.0, 1.0)
) -> transforms.Compose:
    """
    Create training data augmentation transforms.
    
    Args:
        image_size: Target image size (square).
        rotation_degrees: Maximum rotation angle in degrees (±).
        scale_range: Tuple of (min_scale, max_scale) for random scaling.
        horizontal_flip: If True, apply random horizontal flip.
        normalize_range: Output normalization range.
        
    Returns:
        Composed transform pipeline.
    """
    transform_list = []
    
    # Resize to target size
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Random rotation (±degrees)
    if rotation_degrees > 0:
        transform_list.append(
            transforms.RandomRotation(
                degrees=rotation_degrees,
                fill=255  # White background for signatures
            )
        )
    
    # Random scaling via affine transform
    if scale_range != (1.0, 1.0):
        transform_list.append(
            transforms.RandomAffine(
                degrees=0,
                scale=scale_range,
                fill=255
            )
        )
    
    # Optional horizontal flip (use with caution for signatures)
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Convert to tensor (scales to [0, 1])
    transform_list.append(transforms.ToTensor())
    
    # Normalize to target range (default [-1, 1] for tanh output)
    min_val, max_val = normalize_range
    if normalize_range != (0.0, 1.0):
        # ToTensor gives [0, 1], we need to scale to [min_val, max_val]
        # Formula: x' = x * (max - min) + min
        # For [-1, 1]: x' = x * 2 - 1
        transform_list.append(
            transforms.Normalize(
                mean=[0.5],  # Grayscale has 1 channel
                std=[0.5]    # This maps [0, 1] to [-1, 1]
            )
        )
    
    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    normalize_range: Tuple[float, float] = (-1.0, 1.0)
) -> transforms.Compose:
    """
    Create validation transforms (no augmentation).
    
    Args:
        image_size: Target image size (square).
        normalize_range: Output normalization range.
        
    Returns:
        Composed transform pipeline.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    # Normalize to target range
    if normalize_range != (0.0, 1.0):
        transform_list.append(
            transforms.Normalize(mean=[0.5], std=[0.5])
        )
    
    return transforms.Compose(transform_list)


def create_data_loader(
    data_dir: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    shuffle: bool = True,
    augment: bool = True,
    rotation_degrees: float = 5.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    horizontal_flip: bool = False,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a DataLoader for signature images.
    
    Factory function that creates a complete data loading pipeline
    with configurable augmentation.
    
    Args:
        data_dir: Path to directory containing images.
        batch_size: Number of images per batch (default: 64).
        num_workers: Number of data loading workers (default: 4).
        image_size: Target image size (default: 64).
        shuffle: Whether to shuffle data each epoch.
        augment: Whether to apply data augmentation.
        rotation_degrees: Max rotation for augmentation (default: ±5°).
        scale_range: Scale range for augmentation.
        horizontal_flip: Whether to apply horizontal flip.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        
    Returns:
        Configured DataLoader instance.
    """
    # Select transform based on augmentation flag
    if augment:
        transform = get_train_transforms(
            image_size=image_size,
            rotation_degrees=rotation_degrees,
            scale_range=scale_range,
            horizontal_flip=horizontal_flip
        )
    else:
        transform = get_val_transforms(image_size=image_size)
    
    # Create dataset
    dataset = SignatureDataset(
        root_dir=data_dir,
        transform=transform
    )
    
    # Adjust num_workers for Windows compatibility
    if os.name == 'nt' and num_workers > 0:
        # Windows has issues with multiple workers sometimes
        num_workers = min(num_workers, 4)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last
    )
    
    logger.info(
        f"Created DataLoader: {len(dataset)} images, "
        f"batch_size={batch_size}, workers={num_workers}"
    )
    
    return loader


def create_train_val_loaders(
    data_dir: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    val_split: float = DEFAULT_VAL_SPLIT,
    rotation_degrees: float = 5.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    horizontal_flip: bool = False,
    seed: int = 42,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders with split.
    
    Args:
        data_dir: Path to directory containing images.
        batch_size: Number of images per batch.
        num_workers: Number of data loading workers.
        image_size: Target image size.
        val_split: Fraction of data for validation (default: 0.1).
        rotation_degrees: Max rotation for training augmentation.
        scale_range: Scale range for training augmentation.
        horizontal_flip: Whether to apply horizontal flip.
        seed: Random seed for reproducible splits.
        pin_memory: Whether to pin memory for GPU transfer.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Create full dataset with training transforms
    train_transform = get_train_transforms(
        image_size=image_size,
        rotation_degrees=rotation_degrees,
        scale_range=scale_range,
        horizontal_flip=horizontal_flip
    )
    
    val_transform = get_val_transforms(image_size=image_size)
    
    # Create two datasets with different transforms
    full_dataset = SignatureDataset(root_dir=data_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # Get indices for split
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets with appropriate transforms
    train_dataset = _SubsetWithTransform(full_dataset, train_indices, train_transform)
    val_dataset = _SubsetWithTransform(full_dataset, val_indices, val_transform)
    
    # Adjust num_workers for Windows
    if os.name == 'nt' and num_workers > 0:
        num_workers = min(num_workers, 4)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    logger.info(
        f"Created train/val split: {train_size} training, {val_size} validation"
    )
    
    return train_loader, val_loader


class _SubsetWithTransform(Dataset):
    """
    Subset of a dataset with a specific transform.
    
    Helper class to apply different transforms to train/val splits.
    """
    
    def __init__(
        self,
        dataset: SignatureDataset,
        indices: List[int],
        transform: Callable
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        original_idx = self.indices[idx]
        img_path = self.dataset.get_image_path(original_idx)
        
        # Load image
        image = Image.open(img_path).convert('L')
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image


def get_sample_batch(
    data_loader: DataLoader,
    num_samples: Optional[int] = None
) -> torch.Tensor:
    """
    Get a sample batch from the DataLoader for visualization.
    
    Args:
        data_loader: DataLoader to sample from.
        num_samples: Optional number of samples (default: one full batch).
        
    Returns:
        Tensor of shape (N, 1, H, W) with sample images.
    """
    # Get one batch
    batch = next(iter(data_loader))
    
    if num_samples is not None:
        batch = batch[:num_samples]
    
    return batch


def denormalize_batch(
    batch: torch.Tensor,
    input_range: Tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Denormalize a batch from [-1, 1] to [0, 1] for visualization.
    
    Args:
        batch: Tensor of shape (N, C, H, W) in [-1, 1] range.
        input_range: Current normalization range.
        
    Returns:
        Tensor in [0, 1] range.
    """
    min_val, max_val = input_range
    # Scale from [min_val, max_val] to [0, 1]
    denorm = (batch - min_val) / (max_val - min_val)
    return denorm.clamp(0, 1)


def batch_to_grid(
    batch: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert a batch of images to a grid for visualization.
    
    Args:
        batch: Tensor of shape (N, 1, H, W).
        nrow: Number of images per row.
        padding: Padding between images.
        normalize: If True, denormalize from [-1, 1].
        
    Returns:
        Numpy array representing the image grid.
    """
    from torchvision.utils import make_grid
    
    if normalize:
        batch = denormalize_batch(batch)
    
    # Create grid
    grid = make_grid(batch, nrow=nrow, padding=padding, normalize=False)
    
    # Convert to numpy (H, W, C)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Scale to [0, 255] for image display
    grid_np = (grid_np * 255).astype(np.uint8)
    
    # Squeeze channel dimension if grayscale
    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)
    
    return grid_np


def save_sample_grid(
    batch: torch.Tensor,
    save_path: Union[str, Path],
    nrow: int = 8,
    normalize: bool = True
) -> None:
    """
    Save a batch of images as a grid image.
    
    Args:
        batch: Tensor of shape (N, 1, H, W).
        save_path: Path to save the grid image.
        nrow: Number of images per row.
        normalize: If True, denormalize from [-1, 1].
    """
    from PIL import Image
    
    grid = batch_to_grid(batch, nrow=nrow, normalize=normalize)
    
    # Save
    img = Image.fromarray(grid)
    img.save(save_path)
    
    logger.info(f"Saved sample grid to {save_path}")


def get_dataset_stats(data_loader: DataLoader) -> dict:
    """
    Calculate statistics over the dataset.
    
    Args:
        data_loader: DataLoader to analyze.
        
    Returns:
        Dictionary containing mean, std, min, max statistics.
    """
    all_values = []
    
    for batch in data_loader:
        all_values.append(batch.flatten())
    
    all_values = torch.cat(all_values)
    
    stats = {
        'mean': all_values.mean().item(),
        'std': all_values.std().item(),
        'min': all_values.min().item(),
        'max': all_values.max().item(),
        'num_samples': len(data_loader.dataset)
    }
    
    return stats


# Convenience function for common use case
def get_signature_dataloader(
    data_dir: str = 'data/signatures/train/',
    batch_size: int = 64,
    image_size: int = 64,
    augment: bool = True
) -> DataLoader:
    """
    Convenience function to create a signature DataLoader.
    
    This is the main entry point for loading signature data
    with sensible defaults.
    
    Args:
        data_dir: Path to training data directory.
        batch_size: Batch size (default: 64).
        image_size: Target image size (default: 64).
        augment: Whether to apply augmentation.
        
    Returns:
        Configured DataLoader ready for training.
    """
    return create_data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        augment=augment,
        rotation_degrees=5.0,
        scale_range=(0.95, 1.05),
        horizontal_flip=False  # Disabled by default for signatures
    )


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test signature data loader')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/signatures/train/',
        help='Path to signature images directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=64,
        help='Target image size'
    )
    parser.add_argument(
        '--save-sample',
        type=str,
        default=None,
        help='Path to save sample grid image'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("Testing Signature DataLoader")
    print(f"{'='*50}")
    
    # Create data loader
    try:
        loader = get_signature_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            augment=True
        )
        
        print(f"\nDataset size: {len(loader.dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches: {len(loader)}")
        
        # Get sample batch
        sample = get_sample_batch(loader)
        print(f"\nSample batch shape: {sample.shape}")
        print(f"Value range: [{sample.min():.2f}, {sample.max():.2f}]")
        
        # Calculate stats
        print("\nCalculating dataset statistics...")
        stats = get_dataset_stats(loader)
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Std: {stats['std']:.4f}")
        print(f"Min: {stats['min']:.4f}")
        print(f"Max: {stats['max']:.4f}")
        
        # Save sample if requested
        if args.save_sample:
            save_sample_grid(sample, args.save_sample, nrow=4)
            print(f"\nSaved sample grid to: {args.save_sample}")
        
        print(f"\n{'='*50}")
        print("DataLoader test completed successfully!")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the data directory exists and contains images.")
