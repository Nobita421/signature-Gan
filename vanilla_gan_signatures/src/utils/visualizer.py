"""
Visualization utilities for GAN training and evaluation.

Provides functions for plotting training curves, creating image grids,
comparing real vs fake samples, visualizing latent interpolations,
and generating training progress GIFs.
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


def plot_training_curves(
    log_file: Union[str, Path],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot generator and discriminator loss curves from training log.

    Args:
        log_file: Path to JSON log file containing training metrics.
        save_path: Path to save the generated plot.
        figsize: Figure size as (width, height).

    Raises:
        FileNotFoundError: If log file does not exist.
        ValueError: If log file format is invalid.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    # Load training logs
    epochs = []
    g_losses = []
    d_losses = []

    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if 'epoch' in entry:
                    epochs.append(entry['epoch'])
                    g_losses.append(entry.get('g_loss', entry.get('generator_loss', 0)))
                    d_losses.append(entry.get('d_loss', entry.get('discriminator_loss', 0)))
            except json.JSONDecodeError:
                continue

    if not epochs:
        raise ValueError(f"No valid training data found in {log_file}")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot Generator Loss
    axes[0].plot(epochs, g_losses, 'b-', linewidth=1.5, label='Generator Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Generator Loss', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot Discriminator Loss
    axes[1].plot(epochs, d_losses, 'r-', linewidth=1.5, label='Discriminator Loss')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Discriminator Loss', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Ensure save directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = (-1, 1),
    padding: int = 2,
    pad_value: float = 0
) -> torch.Tensor:
    """
    Create a grid of images from a batch tensor.

    Args:
        images: Batch of images as tensor (N, C, H, W).
        nrow: Number of images per row in the grid.
        normalize: If True, normalize images to [0, 1] range.
        value_range: Original value range for normalization.
        padding: Padding between images in pixels.
        pad_value: Value for padding pixels.

    Returns:
        Grid tensor of shape (C, H_grid, W_grid).
    """
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)

    # Ensure images are on CPU for visualization
    if images.is_cuda:
        images = images.cpu()

    # Create grid using torchvision
    grid = make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=padding,
        pad_value=pad_value
    )

    return grid


def save_sample_grid(
    images: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = (-1, 1)
) -> None:
    """
    Save a grid of generated samples to disk.

    Args:
        images: Batch of images as tensor (N, C, H, W).
        path: Output path for the saved image.
        nrow: Number of images per row in the grid.
        normalize: If True, normalize images to [0, 1] range.
        value_range: Original value range for normalization.
    """
    # Create grid
    grid = create_image_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range
    )

    # Convert to numpy for saving
    grid_np = grid.permute(1, 2, 0).numpy()

    # Handle grayscale images
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)

    # Scale to 0-255 range
    grid_np = (grid_np * 255).astype(np.uint8)

    # Save using PIL
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if grid_np.ndim == 2:
        img = Image.fromarray(grid_np, mode='L')
    else:
        img = Image.fromarray(grid_np, mode='RGB')

    img.save(path)


def plot_real_vs_fake(
    real_batch: torch.Tensor,
    fake_batch: torch.Tensor,
    save_path: Union[str, Path],
    nrow: int = 8,
    figsize: Tuple[int, int] = (14, 7)
) -> None:
    """
    Create side-by-side comparison of real and fake samples.

    Args:
        real_batch: Batch of real images (N, C, H, W).
        fake_batch: Batch of generated fake images (N, C, H, W).
        save_path: Path to save the comparison plot.
        nrow: Number of images per row in each grid.
        figsize: Figure size as (width, height).
    """
    # Create grids for real and fake images
    real_grid = create_image_grid(real_batch, nrow=nrow)
    fake_grid = create_image_grid(fake_batch, nrow=nrow)

    # Convert to numpy
    real_np = real_grid.permute(1, 2, 0).numpy()
    fake_np = fake_grid.permute(1, 2, 0).numpy()

    # Handle grayscale
    if real_np.shape[2] == 1:
        real_np = real_np.squeeze(2)
        fake_np = fake_np.squeeze(2)
        cmap = 'gray'
    else:
        cmap = None

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot real images
    axes[0].imshow(real_np, cmap=cmap)
    axes[0].set_title('Real Samples', fontsize=16, fontweight='bold')
    axes[0].axis('off')

    # Plot fake images
    axes[1].imshow(fake_np, cmap=cmap)
    axes[1].set_title('Generated Samples', fontsize=16, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_interpolation_grid(
    images: torch.Tensor,
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (15, 3)
) -> None:
    """
    Create and save a visualization of latent space interpolation.

    Args:
        images: Tensor of interpolated images (N, C, H, W).
                Expected to be a sequence from one latent point to another.
        save_path: Path to save the interpolation visualization.
        figsize: Figure size as (width, height).
    """
    n_images = images.shape[0]

    # Create grid with all images in one row
    grid = create_image_grid(images, nrow=n_images, padding=4)

    # Convert to numpy
    grid_np = grid.permute(1, 2, 0).numpy()

    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
        cmap = 'gray'
    else:
        cmap = None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(grid_np, cmap=cmap)
    ax.set_title('Latent Space Interpolation', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add start/end labels
    ax.annotate('Start', xy=(0.02, -0.05), xycoords='axes fraction',
                fontsize=10, ha='left', va='top')
    ax.annotate('End', xy=(0.98, -0.05), xycoords='axes fraction',
                fontsize=10, ha='right', va='top')

    plt.tight_layout()

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_training_gif(
    sample_dir: Union[str, Path],
    output_path: Union[str, Path],
    duration: int = 200,
    loop: int = 0,
    pattern: str = "*.png"
) -> None:
    """
    Create an animated GIF from training sample images.

    Args:
        sample_dir: Directory containing sample images from training.
        output_path: Path to save the output GIF.
        duration: Duration of each frame in milliseconds.
        loop: Number of loops (0 = infinite loop).
        pattern: Glob pattern to match image files.

    Raises:
        FileNotFoundError: If sample directory does not exist.
        ValueError: If no images are found in the directory.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Find all matching images
    image_paths = sorted(glob.glob(str(sample_dir / pattern)))

    # Also try common patterns if no matches
    if not image_paths:
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths = sorted(glob.glob(str(sample_dir / ext)))
            if image_paths:
                break

    if not image_paths:
        raise ValueError(f"No images found in {sample_dir}")

    # Load images
    frames: List[Image.Image] = []
    for img_path in image_paths:
        img = Image.open(img_path)
        # Convert to RGB if necessary for GIF compatibility
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('P')
        frames.append(img)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )


def plot_loss_comparison(
    log_files: List[Union[str, Path]],
    labels: List[str],
    save_path: Union[str, Path],
    loss_type: str = 'g_loss',
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Compare loss curves from multiple training runs.

    Args:
        log_files: List of paths to log files.
        labels: Labels for each training run.
        save_path: Path to save the comparison plot.
        loss_type: Which loss to plot ('g_loss' or 'd_loss').
        figsize: Figure size as (width, height).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))

    for log_file, label, color in zip(log_files, labels, colors):
        epochs = []
        losses = []

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if 'epoch' in entry:
                        epochs.append(entry['epoch'])
                        losses.append(entry.get(loss_type, 0))
                except json.JSONDecodeError:
                    continue

        if epochs:
            ax.plot(epochs, losses, linewidth=1.5, label=label, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{loss_type.replace("_", " ").title()} Comparison', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_generator_progress(
    sample_images: List[torch.Tensor],
    epoch_labels: List[int],
    save_path: Union[str, Path],
    nrow: int = 8,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize generator improvement over training epochs.

    Args:
        sample_images: List of sample batches from different epochs.
        epoch_labels: Corresponding epoch numbers.
        save_path: Path to save the visualization.
        nrow: Number of images per row in each grid.
        figsize: Figure size as (width, height).
    """
    n_epochs = len(sample_images)
    fig, axes = plt.subplots(n_epochs, 1, figsize=figsize)

    if n_epochs == 1:
        axes = [axes]

    for ax, images, epoch in zip(axes, sample_images, epoch_labels):
        grid = create_image_grid(images, nrow=nrow)
        grid_np = grid.permute(1, 2, 0).numpy()

        if grid_np.shape[2] == 1:
            grid_np = grid_np.squeeze(2)
            cmap = 'gray'
        else:
            cmap = None

        ax.imshow(grid_np, cmap=cmap)
        ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
