"""
Signature Generation Inference Script.

Generate synthetic signature images from a trained Vanilla GAN generator.

Features:
- Load trained generator from checkpoint
- Generate N synthetic signatures with batched inference
- Save as PNG images with sequential naming
- Support for random seed for reproducibility
- Support for different model checkpoints (person-specific if available)

Usage:
    python generate_signatures.py --checkpoint ./checkpoints/G_final.pth --n_samples 100 --output_dir ./generated
    python generate_signatures.py --checkpoint ./checkpoints/G_final.pth --n_samples 50 --seed 42
    python generate_signatures.py --checkpoint ./checkpoints/person_01/G_final.pth --n_samples 200 --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from inference utils
from utils.inference import (
    load_generator as load_generator_inference,
    generate_signatures_batch,
    DEFAULT_LATENT_DIM,
    DEFAULT_IMAGE_SIZE
)
from generator_vanilla_gan import Generator








def generate_signatures(
    generator: Generator,
    n_samples: int,
    output_dir: str,
    batch_size: int = 64,
    device: torch.device = torch.device('cpu'),
    seed: Optional[int] = None,
    prefix: str = "signature"
) -> None:
    """
    Generate synthetic signature images and save as PNG files.
    
    Args:
        generator: Trained generator model in eval mode
        n_samples: Number of signatures to generate
        output_dir: Directory to save generated images
        batch_size: Batch size for generation (default: 64)
        device: Device to run inference on
        seed: Optional random seed for reproducibility
        prefix: Filename prefix for generated images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Generating {n_samples} signatures...")
    
    # Use inference utility
    images = generate_signatures_batch(
        generator=generator,
        n_samples=n_samples,
        latent_dim=generator.latent_dim,
        device=device,
        seed=seed,
        batch_size=batch_size
    )
    
    # Save images
    print(f"Saving {len(images)} images...")
    for i, img in enumerate(tqdm(images, desc="Saving")):
        filename = f"{prefix}_{i + 1:06d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, 'PNG')
    
    print(f"\nGeneration complete!")
    print(f"Generated {len(images)} signatures saved to: {output_dir}")


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract and display information from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        return {"error": f"Checkpoint not found: {checkpoint_path}"}
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    info = {
        "path": checkpoint_path,
        "type": type(checkpoint).__name__
    }
    
    if isinstance(checkpoint, dict):
        info["keys"] = list(checkpoint.keys())
        
        if 'epoch' in checkpoint:
            info["epoch"] = checkpoint['epoch']
        if 'config' in checkpoint:
            info["config"] = checkpoint['config']
        if 'g_loss' in checkpoint:
            info["g_loss"] = checkpoint['g_loss']
        if 'd_loss' in checkpoint:
            info["d_loss"] = checkpoint['d_loss']
    
    return info


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic signatures using a trained Vanilla GAN generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the generator checkpoint file (e.g., G_final.pth)'
    )
    
    # Generation parameters
    parser.add_argument(
        '--n_samples',
        type=int,
        default=100,
        help='Number of signatures to generate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./generated_signatures',
        help='Output directory for generated images'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for generation (adjust based on GPU memory)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    # Additional options
    parser.add_argument(
        '--prefix',
        type=str,
        default='signature',
        help='Filename prefix for generated images'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display checkpoint information and exit'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for signature generation."""
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # If info flag, just display checkpoint info and exit
    if args.info:
        info = get_checkpoint_info(args.checkpoint)
        print("\nCheckpoint Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return
    
    # Load generator
    generator, _ = load_generator_inference(args.checkpoint, device)
    
    # Generate signatures
    generate_signatures(
        generator=generator,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        prefix=args.prefix
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Generation Summary:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples generated: {args.n_samples}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Seed: {args.seed if args.seed is not None else 'Random'}")
    print(f"  Device: {device}")
    print("=" * 50)


if __name__ == "__main__":
    main()
