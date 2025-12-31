"""
GAN Evaluation Script for Signature Generation.

This module provides comprehensive evaluation tools for trained Vanilla GAN models:
- Load trained generator from checkpoint
- Generate samples for visual inspection
- Compute quality metrics (FID, LPIPS diversity)
- Analyze stroke density and foreground ratio distributions
- Export evaluation reports and sample grids

Usage:
    python evaluate_vanilla_gan_signatures.py --checkpoint checkpoints/best_model.pt \
        --n_samples 500 --real_dir data/signatures/train --output_dir figures/eval
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vanilla_gan_model import VanillaGAN
from generator_vanilla_gan import Generator
from utils.metrics import (
    calculate_fid,
    calculate_lpips_diversity,
    calculate_stroke_density,
    calculate_foreground_ratio,
    INCEPTION_AVAILABLE,
    LPIPS_AVAILABLE
)
from utils.visualizer import save_sample_grid, create_image_grid


def load_generator_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device
) -> Tuple[Generator, Dict[str, Any]]:
    """
    Load trained generator from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        device: Device to load the model on
    
    Returns:
        Tuple of (Generator model, checkpoint config dict)
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        KeyError: If checkpoint is missing required keys
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 100)
    image_size = config.get('image_size', 64)
    image_channels = config.get('image_channels', 1)
    
    # Create generator with same architecture
    generator = Generator(
        latent_dim=latent_dim,
        output_size=image_size,
        output_channels=image_channels
    ).to(device)
    
    # Load weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"Generator loaded successfully:")
    print(f"  - Latent dim: {latent_dim}")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Epoch trained: {config.get('current_epoch', 'N/A')}")
    
    return generator, config


@torch.no_grad()
def generate_samples(
    generator: Generator,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 64
) -> torch.Tensor:
    """
    Generate N samples from the trained generator.
    
    Args:
        generator: Trained generator model
        n_samples: Number of samples to generate
        latent_dim: Dimension of latent vector
        device: Computation device
        batch_size: Batch size for generation
    
    Returns:
        Generated images tensor (N, C, H, W) in range [-1, 1]
    """
    generator.eval()
    all_samples: List[torch.Tensor] = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Generating {n_samples} samples...")
    for i in range(n_batches):
        current_batch_size = min(batch_size, n_samples - i * batch_size)
        z = torch.randn(current_batch_size, latent_dim, device=device)
        fake_images = generator(z)
        all_samples.append(fake_images.cpu())
        
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  Generated {min((i + 1) * batch_size, n_samples)}/{n_samples} samples")
    
    samples = torch.cat(all_samples, dim=0)
    return samples


def load_real_images(
    real_dir: Path,
    n_images: int,
    image_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Load real signature images for comparison metrics.
    
    Args:
        real_dir: Directory containing real signature images
        n_images: Number of images to load
        image_size: Target image size
        device: Computation device
    
    Returns:
        Real images tensor (N, 1, H, W) in range [-1, 1]
    """
    if not real_dir.exists():
        raise FileNotFoundError(f"Real images directory not found: {real_dir}")
    
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(real_dir.glob(f'*{ext}'))
        image_paths.extend(real_dir.glob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {real_dir}")
    
    # Sample subset if needed
    if len(image_paths) > n_images:
        indices = np.random.choice(len(image_paths), n_images, replace=False)
        image_paths = [image_paths[i] for i in indices]
    
    print(f"Loading {len(image_paths)} real images from {real_dir}")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('L')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")
    
    if len(images) == 0:
        raise ValueError("No images could be loaded successfully")
    
    return torch.stack(images)


def create_sample_grids(
    samples: torch.Tensor,
    output_dir: Path,
    n_grids: int = 3,
    grid_size: int = 64
) -> List[Path]:
    """
    Create and save multiple sample grids for visual inspection.
    
    Args:
        samples: Generated samples tensor (N, C, H, W)
        output_dir: Directory to save grids
        n_grids: Number of grids to create
        grid_size: Number of samples per grid
    
    Returns:
        List of paths to saved grid images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(n_grids):
        start_idx = i * grid_size
        end_idx = min(start_idx + grid_size, len(samples))
        
        if start_idx >= len(samples):
            break
        
        grid_samples = samples[start_idx:end_idx]
        nrow = int(np.sqrt(len(grid_samples)))
        
        grid_path = output_dir / f"sample_grid_{timestamp}_{i + 1}.png"
        save_sample_grid(
            grid_samples,
            grid_path,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1)
        )
        saved_paths.append(grid_path)
        print(f"  Saved grid: {grid_path}")
    
    return saved_paths


def compute_metrics(
    fake_images: torch.Tensor,
    real_images: Optional[torch.Tensor],
    device: torch.device
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for generated images.
    
    Args:
        fake_images: Generated images tensor (N, C, H, W)
        real_images: Real images tensor for FID computation (optional)
        device: Computation device
    
    Returns:
        Dictionary containing all computed metrics
    """
    metrics: Dict[str, Any] = {
        'n_samples': len(fake_images),
        'image_shape': list(fake_images.shape[1:]),
        'metrics_computed_at': datetime.now().isoformat()
    }
    
    # FID Score (requires real images)
    if real_images is not None and INCEPTION_AVAILABLE:
        print("Computing FID score...")
        try:
            fid_score = calculate_fid(real_images, fake_images, device)
            metrics['fid_score'] = fid_score
            print(f"  FID Score: {fid_score:.4f}")
        except Exception as e:
            print(f"  Warning: FID computation failed: {e}")
            metrics['fid_score'] = None
            metrics['fid_error'] = str(e)
    else:
        if not INCEPTION_AVAILABLE:
            print("  Skipping FID: torchvision not available")
            metrics['fid_score'] = None
            metrics['fid_error'] = "torchvision not available"
        elif real_images is None:
            print("  Skipping FID: no real images provided")
            metrics['fid_score'] = None
            metrics['fid_error'] = "no real images provided"
    
    # LPIPS Diversity
    if LPIPS_AVAILABLE:
        print("Computing LPIPS diversity...")
        try:
            # Use subset for LPIPS (computationally expensive)
            n_lpips = min(100, len(fake_images))
            lpips_samples = [fake_images[i] for i in range(n_lpips)]
            lpips_diversity = calculate_lpips_diversity(lpips_samples, device)
            metrics['lpips_diversity'] = lpips_diversity
            print(f"  LPIPS Diversity: {lpips_diversity:.4f}")
        except Exception as e:
            print(f"  Warning: LPIPS computation failed: {e}")
            metrics['lpips_diversity'] = None
            metrics['lpips_error'] = str(e)
    else:
        print("  Skipping LPIPS: lpips package not available")
        metrics['lpips_diversity'] = None
        metrics['lpips_error'] = "lpips package not available"
    
    # Stroke Density Distribution
    print("Computing stroke density distribution...")
    try:
        stroke_density = calculate_stroke_density(fake_images, threshold=0.5)
        metrics['stroke_density'] = stroke_density
        print(f"  Stroke Density - Mean: {stroke_density['mean']:.4f}, "
              f"Std: {stroke_density['std']:.4f}")
    except Exception as e:
        print(f"  Warning: Stroke density computation failed: {e}")
        metrics['stroke_density'] = None
        metrics['stroke_density_error'] = str(e)
    
    # Foreground Ratio Statistics
    print("Computing foreground ratio statistics...")
    try:
        foreground_ratio = calculate_foreground_ratio(fake_images, threshold=0.5)
        metrics['foreground_ratio'] = foreground_ratio
        print(f"  Foreground Ratio - Mean: {foreground_ratio['mean']:.4f}, "
              f"Std: {foreground_ratio['std']:.4f}")
    except Exception as e:
        print(f"  Warning: Foreground ratio computation failed: {e}")
        metrics['foreground_ratio'] = None
        metrics['foreground_ratio_error'] = str(e)
    
    # Compare with real images if available
    if real_images is not None:
        print("Computing real image statistics for comparison...")
        try:
            real_stroke = calculate_stroke_density(real_images, threshold=0.5)
            real_foreground = calculate_foreground_ratio(real_images, threshold=0.5)
            metrics['real_stroke_density'] = real_stroke
            metrics['real_foreground_ratio'] = real_foreground
            print(f"  Real Stroke Density - Mean: {real_stroke['mean']:.4f}")
            print(f"  Real Foreground Ratio - Mean: {real_foreground['mean']:.4f}")
        except Exception as e:
            print(f"  Warning: Real image statistics failed: {e}")
    
    return metrics


def save_evaluation_report(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    grid_paths: List[Path]
) -> Path:
    """
    Save comprehensive evaluation report as JSON.
    
    Args:
        metrics: Computed metrics dictionary
        config: Model configuration
        output_dir: Output directory
        checkpoint_path: Path to evaluated checkpoint
        grid_paths: Paths to saved sample grids
    
    Returns:
        Path to saved report file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'evaluation_info': {
            'checkpoint': str(checkpoint_path),
            'evaluation_timestamp': datetime.now().isoformat(),
            'sample_grids': [str(p) for p in grid_paths]
        },
        'model_config': config,
        'metrics': metrics,
        'summary': {
            'fid_score': metrics.get('fid_score'),
            'lpips_diversity': metrics.get('lpips_diversity'),
            'stroke_density_mean': metrics.get('stroke_density', {}).get('mean'),
            'foreground_ratio_mean': metrics.get('foreground_ratio', {}).get('mean'),
            'n_samples_evaluated': metrics.get('n_samples')
        }
    }
    
    report_path = output_dir / f"evaluation_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nEvaluation report saved to: {report_path}")
    return report_path


def print_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\nSamples Evaluated: {metrics.get('n_samples', 'N/A')}")
    print(f"Image Shape: {metrics.get('image_shape', 'N/A')}")
    
    print("\n--- Quality Metrics ---")
    fid = metrics.get('fid_score')
    if fid is not None:
        print(f"FID Score: {fid:.4f} (lower is better)")
    else:
        print(f"FID Score: Not computed - {metrics.get('fid_error', 'unknown reason')}")
    
    lpips = metrics.get('lpips_diversity')
    if lpips is not None:
        print(f"LPIPS Diversity: {lpips:.4f} (higher = more diverse)")
    else:
        print(f"LPIPS Diversity: Not computed - {metrics.get('lpips_error', 'unknown reason')}")
    
    print("\n--- Stroke Analysis ---")
    stroke = metrics.get('stroke_density')
    if stroke:
        print(f"Stroke Density:")
        print(f"  Mean: {stroke['mean']:.4f}")
        print(f"  Std:  {stroke['std']:.4f}")
        print(f"  Range: [{stroke['min']:.4f}, {stroke['max']:.4f}]")
    
    print("\n--- Foreground Analysis ---")
    fg = metrics.get('foreground_ratio')
    if fg:
        print(f"Foreground Ratio:")
        print(f"  Mean: {fg['mean']:.4f}")
        print(f"  Std:  {fg['std']:.4f}")
        if 'percentiles' in fg:
            print(f"  Percentiles: 25%={fg['percentiles']['25']:.4f}, "
                  f"50%={fg['percentiles']['50']:.4f}, "
                  f"75%={fg['percentiles']['75']:.4f}")
    
    # Comparison with real images
    if 'real_stroke_density' in metrics:
        print("\n--- Comparison with Real Images ---")
        real_stroke = metrics['real_stroke_density']
        real_fg = metrics.get('real_foreground_ratio', {})
        print(f"Real Stroke Density Mean: {real_stroke['mean']:.4f} "
              f"(Generated: {stroke['mean']:.4f})")
        if real_fg:
            print(f"Real Foreground Ratio Mean: {real_fg['mean']:.4f} "
                  f"(Generated: {fg['mean']:.4f})")
    
    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Vanilla GAN for signature generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=500,
        help='Number of samples to generate for evaluation (500-1000 recommended)'
    )
    
    parser.add_argument(
        '--real_dir',
        type=str,
        default=None,
        help='Directory containing real signature images for FID computation'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures/evaluation',
        help='Directory to save evaluation outputs (grids and reports)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for sample generation'
    )
    
    parser.add_argument(
        '--n_grids',
        type=int,
        default=3,
        help='Number of sample grids to generate'
    )
    
    parser.add_argument(
        '--grid_size',
        type=int,
        default=64,
        help='Number of samples per grid'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detected if not specified.'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    real_dir = Path(args.real_dir) if args.real_dir else None
    
    try:
        # Load generator
        generator, config = load_generator_from_checkpoint(checkpoint_path, device)
        latent_dim = config.get('latent_dim', 100)
        image_size = config.get('image_size', 64)
        
        # Generate samples
        fake_images = generate_samples(
            generator=generator,
            n_samples=args.n_samples,
            latent_dim=latent_dim,
            device=device,
            batch_size=args.batch_size
        )
        
        # Create sample grids
        print("\nCreating sample grids...")
        grid_paths = create_sample_grids(
            samples=fake_images,
            output_dir=output_dir,
            n_grids=args.n_grids,
            grid_size=args.grid_size
        )
        
        # Load real images if provided
        real_images = None
        if real_dir:
            try:
                real_images = load_real_images(
                    real_dir=real_dir,
                    n_images=args.n_samples,
                    image_size=image_size,
                    device=device
                )
            except Exception as e:
                print(f"Warning: Could not load real images: {e}")
        
        # Compute metrics
        print("\nComputing evaluation metrics...")
        metrics = compute_metrics(fake_images, real_images, device)
        
        # Save evaluation report
        report_path = save_evaluation_report(
            metrics=metrics,
            config=config,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            grid_paths=grid_paths
        )
        
        # Print summary
        print_summary(metrics)
        
        print(f"\nEvaluation complete!")
        print(f"  Sample grids saved to: {output_dir}")
        print(f"  Report saved to: {report_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
