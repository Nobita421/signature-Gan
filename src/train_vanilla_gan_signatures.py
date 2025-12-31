"""
Vanilla GAN Training Script for Signature Generation.

Complete training pipeline with:
- Configurable hyperparameters via Config class
- Training loop with discriminator/generator alternation
- Label smoothing and gradient clipping for stability
- Mode collapse detection
- Checkpoint saving and resumption
- Fixed noise samples for visual comparison
- Progress tracking with tqdm
- Comprehensive logging

Usage:
    python train_vanilla_gan_signatures.py --data_dir ./data/signatures/train --epochs 200
    python train_vanilla_gan_signatures.py --resume --checkpoint_dir ./checkpoints
"""

import argparse
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader_signatures import create_data_loader
from vanilla_gan_model import VanillaGAN
from utils.logger import GANLogger
from utils.visualizer import save_sample_grid


@dataclass
class TrainingConfig:
    """
    Configuration class for GAN training hyperparameters.
    
    Attributes:
        latent_dim: Dimension of the latent noise vector z
        image_size: Size of generated images (64 or 128)
        batch_size: Number of images per training batch
        epochs: Total number of training epochs
        g_lr: Generator learning rate
        d_lr: Discriminator learning rate
        beta1: Adam optimizer beta1 parameter
        beta2: Adam optimizer beta2 parameter
        label_smoothing: Soft label value for real samples (0.9 = one-sided smoothing)
        gradient_clip_value: Max gradient norm for clipping (None to disable)
        n_critic: Number of discriminator updates per generator update
        sample_interval: Save sample grid every N epochs
        checkpoint_interval: Save model checkpoint every N epochs
        num_workers: DataLoader workers for parallel loading
        fixed_noise_samples: Number of fixed noise samples for comparison
        mode_collapse_threshold: Threshold for detecting mode collapse
        mode_collapse_window: Number of batches to track for mode collapse detection
    """
    # Model architecture
    latent_dim: int = 100
    image_size: int = 64
    image_channels: int = 1
    
    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 200
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Stabilization techniques
    label_smoothing: float = 0.9
    gradient_clip_value: Optional[float] = None  # e.g., 1.0 to enable
    n_critic: int = 1
    
    # Logging and saving
    sample_interval: int = 5
    checkpoint_interval: int = 10
    num_workers: int = 4
    
    # Fixed noise for visual tracking
    fixed_noise_samples: int = 64
    
    # Mode collapse detection
    mode_collapse_threshold: float = 0.1
    mode_collapse_window: int = 50
    
    # Paths (set via CLI)
    data_dir: str = ""
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    log_dir: str = "./logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return asdict(self)


class ModeCollapseDetector:
    """
    Detects potential mode collapse during GAN training.
    
    Monitors discriminator outputs and generator loss variance
    to identify when the generator produces limited variety.
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 50
    ) -> None:
        """
        Initialize mode collapse detector.
        
        Args:
            threshold: Variance threshold below which collapse is suspected
            window_size: Number of recent batches to analyze
        """
        self.threshold = threshold
        self.window_size = window_size
        self.g_losses: List[float] = []
        self.d_fake_outputs: List[float] = []
    
    def update(self, g_loss: float, d_fake_mean: float) -> None:
        """Update with latest batch metrics."""
        self.g_losses.append(g_loss)
        self.d_fake_outputs.append(d_fake_mean)
        
        # Keep only recent window
        if len(self.g_losses) > self.window_size:
            self.g_losses.pop(0)
            self.d_fake_outputs.pop(0)
    
    def check_collapse(self) -> Tuple[bool, str]:
        """
        Check for mode collapse indicators.
        
        Returns:
            Tuple of (is_collapsed, reason_message)
        """
        if len(self.g_losses) < self.window_size:
            return False, "Insufficient data"
        
        # Check 1: Very low variance in discriminator output on fakes
        d_fake_var = torch.tensor(self.d_fake_outputs).var().item()
        if d_fake_var < self.threshold * 0.1:
            return True, f"D(fake) variance too low: {d_fake_var:.6f}"
        
        # Check 2: Generator loss stuck at very low value
        g_loss_mean = sum(self.g_losses) / len(self.g_losses)
        g_loss_var = torch.tensor(self.g_losses).var().item()
        if g_loss_var < self.threshold and g_loss_mean < 0.5:
            return True, f"G_loss stuck: mean={g_loss_mean:.4f}, var={g_loss_var:.6f}"
        
        # Check 3: D(fake) consistently near 0.5 (D is confused/collapsed)
        d_fake_mean = sum(self.d_fake_outputs) / len(self.d_fake_outputs)
        if abs(d_fake_mean - 0.5) < 0.05 and d_fake_var < self.threshold:
            return True, f"D(fake) stuck at ~0.5: mean={d_fake_mean:.4f}"
        
        return False, "Training appears stable"
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.g_losses.clear()
        self.d_fake_outputs.clear()


class GANTrainer:
    """
    Complete training manager for Vanilla GAN signature generation.
    
    Handles the full training loop including:
    - Data loading and batching
    - Training step execution
    - Loss logging and visualization
    - Checkpoint management
    - Mode collapse monitoring
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        device: Optional[str] = None,
        stop_file: Optional[str] = None
    ) -> None:
        """
        Initialize the GAN trainer.
        
        Args:
            config: Training configuration
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Optional stop request file path (cross-platform cooperative stop)
        self.stop_file = Path(stop_file).resolve() if stop_file else None
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.sample_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = VanillaGAN(
            latent_dim=config.latent_dim,
            image_size=config.image_size,
            image_channels=config.image_channels,
            g_lr=config.g_lr,
            d_lr=config.d_lr,
            beta1=config.beta1,
            beta2=config.beta2,
            label_smoothing=config.label_smoothing,
            device=self.device
        )
        
        # Initialize logger
        self.logger = GANLogger(
            log_dir=config.log_dir,
            experiment_name="vanilla_gan_signatures"
        )
        self.logger.log_config(config.to_dict())
        
        # Initialize mode collapse detector
        self.collapse_detector = ModeCollapseDetector(
            threshold=config.mode_collapse_threshold,
            window_size=config.mode_collapse_window
        )
        
        # Fixed noise for consistent sample generation across epochs
        self.fixed_noise = torch.randn(
            config.fixed_noise_samples,
            config.latent_dim,
            device=self.device
        )
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_g_loss = float('inf')
        
        # Data loader (initialized in train())
        self.data_loader = None
        
        print(f"[Trainer] Initialized on device: {self.device}")
        print(f"[Trainer] Generator params: {sum(p.numel() for p in self.model.generator.parameters()):,}")
        print(f"[Trainer] Discriminator params: {sum(p.numel() for p in self.model.discriminator.parameters()):,}")

    def _stop_requested(self) -> bool:
        if self.stop_file is None:
            return False
        try:
            return self.stop_file.exists()
        except Exception:
            return False
    
    def _apply_gradient_clipping(self, model: nn.Module) -> Optional[float]:
        """
        Apply gradient clipping if enabled.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Total gradient norm before clipping, or None if disabled
        """
        if self.config.gradient_clip_value is None:
            return None
        
        total_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            self.config.gradient_clip_value
        )
        return total_norm.item()
    
    def _train_discriminator(
        self,
        real_batch: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train discriminator for one batch.
        
        Args:
            real_batch: Batch of real images
            
        Returns:
            Dictionary of discriminator metrics
        """
        self.model.discriminator.train()
        self.model.generator.eval()
        
        batch_size = real_batch.size(0)
        real_batch = real_batch.to(self.device)
        
        # Zero gradients
        self.model.d_optimizer.zero_grad()
        
        # Train on real images
        real_labels = torch.full(
            (batch_size, 1),
            self.config.label_smoothing,
            device=self.device
        )
        real_preds = self.model.discriminator(real_batch)
        d_loss_real = self.model.criterion(real_preds, real_labels)
        
        # Train on fake images
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        with torch.no_grad():
            fake_batch = self.model.generator(noise)
        
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_preds = self.model.discriminator(fake_batch)
        d_loss_fake = self.model.criterion(fake_preds, fake_labels)
        
        # Total loss and backward
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        
        # Gradient clipping
        grad_norm = self._apply_gradient_clipping(self.model.discriminator)
        
        self.model.d_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'd_real_mean': real_preds.mean().item(),
            'd_fake_mean': fake_preds.mean().item(),
            'd_grad_norm': grad_norm
        }
    
    def _train_generator(self, batch_size: int) -> Dict[str, float]:
        """
        Train generator for one batch.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Dictionary of generator metrics
        """
        self.model.generator.train()
        self.model.discriminator.eval()
        
        # Zero gradients
        self.model.g_optimizer.zero_grad()
        
        # Generate fake images
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_batch = self.model.generator(noise)
        
        # Generator wants discriminator to classify fakes as real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_preds = self.model.discriminator(fake_batch)
        g_loss = self.model.criterion(fake_preds, real_labels)
        
        # Backward
        g_loss.backward()
        
        # Gradient clipping
        grad_norm = self._apply_gradient_clipping(self.model.generator)
        
        self.model.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'g_fake_mean': fake_preds.mean().item(),
            'g_grad_norm': grad_norm
        }
    
    @torch.no_grad()
    def _generate_samples(self, epoch: int) -> None:
        """
        Generate and save sample grid using fixed noise.
        
        Args:
            epoch: Current epoch number
        """
        self.model.generator.eval()
        
        # Generate from fixed noise
        fake_samples = self.model.generator(self.fixed_noise)
        
        # Save grid
        sample_path = Path(self.config.sample_dir) / f"epoch_{epoch:04d}.png"
        save_sample_grid(
            images=fake_samples,
            path=sample_path,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
        print(f"[Trainer] Saved samples to {sample_path}")
    
    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: If True, also save as best model
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'g_optimizer_state_dict': self.model.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.model.d_optimizer.state_dict(),
            'config': self.config.to_dict(),
            'fixed_noise': self.fixed_noise.cpu(),
            'best_g_loss': self.best_g_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"[Trainer] Saved checkpoint: {checkpoint_path}")
        
        # Save latest checkpoint (for easy resumption)
        latest_path = Path(self.config.checkpoint_dir) / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[Trainer] New best model saved!")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> int:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file, or None to load latest
            
        Returns:
            Epoch to resume from
        """
        if checkpoint_path is None:
            checkpoint_path = Path(self.config.checkpoint_dir) / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"[Trainer] No checkpoint found at {checkpoint_path}")
            return 0
        
        print(f"[Trainer] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model states
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.model.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.model.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_g_loss = checkpoint.get('best_g_loss', float('inf'))
        
        # Load fixed noise for consistency
        if 'fixed_noise' in checkpoint:
            self.fixed_noise = checkpoint['fixed_noise'].to(self.device)
        
        print(f"[Trainer] Resumed from epoch {checkpoint['epoch']}")
        return self.start_epoch
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the full training loop.
        
        Returns:
            Dictionary with training summary and final metrics
        """
        print(f"[Trainer] Loading data from: {self.config.data_dir}")
        self.data_loader = create_data_loader(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            image_size=self.config.image_size,
            shuffle=True,
            augment=True,
            drop_last=True,
        )

        num_batches = len(self.data_loader)
        print(f"[Trainer] Data loaded: {num_batches} batches per epoch")

        if self.start_epoch == 0:
            self._generate_samples(epoch=0)

        stopped = False
        last_logged_epoch: Optional[int] = None
        last_avg_g_loss: Optional[float] = None
        last_avg_d_loss: Optional[float] = None

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self.model.current_epoch = epoch

                if self._stop_requested():
                    print("\n[Trainer] Stop requested. Exiting before starting next epoch...")
                    stopped = True
                    break

                epoch_d_loss = 0.0
                epoch_g_loss = 0.0
                epoch_d_real = 0.0
                epoch_d_fake = 0.0
                batches_seen = 0

                pbar = tqdm(
                    self.data_loader,
                    desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                    leave=True,
                    ncols=120,
                )

                for real_batch in pbar:
                    batch_size = real_batch.size(0)

                    d_metrics = None
                    for _ in range(self.config.n_critic):
                        d_metrics = self._train_discriminator(real_batch)

                    g_metrics = self._train_generator(batch_size)
                    self.global_step += 1

                    batches_seen += 1
                    epoch_d_loss += d_metrics['d_loss']
                    epoch_g_loss += g_metrics['g_loss']
                    epoch_d_real += d_metrics['d_real_mean']
                    epoch_d_fake += d_metrics['d_fake_mean']

                    self.collapse_detector.update(
                        g_loss=g_metrics['g_loss'],
                        d_fake_mean=d_metrics['d_fake_mean'],
                    )

                    pbar.set_postfix(
                        {
                            'D': f"{d_metrics['d_loss']:.4f}",
                            'G': f"{g_metrics['g_loss']:.4f}",
                            'D(r)': f"{d_metrics['d_real_mean']:.3f}",
                            'D(f)': f"{d_metrics['d_fake_mean']:.3f}",
                        }
                    )

                    if self._stop_requested():
                        print("\n[Trainer] Stop requested. Stopping after current batch...")
                        stopped = True
                        break

                if batches_seen == 0:
                    print("[Trainer] No batches processed for this epoch. Stopping.")
                    stopped = True
                    break

                avg_d_loss = epoch_d_loss / batches_seen
                avg_g_loss = epoch_g_loss / batches_seen
                avg_d_real = epoch_d_real / batches_seen
                avg_d_fake = epoch_d_fake / batches_seen

                last_logged_epoch = epoch + 1
                last_avg_g_loss = avg_g_loss
                last_avg_d_loss = avg_d_loss

                self.logger.log_metrics(
                    epoch=epoch + 1,
                    g_loss=avg_g_loss,
                    d_loss=avg_d_loss,
                    d_real=avg_d_real,
                    d_fake=avg_d_fake,
                )

                collapsed, reason = self.collapse_detector.check_collapse()
                if collapsed:
                    print(f"\n[WARNING] Potential mode collapse detected: {reason}")

                is_best = avg_g_loss < self.best_g_loss
                if is_best:
                    self.best_g_loss = avg_g_loss

                if (epoch + 1) % self.config.sample_interval == 0 or stopped:
                    self._generate_samples(epoch + 1)

                if (epoch + 1) % self.config.checkpoint_interval == 0 or stopped:
                    self._save_checkpoint(epoch + 1, is_best=is_best)

                if stopped:
                    break

        except KeyboardInterrupt:
            print("\n[Trainer] KeyboardInterrupt received. Stopping training...")
            stopped = True

        finally:
            # Always persist logs.
            try:
                self.logger.save_to_csv()
                self.logger.save_to_json()
            except Exception:
                pass

        summary = self.logger.get_summary()
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total Epochs: {summary.get('total_epochs', self.config.epochs)}")
        if last_avg_g_loss is not None:
            print(f"Final G Loss: {summary.get('final_g_loss', last_avg_g_loss):.4f}")
        if last_avg_d_loss is not None:
            print(f"Final D Loss: {summary.get('final_d_loss', last_avg_d_loss):.4f}")
        print(f"Best G Loss: {self.best_g_loss:.4f}")
        print("=" * 60)

        return summary


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train Vanilla GAN for Signature Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/signatures/train',
        help='Path to training data directory'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--sample_dir',
        type=str,
        default='./samples',
        help='Directory to save generated samples'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Directory to save training logs'
    )

    # Run management
    parser.add_argument(
        '--run_dir',
        type=str,
        default=None,
        help='If set, overrides checkpoint/sample/log dirs under this run directory'
    )
    parser.add_argument(
        '--stop_file',
        type=str,
        default=None,
        help='If set, training will stop when this file exists (cooperative stop)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=100,
        help='Dimension of latent noise vector'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=64,
        choices=[64, 128],
        help='Size of generated images'
    )
    
    # Learning rates
    parser.add_argument(
        '--g_lr',
        type=float,
        default=2e-4,
        help='Generator learning rate'
    )
    parser.add_argument(
        '--d_lr',
        type=float,
        default=2e-4,
        help='Discriminator learning rate'
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='Adam beta1 parameter'
    )
    
    # Stabilization
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=0.9,
        help='Label smoothing value for real samples'
    )
    parser.add_argument(
        '--gradient_clip',
        type=float,
        default=None,
        help='Gradient clipping value (None to disable)'
    )
    parser.add_argument(
        '--n_critic',
        type=int,
        default=1,
        help='Number of D updates per G update'
    )
    
    # Intervals
    parser.add_argument(
        '--sample_interval',
        type=int,
        default=5,
        help='Save sample grid every N epochs'
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to specific checkpoint to resume from'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda/cpu, None for auto)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loader workers'
    )
    
    return parser.parse_args()


def main() -> Dict[str, Any]:
    """
    Main entry point for GAN training.
    
    Parses CLI arguments, creates configuration, initializes trainer,
    and executes the training loop.
    
    Returns:
        Dictionary containing training summary with keys:
            - experiment_name: str
            - total_epochs: int
            - final_g_loss: float
            - final_d_loss: float
            - min_g_loss: float
            - min_d_loss: float
            - avg_g_loss: float
            - avg_d_loss: float
    """
    # Parse arguments
    args = parse_arguments()

    # If run_dir is provided, derive output directories under it.
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        args.checkpoint_dir = str(run_dir / 'checkpoints')
        args.sample_dir = str(run_dir / 'samples')
        args.log_dir = str(run_dir / 'logs')
    
    # Create configuration
    config = TrainingConfig(
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        beta1=args.beta1,
        label_smoothing=args.label_smoothing,
        gradient_clip_value=args.gradient_clip,
        n_critic=args.n_critic,
        sample_interval=args.sample_interval,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        log_dir=args.log_dir
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("Vanilla GAN Training - Signature Generation")
    print("="*60)
    print(f"Data Directory: {config.data_dir}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.image_size}x{config.image_size}")
    print(f"Latent Dim: {config.latent_dim}")
    print(f"G Learning Rate: {config.g_lr}")
    print(f"D Learning Rate: {config.d_lr}")
    print(f"Label Smoothing: {config.label_smoothing}")
    print(f"Gradient Clipping: {config.gradient_clip_value or 'Disabled'}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = GANTrainer(config=config, device=args.device, stop_file=args.stop_file)
    
    # Resume if requested
    if args.resume or args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Run training
    summary = trainer.train()
    
    return summary


if __name__ == "__main__":
    main()
