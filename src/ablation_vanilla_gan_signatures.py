"""
Ablation Study Script for Vanilla GAN Signature Generation.

This module implements a comprehensive ablation study to analyze the impact of:
- Different latent dimensions: [50, 100, 200]
- Different generator activations: ReLU vs LeakyReLU
- With/without spectral normalization in discriminator

For each configuration, the script:
- Trains a GAN for a fixed number of epochs
- Evaluates training stability (loss variance), visual quality, and FID score
- Saves comparison results and generates visualization plots

Usage:
    python ablation_vanilla_gan_signatures.py --data_dir ./data/signatures/train --epochs 50 --output_dir ./ablation_results
    python ablation_vanilla_gan_signatures.py --data_dir ./data/signatures/train --epochs 100 --output_dir ./ablation_results --batch_size 32
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader_signatures import create_data_loader
from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator
from utils.logger import GANLogger
from utils.visualizer import save_sample_grid, create_image_grid
from utils.metrics import calculate_fid, INCEPTION_AVAILABLE


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AblationConfig:
    """
    Configuration for a single ablation experiment.
    
    Attributes:
        name: Unique experiment name
        latent_dim: Dimension of the latent noise vector
        activation: Generator activation type ('relu' or 'leaky_relu')
        use_spectral_norm: Whether to use spectral normalization in discriminator
        image_size: Size of generated images
        image_channels: Number of image channels
        batch_size: Training batch size
        epochs: Number of training epochs
        g_lr: Generator learning rate
        d_lr: Discriminator learning rate
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        label_smoothing: Label smoothing factor for real samples
    """
    name: str
    latent_dim: int = 100
    activation: str = "relu"  # 'relu' or 'leaky_relu'
    use_spectral_norm: bool = False
    image_size: int = 64
    image_channels: int = 1
    batch_size: int = 64
    epochs: int = 50
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def get_short_name(self) -> str:
        """Generate a short descriptive name for the configuration."""
        spec_str = "SN" if self.use_spectral_norm else "noSN"
        act_str = "LReLU" if self.activation == "leaky_relu" else "ReLU"
        return f"z{self.latent_dim}_{act_str}_{spec_str}"


@dataclass
class AblationResult:
    """
    Results from a single ablation experiment.
    
    Attributes:
        config: The experiment configuration
        g_losses: List of generator losses per epoch
        d_losses: List of discriminator losses per epoch
        d_real_scores: List of discriminator scores on real images
        d_fake_scores: List of discriminator scores on fake images
        fid_score: Final FID score (if computed)
        loss_variance_g: Variance of generator loss (stability metric)
        loss_variance_d: Variance of discriminator loss
        final_g_loss: Final generator loss
        final_d_loss: Final discriminator loss
        training_time: Total training time in seconds
        sample_path: Path to saved sample images
    """
    config: AblationConfig
    g_losses: List[float] = field(default_factory=list)
    d_losses: List[float] = field(default_factory=list)
    d_real_scores: List[float] = field(default_factory=list)
    d_fake_scores: List[float] = field(default_factory=list)
    fid_score: Optional[float] = None
    loss_variance_g: float = 0.0
    loss_variance_d: float = 0.0
    final_g_loss: float = 0.0
    final_d_loss: float = 0.0
    training_time: float = 0.0
    sample_path: str = ""
    
    def compute_stability_metrics(self) -> None:
        """Compute training stability metrics from loss histories."""
        if self.g_losses:
            self.loss_variance_g = float(np.var(self.g_losses))
            self.final_g_loss = self.g_losses[-1]
        if self.d_losses:
            self.loss_variance_d = float(np.var(self.d_losses))
            self.final_d_loss = self.d_losses[-1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
            "d_real_scores": self.d_real_scores,
            "d_fake_scores": self.d_fake_scores,
            "fid_score": self.fid_score,
            "loss_variance_g": self.loss_variance_g,
            "loss_variance_d": self.loss_variance_d,
            "final_g_loss": self.final_g_loss,
            "final_d_loss": self.final_d_loss,
            "training_time": self.training_time,
            "sample_path": self.sample_path
        }


# =============================================================================
# Custom Generator with Configurable Activation
# =============================================================================

class UpsampleBlockConfigurable(nn.Module):
    """
    Upsampling block with configurable activation function.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to both sides
        output_padding: Additional size added to output
        use_batch_norm: Whether to apply batch normalization
        activation: Activation type ('relu' or 'leaky_relu')
        leaky_slope: Negative slope for LeakyReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        use_batch_norm: bool = True,
        activation: str = "relu",
        leaky_slope: float = 0.2
    ) -> None:
        super().__init__()
        
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=not use_batch_norm
            )
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling block."""
        return self.block(x)


class ConfigurableGenerator(nn.Module):
    """
    Generator network with configurable activation function.
    
    This generator allows switching between ReLU and LeakyReLU activations
    for ablation studies.
    
    Args:
        latent_dim: Dimension of the latent vector z
        output_size: Output image size (64 or 128)
        output_channels: Number of output channels
        base_features: Base number of features
        activation: Activation type ('relu' or 'leaky_relu')
        leaky_slope: Negative slope for LeakyReLU
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        output_size: int = 64,
        output_channels: int = 1,
        base_features: int = 256,
        activation: str = "relu",
        leaky_slope: float = 0.2
    ) -> None:
        super().__init__()
        
        if output_size not in [64, 128]:
            raise ValueError(f"output_size must be 64 or 128, got {output_size}")
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.output_channels = output_channels
        self.base_features = base_features
        self.activation = activation
        
        # Initial projection dimensions
        self.init_size = 4
        if output_size == 64:
            self.init_channels = base_features  # 256
        else:
            self.init_channels = base_features * 2  # 512
        
        # Dense layer for latent projection
        fc_layers: List[nn.Module] = [
            nn.Linear(latent_dim, self.init_channels * self.init_size * self.init_size),
            nn.BatchNorm1d(self.init_channels * self.init_size * self.init_size)
        ]
        
        if activation == "leaky_relu":
            fc_layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        else:
            fc_layers.append(nn.ReLU(inplace=True))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Build upsampling blocks based on output size
        if output_size == 64:
            self.upsample_blocks = nn.Sequential(
                UpsampleBlockConfigurable(256, 128, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(128, 64, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(64, 32, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(32, 32, activation=activation, leaky_slope=leaky_slope),
            )
            final_in_channels = 32
        else:
            self.upsample_blocks = nn.Sequential(
                UpsampleBlockConfigurable(512, 256, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(256, 128, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(128, 64, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(64, 32, activation=activation, leaky_slope=leaky_slope),
                UpsampleBlockConfigurable(32, 32, activation=activation, leaky_slope=leaky_slope),
            )
            final_in_channels = 32
        
        # Final convolution to output image
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using normal distribution with std=0.02."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: transform latent vector to image.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            Generated images of shape (batch_size, output_channels, output_size, output_size)
        """
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.upsample_blocks(x)
        x = self.final_conv(x)
        return x


# =============================================================================
# Ablation GAN Trainer
# =============================================================================

class AblationGANTrainer:
    """
    Trainer for running ablation experiments on Vanilla GAN.
    
    Args:
        config: Ablation configuration
        data_loader: PyTorch DataLoader for training data
        device: Device to train on
        output_dir: Directory to save results
    """
    
    def __init__(
        self,
        config: AblationConfig,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        output_dir: Path
    ) -> None:
        self.config = config
        self.data_loader = data_loader
        self.device = device
        self.output_dir = output_dir
        
        # Initialize generator with configurable activation
        self.generator = ConfigurableGenerator(
            latent_dim=config.latent_dim,
            output_size=config.image_size,
            output_channels=config.image_channels,
            activation=config.activation
        ).to(device)
        
        # Initialize discriminator
        self.discriminator = Discriminator(
            input_size=config.image_size,
            input_channels=config.image_channels,
            use_spectral_norm=config.use_spectral_norm
        ).to(device)
        
        # Loss and optimizers
        self.criterion = nn.BCELoss()
        
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(config.beta1, config.beta2)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(config.beta1, config.beta2)
        )
        
        # Fixed noise for visual tracking
        self.fixed_noise = torch.randn(64, config.latent_dim, device=device)
        
        # Training state
        self.g_losses: List[float] = []
        self.d_losses: List[float] = []
        self.d_real_scores: List[float] = []
        self.d_fake_scores: List[float] = []
    
    def train_epoch(self) -> Tuple[float, float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (avg_g_loss, avg_d_loss, avg_d_real, avg_d_fake)
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        num_batches = 0
        
        for real_images in self.data_loader:
            if isinstance(real_images, (list, tuple)):
                real_images = real_images[0]
            
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Labels with smoothing
            real_labels = torch.full((batch_size, 1), self.config.label_smoothing, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # ----- Train Discriminator -----
            self.d_optimizer.zero_grad()
            
            # Real images
            d_real_output = self.discriminator(real_images)
            d_real_loss = self.criterion(d_real_output, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_images = self.generator(z)
            d_fake_output = self.discriminator(fake_images.detach())
            d_fake_loss = self.criterion(d_fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # ----- Train Generator -----
            self.g_optimizer.zero_grad()
            
            d_output_for_g = self.discriminator(fake_images)
            g_loss = self.criterion(d_output_for_g, real_labels)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Track metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real += d_real_output.mean().item()
            epoch_d_fake += d_fake_output.mean().item()
            num_batches += 1
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        self.d_real_scores.append(avg_d_real)
        self.d_fake_scores.append(avg_d_fake)
        
        return avg_g_loss, avg_d_loss, avg_d_real, avg_d_fake
    
    def train(self, progress_bar: bool = True) -> AblationResult:
        """
        Run full training and return results.
        
        Args:
            progress_bar: Whether to show progress bar
        
        Returns:
            AblationResult containing training metrics
        """
        import time
        start_time = time.time()
        
        epochs_iter = range(1, self.config.epochs + 1)
        if progress_bar:
            epochs_iter = tqdm(epochs_iter, desc=f"Training {self.config.get_short_name()}", leave=False)
        
        for epoch in epochs_iter:
            g_loss, d_loss, d_real, d_fake = self.train_epoch()
            
            if progress_bar:
                epochs_iter.set_postfix({  # type: ignore
                    "G": f"{g_loss:.4f}",
                    "D": f"{d_loss:.4f}"
                })
        
        training_time = time.time() - start_time
        
        # Save final samples
        sample_path = self._save_samples()
        
        # Create result
        result = AblationResult(
            config=self.config,
            g_losses=self.g_losses,
            d_losses=self.d_losses,
            d_real_scores=self.d_real_scores,
            d_fake_scores=self.d_fake_scores,
            training_time=training_time,
            sample_path=str(sample_path)
        )
        result.compute_stability_metrics()
        
        return result
    
    def _save_samples(self) -> Path:
        """Generate and save sample images."""
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(self.fixed_noise)
        
        sample_path = self.output_dir / "samples" / f"{self.config.get_short_name()}_samples.png"
        save_sample_grid(samples, sample_path, nrow=8)
        
        return sample_path
    
    def generate_samples(self, n_samples: int = 64) -> torch.Tensor:
        """Generate samples for evaluation."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim, device=self.device)
            samples = self.generator(z)
        return samples


# =============================================================================
# Ablation Study Manager
# =============================================================================

class AblationStudyManager:
    """
    Manager for running complete ablation studies.
    
    Orchestrates multiple ablation experiments with different configurations
    and aggregates results.
    
    Args:
        data_dir: Path to training data directory
        output_dir: Path to save all results
        epochs: Number of training epochs per experiment
        batch_size: Training batch size
        image_size: Image size for training
        device: Device to train on
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        epochs: int = 50,
        batch_size: int = 64,
        image_size: int = 64,
        device: Optional[str] = None
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[AblationStudy] Using device: {self.device}")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Results storage
        self.results: List[AblationResult] = []
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_configurations(self) -> List[AblationConfig]:
        """
        Generate all ablation configurations.
        
        Returns:
            List of AblationConfig for each experiment
        """
        # Define ablation parameters
        latent_dims = [50, 100, 200]
        activations = ["relu", "leaky_relu"]
        spectral_norms = [False, True]
        
        configs = []
        
        for latent_dim, activation, use_spectral_norm in product(latent_dims, activations, spectral_norms):
            config = AblationConfig(
                name=f"ablation_{latent_dim}_{activation}_{use_spectral_norm}",
                latent_dim=latent_dim,
                activation=activation,
                use_spectral_norm=use_spectral_norm,
                image_size=self.image_size,
                batch_size=self.batch_size,
                epochs=self.epochs
            )
            configs.append(config)
        
        return configs
    
    def run_experiment(
        self,
        config: AblationConfig,
        data_loader: torch.utils.data.DataLoader,
        real_images_for_fid: Optional[torch.Tensor] = None
    ) -> AblationResult:
        """
        Run a single ablation experiment.
        
        Args:
            config: Experiment configuration
            data_loader: Training data loader
            real_images_for_fid: Real images for FID calculation
        
        Returns:
            AblationResult from the experiment
        """
        print(f"\n[AblationStudy] Running: {config.get_short_name()}")
        print(f"  Latent dim: {config.latent_dim}, Activation: {config.activation}, "
              f"Spectral Norm: {config.use_spectral_norm}")
        
        # Create trainer
        trainer = AblationGANTrainer(
            config=config,
            data_loader=data_loader,
            device=self.device,
            output_dir=self.output_dir
        )
        
        # Train
        result = trainer.train(progress_bar=True)
        
        # Compute FID if real images provided and inception available
        if INCEPTION_AVAILABLE and real_images_for_fid is not None:
            try:
                print("  Computing FID score...")
                fake_images = trainer.generate_samples(n_samples=min(len(real_images_for_fid), 256))
                fid = calculate_fid(
                    real_images_for_fid[:256],
                    fake_images.cpu(),
                    device=self.device
                )
                result.fid_score = fid
                print(f"  FID: {fid:.2f}")
            except Exception as e:
                print(f"  FID calculation failed: {e}")
                result.fid_score = None
        
        print(f"  Final G Loss: {result.final_g_loss:.4f}, D Loss: {result.final_d_loss:.4f}")
        print(f"  G Loss Variance: {result.loss_variance_g:.6f}, D Loss Variance: {result.loss_variance_d:.6f}")
        print(f"  Training Time: {result.training_time:.1f}s")
        
        return result
    
    def run_all(self) -> List[AblationResult]:
        """
        Run all ablation experiments.
        
        Returns:
            List of all AblationResults
        """
        # Create data loader
        print(f"[AblationStudy] Loading data from: {self.data_dir}")
        data_loader = create_data_loader(
            data_dir=str(self.data_dir),
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_workers=4
        )
        
        print(f"[AblationStudy] Dataset size: {len(data_loader.dataset)} images")
        
        # Collect real images for FID
        real_images_list = []
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            real_images_list.append(batch)
            if len(real_images_list) * self.batch_size >= 512:
                break
        real_images_for_fid = torch.cat(real_images_list, dim=0)[:512]
        
        # Generate configurations
        configs = self.generate_configurations()
        print(f"[AblationStudy] Running {len(configs)} experiments")
        
        # Run all experiments
        for i, config in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"Experiment {i}/{len(configs)}")
            print(f"{'='*60}")
            
            result = self.run_experiment(config, data_loader, real_images_for_fid)
            self.results.append(result)
        
        return self.results
    
    def save_results_table(self) -> Path:
        """
        Save results as CSV and markdown tables.
        
        Returns:
            Path to the saved CSV file
        """
        # Prepare data for DataFrame
        data = []
        for result in self.results:
            row = {
                "Configuration": result.config.get_short_name(),
                "Latent Dim": result.config.latent_dim,
                "Activation": result.config.activation,
                "Spectral Norm": result.config.use_spectral_norm,
                "Final G Loss": f"{result.final_g_loss:.4f}",
                "Final D Loss": f"{result.final_d_loss:.4f}",
                "G Loss Variance": f"{result.loss_variance_g:.6f}",
                "D Loss Variance": f"{result.loss_variance_d:.6f}",
                "FID Score": f"{result.fid_score:.2f}" if result.fid_score else "N/A",
                "Training Time (s)": f"{result.training_time:.1f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save CSV
        csv_path = self.output_dir / f"ablation_results_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[AblationStudy] Results saved to: {csv_path}")
        
        # Save markdown table
        md_path = self.output_dir / f"ablation_results_{self.timestamp}.md"
        with open(md_path, 'w') as f:
            f.write("# Ablation Study Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Epochs**: {self.epochs}\n\n")
            f.write(f"**Batch Size**: {self.batch_size}\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Legend\n\n")
            f.write("- **G Loss Variance**: Lower is more stable training\n")
            f.write("- **FID Score**: Lower is better visual quality\n")
            f.write("- **SN**: Spectral Normalization\n")
            f.write("- **LReLU**: LeakyReLU activation\n")
        
        print(f"[AblationStudy] Markdown table saved to: {md_path}")
        
        # Save JSON with full data
        json_path = self.output_dir / f"ablation_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        print(f"[AblationStudy] Full results saved to: {json_path}")
        
        return csv_path
    
    def generate_comparison_plots(self) -> List[Path]:
        """
        Generate comparison plots for all experiments.
        
        Returns:
            List of paths to generated plot files
        """
        plot_paths = []
        plots_dir = self.output_dir / "plots"
        
        # 1. Loss curves comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for result in self.results:
            label = result.config.get_short_name()
            epochs = range(1, len(result.g_losses) + 1)
            
            axes[0, 0].plot(epochs, result.g_losses, label=label, alpha=0.7)
            axes[0, 1].plot(epochs, result.d_losses, label=label, alpha=0.7)
            axes[1, 0].plot(epochs, result.d_real_scores, label=label, alpha=0.7)
            axes[1, 1].plot(epochs, result.d_fake_scores, label=label, alpha=0.7)
        
        axes[0, 0].set_title('Generator Loss', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(fontsize=8, loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Discriminator Loss', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(fontsize=8, loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('D(real) - Discriminator Score on Real', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(fontsize=8, loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('D(fake) - Discriminator Score on Fake', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend(fontsize=8, loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_curves_path = plots_dir / f"loss_curves_comparison_{self.timestamp}.png"
        plt.savefig(loss_curves_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(loss_curves_path)
        
        # 2. Bar chart: Final losses by configuration
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        configs = [r.config.get_short_name() for r in self.results]
        g_losses = [r.final_g_loss for r in self.results]
        d_losses = [r.final_d_loss for r in self.results]
        
        x = np.arange(len(configs))
        width = 0.35
        
        axes[0].bar(x, g_losses, width, label='G Loss', color='blue', alpha=0.7)
        axes[0].bar(x + width, d_losses, width, label='D Loss', color='orange', alpha=0.7)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Final Loss')
        axes[0].set_title('Final Losses by Configuration')
        axes[0].set_xticks(x + width / 2)
        axes[0].set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 3. Loss variance (stability) comparison
        g_var = [r.loss_variance_g for r in self.results]
        d_var = [r.loss_variance_d for r in self.results]
        
        axes[1].bar(x, g_var, width, label='G Loss Variance', color='blue', alpha=0.7)
        axes[1].bar(x + width, d_var, width, label='D Loss Variance', color='orange', alpha=0.7)
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('Loss Variance')
        axes[1].set_title('Training Stability (Lower = More Stable)')
        axes[1].set_xticks(x + width / 2)
        axes[1].set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        bar_chart_path = plots_dir / f"final_metrics_comparison_{self.timestamp}.png"
        plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(bar_chart_path)
        
        # 4. FID scores comparison (if available)
        fid_scores = [(r.config.get_short_name(), r.fid_score) for r in self.results if r.fid_score is not None]
        if fid_scores:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            names, scores = zip(*fid_scores)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(scores)))
            
            bars = ax.bar(names, scores, color=colors)
            ax.set_xlabel('Configuration')
            ax.set_ylabel('FID Score (Lower is Better)')
            ax.set_title('FID Score Comparison')
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{score:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            fid_path = plots_dir / f"fid_comparison_{self.timestamp}.png"
            plt.savefig(fid_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plot_paths.append(fid_path)
        
        # 5. Grouped comparison by factor
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Group by latent dimension
        latent_groups: Dict[int, List[float]] = {50: [], 100: [], 200: []}
        for r in self.results:
            if r.fid_score is not None:
                latent_groups[r.config.latent_dim].append(r.fid_score)
        
        latent_dims = list(latent_groups.keys())
        latent_means = [np.mean(v) if v else 0 for v in latent_groups.values()]
        latent_stds = [np.std(v) if v else 0 for v in latent_groups.values()]
        
        axes[0].bar(range(len(latent_dims)), latent_means, yerr=latent_stds, capsize=5, color='steelblue', alpha=0.7)
        axes[0].set_xticks(range(len(latent_dims)))
        axes[0].set_xticklabels([str(d) for d in latent_dims])
        axes[0].set_xlabel('Latent Dimension')
        axes[0].set_ylabel('Average FID Score')
        axes[0].set_title('FID by Latent Dimension')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Group by activation
        act_groups: Dict[str, List[float]] = {"relu": [], "leaky_relu": []}
        for r in self.results:
            if r.fid_score is not None:
                act_groups[r.config.activation].append(r.fid_score)
        
        act_names = ["ReLU", "LeakyReLU"]
        act_means = [np.mean(act_groups["relu"]) if act_groups["relu"] else 0,
                     np.mean(act_groups["leaky_relu"]) if act_groups["leaky_relu"] else 0]
        act_stds = [np.std(act_groups["relu"]) if act_groups["relu"] else 0,
                    np.std(act_groups["leaky_relu"]) if act_groups["leaky_relu"] else 0]
        
        axes[1].bar(range(len(act_names)), act_means, yerr=act_stds, capsize=5, color='coral', alpha=0.7)
        axes[1].set_xticks(range(len(act_names)))
        axes[1].set_xticklabels(act_names)
        axes[1].set_xlabel('Activation Function')
        axes[1].set_ylabel('Average FID Score')
        axes[1].set_title('FID by Activation')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Group by spectral norm
        sn_groups: Dict[bool, List[float]] = {False: [], True: []}
        for r in self.results:
            if r.fid_score is not None:
                sn_groups[r.config.use_spectral_norm].append(r.fid_score)
        
        sn_names = ["Without SN", "With SN"]
        sn_means = [np.mean(sn_groups[False]) if sn_groups[False] else 0,
                    np.mean(sn_groups[True]) if sn_groups[True] else 0]
        sn_stds = [np.std(sn_groups[False]) if sn_groups[False] else 0,
                   np.std(sn_groups[True]) if sn_groups[True] else 0]
        
        axes[2].bar(range(len(sn_names)), sn_means, yerr=sn_stds, capsize=5, color='seagreen', alpha=0.7)
        axes[2].set_xticks(range(len(sn_names)))
        axes[2].set_xticklabels(sn_names)
        axes[2].set_xlabel('Spectral Normalization')
        axes[2].set_ylabel('Average FID Score')
        axes[2].set_title('FID by Spectral Norm')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        grouped_path = plots_dir / f"grouped_fid_comparison_{self.timestamp}.png"
        plt.savefig(grouped_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(grouped_path)
        
        print(f"\n[AblationStudy] Generated {len(plot_paths)} comparison plots")
        for path in plot_paths:
            print(f"  - {path}")
        
        return plot_paths
    
    def create_sample_grid_comparison(self) -> Path:
        """
        Create a grid showing samples from all configurations.
        
        Returns:
            Path to the saved comparison grid
        """
        n_configs = len(self.results)
        if n_configs == 0:
            raise ValueError("No results to compare")
        
        # Calculate grid layout
        ncols = min(4, n_configs)
        nrows = (n_configs + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if n_configs == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(self.results):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            
            # Load sample image
            sample_path = Path(result.sample_path)
            if sample_path.exists():
                img = plt.imread(sample_path)
                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            else:
                ax.text(0.5, 0.5, 'No Sample', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f"{result.config.get_short_name()}\nFID: {result.fid_score:.1f}" if result.fid_score else result.config.get_short_name(), fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_configs, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        grid_path = self.output_dir / "plots" / f"sample_grid_comparison_{self.timestamp}.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[AblationStudy] Sample grid comparison saved to: {grid_path}")
        return grid_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ablation Study for Vanilla GAN Signature Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing training images"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per experiment"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Directory to save ablation study results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        choices=[64, 128],
        help="Image size for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, cuda:0, etc.)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for ablation study."""
    args = parse_args()
    
    print("=" * 60)
    print("Vanilla GAN Ablation Study")
    print("=" * 60)
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.image_size}")
    print("=" * 60)
    
    # Verify data directory exists
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Create and run ablation study
    manager = AblationStudyManager(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device
    )
    
    # Run all experiments
    results = manager.run_all()
    
    # Save results
    manager.save_results_table()
    
    # Generate comparison plots
    manager.generate_comparison_plots()
    
    # Create sample grid comparison
    manager.create_sample_grid_comparison()
    
    print("\n" + "=" * 60)
    print("Ablation Study Complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print(f"Total experiments: {len(results)}")
    
    # Print summary
    print("\n--- Summary ---")
    best_fid = min((r for r in results if r.fid_score is not None), key=lambda r: r.fid_score, default=None)
    if best_fid:
        print(f"Best FID: {best_fid.fid_score:.2f} ({best_fid.config.get_short_name()})")
    
    most_stable = min(results, key=lambda r: r.loss_variance_g)
    print(f"Most Stable (lowest G variance): {most_stable.loss_variance_g:.6f} ({most_stable.config.get_short_name()})")


if __name__ == "__main__":
    main()
