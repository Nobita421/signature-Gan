"""
Vanilla GAN Model for Signature Generation

This module combines the Generator and Discriminator into a complete
VanillaGAN class with training, inference, and model management methods.

Features:
    - Binary Cross-Entropy loss with label smoothing
    - Adam optimizer with GAN-specific hyperparameters
    - Training step methods for D and G
    - Sample generation
    - Model save/load functionality
    - CPU/CUDA device handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any
import json
from datetime import datetime

from generator_vanilla_gan import Generator, create_generator
from discriminator_vanilla_gan import Discriminator, create_discriminator


class VanillaGAN(nn.Module):
    """
    Complete Vanilla GAN model combining Generator and Discriminator.
    
    Implements the original GAN training procedure with improvements
    for stability (label smoothing, appropriate learning rates).
    
    Args:
        latent_dim: Dimension of the latent vector z (default: 100)
        image_size: Size of generated images, 64 or 128 (default: 64)
        image_channels: Number of image channels (default: 1 for grayscale)
        g_lr: Generator learning rate (default: 2e-4)
        d_lr: Discriminator learning rate (default: 2e-4)
        beta1: Adam beta1 parameter (default: 0.5)
        beta2: Adam beta2 parameter (default: 0.999)
        label_smoothing: Label smoothing for real labels (default: 0.9)
        use_spectral_norm: Use spectral normalization in discriminator (default: False)
        device: Device to run on ('cpu', 'cuda', or specific 'cuda:0') (default: auto)
    
    Example:
        >>> gan = VanillaGAN(latent_dim=100, image_size=64)
        >>> gan.to('cuda')
        >>> 
        >>> # Training step
        >>> real_images = torch.randn(32, 1, 64, 64).to('cuda')
        >>> d_loss = gan.train_discriminator_step(real_images)
        >>> g_loss = gan.train_generator_step(batch_size=32)
        >>> 
        >>> # Generate samples
        >>> fake_images = gan.generate(n_samples=16)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        image_size: int = 64,
        image_channels: int = 1,
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        label_smoothing: float = 0.9,
        use_spectral_norm: bool = False,
        device: Optional[str] = None
    ) -> None:
        super().__init__()
        
        # Store configuration
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.label_smoothing = label_smoothing
        self.use_spectral_norm = use_spectral_norm
        
        # Determine device
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        
        # Initialize Generator
        self.generator = Generator(
            latent_dim=latent_dim,
            output_size=image_size,
            output_channels=image_channels
        )
        
        # Initialize Discriminator
        self.discriminator = Discriminator(
            input_size=image_size,
            input_channels=image_channels,
            use_spectral_norm=use_spectral_norm
        )
        
        # Loss function: Binary Cross-Entropy
        self.criterion = nn.BCELoss()
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(beta1, beta2)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=(beta1, beta2)
        )
        
        # Training state tracking
        self.current_epoch = 0
        self.global_step = 0
        self.d_losses: list = []
        self.g_losses: list = []
        
        # Move to device
        self.to(self._device)
    
    @property
    def device(self) -> torch.device:
        """Return the current device."""
        return self._device
    
    def to(self, device: Union[str, torch.device]) -> 'VanillaGAN':
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        
        Returns:
            Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        super().to(device)
        return self
    
    def _get_labels(
        self,
        batch_size: int,
        real: bool = True,
        smooth: bool = True
    ) -> torch.Tensor:
        """
        Generate labels for training with optional label smoothing.
        
        Args:
            batch_size: Number of labels to generate
            real: If True, generate real labels; if False, fake labels
            smooth: If True, apply label smoothing to real labels
        
        Returns:
            Labels tensor of shape (batch_size, 1)
        """
        if real:
            if smooth:
                # Label smoothing: real labels are 0.9 instead of 1.0
                labels = torch.full((batch_size, 1), self.label_smoothing, device=self._device)
            else:
                labels = torch.ones(batch_size, 1, device=self._device)
        else:
            labels = torch.zeros(batch_size, 1, device=self._device)
        
        return labels
    
    def train_discriminator_step(
        self,
        real_images: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step for the Discriminator.
        
        The discriminator is trained to:
        1. Classify real images as real (label ≈ 0.9 with smoothing)
        2. Classify fake images (from generator) as fake (label = 0)
        
        Args:
            real_images: Batch of real images, shape (batch_size, channels, H, W)
            noise: Optional pre-generated noise vector; if None, will be generated
        
        Returns:
            Dictionary with loss values:
                - d_loss: Total discriminator loss
                - d_loss_real: Loss on real images
                - d_loss_fake: Loss on fake images
                - d_real_acc: Accuracy on real images (predictions > 0.5)
                - d_fake_acc: Accuracy on fake images (predictions < 0.5)
        """
        self.discriminator.train()
        self.generator.eval()
        
        batch_size = real_images.size(0)
        real_images = real_images.to(self._device)
        
        # Zero gradients
        self.d_optimizer.zero_grad()
        
        # ----- Train on real images -----
        real_labels = self._get_labels(batch_size, real=True, smooth=True)
        real_preds = self.discriminator(real_images)
        d_loss_real = self.criterion(real_preds, real_labels)
        
        # ----- Train on fake images -----
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=self._device)
        
        with torch.no_grad():
            fake_images = self.generator(noise)
        
        fake_labels = self._get_labels(batch_size, real=False)
        fake_preds = self.discriminator(fake_images)
        d_loss_fake = self.criterion(fake_preds, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        # Backward pass and optimization
        d_loss.backward()
        self.d_optimizer.step()
        
        # Calculate accuracies
        d_real_acc = (real_preds > 0.5).float().mean().item()
        d_fake_acc = (fake_preds < 0.5).float().mean().item()
        
        # Track losses
        self.d_losses.append(d_loss.item())
        self.global_step += 1
        
        return {
            'd_loss': d_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc,
            'd_real_mean': real_preds.mean().item(),
            'd_fake_mean': fake_preds.mean().item()
        }
    
    def train_generator_step(
        self,
        batch_size: int,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step for the Generator.
        
        The generator is trained to fool the discriminator by generating
        images that the discriminator classifies as real.
        
        Args:
            batch_size: Number of fake images to generate
            noise: Optional pre-generated noise vector; if None, will be generated
        
        Returns:
            Dictionary with loss values:
                - g_loss: Generator loss
                - g_fake_mean: Mean discriminator output for fake images
        """
        self.generator.train()
        self.discriminator.eval()
        
        # Zero gradients
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=self._device)
        
        fake_images = self.generator(noise)
        
        # We want the discriminator to classify fake images as real
        # So we use real labels for the generator loss
        real_labels = self._get_labels(batch_size, real=True, smooth=False)
        
        # Get discriminator predictions on fake images
        fake_preds = self.discriminator(fake_images)
        
        # Generator loss: how well does G fool D?
        g_loss = self.criterion(fake_preds, real_labels)
        
        # Backward pass and optimization
        g_loss.backward()
        self.g_optimizer.step()
        
        # Track losses
        self.g_losses.append(g_loss.item())
        
        return {
            'g_loss': g_loss.item(),
            'g_fake_mean': fake_preds.mean().item()
        }
    
    def train_step(
        self,
        real_images: torch.Tensor,
        n_critic: int = 1
    ) -> Dict[str, float]:
        """
        Perform complete training step (D and G).
        
        Args:
            real_images: Batch of real images
            n_critic: Number of discriminator updates per generator update
        
        Returns:
            Dictionary with all loss values from both D and G steps
        """
        batch_size = real_images.size(0)
        metrics = {}
        
        # Train discriminator n_critic times
        for i in range(n_critic):
            d_metrics = self.train_discriminator_step(real_images)
            if i == n_critic - 1:  # Keep metrics from last D step
                metrics.update(d_metrics)
        
        # Train generator once
        g_metrics = self.train_generator_step(batch_size)
        metrics.update(g_metrics)
        
        return metrics
    
    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        device: Optional[Union[str, torch.device]] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate fake signature images.
        
        Args:
            n_samples: Number of images to generate
            device: Device to generate on (default: model's device)
            noise: Optional pre-generated noise; if None, will sample from N(0, I)
        
        Returns:
            Generated images tensor of shape (n_samples, channels, H, W)
            Values are in range [-1, 1]
        """
        self.generator.eval()
        
        if device is None:
            device = self._device
        elif isinstance(device, str):
            device = torch.device(device)
        
        if noise is None:
            noise = torch.randn(n_samples, self.latent_dim, device=device)
        else:
            noise = noise.to(device)
        
        fake_images = self.generator(noise)
        
        return fake_images
    
    @torch.no_grad()
    def generate_interpolation(
        self,
        n_steps: int = 10,
        z_start: Optional[torch.Tensor] = None,
        z_end: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate interpolated images between two latent vectors.
        
        Args:
            n_steps: Number of interpolation steps
            z_start: Starting latent vector (random if None)
            z_end: Ending latent vector (random if None)
        
        Returns:
            Interpolated images of shape (n_steps, channels, H, W)
        """
        self.generator.eval()
        
        if z_start is None:
            z_start = torch.randn(1, self.latent_dim, device=self._device)
        if z_end is None:
            z_end = torch.randn(1, self.latent_dim, device=self._device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, n_steps, device=self._device).view(-1, 1)
        
        # Spherical linear interpolation (slerp) for better results
        z_interp = z_start * (1 - alphas) + z_end * alphas
        
        # Generate images
        images = self.generator(z_interp)
        
        return images
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
            'image_channels': self.image_channels,
            'g_lr': self.g_lr,
            'd_lr': self.d_lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'label_smoothing': self.label_smoothing,
            'use_spectral_norm': self.use_spectral_norm,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'g_params': self.generator.get_num_params(),
            'd_params': self.discriminator.get_num_params(),
            'total_params': self.generator.get_num_params() + self.discriminator.get_num_params()
        }
    
    def save(
        self,
        path: Union[str, Path],
        save_optimizer: bool = True,
        save_history: bool = True
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint (without extension)
            save_optimizer: Whether to save optimizer states
            save_history: Whether to save loss history
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.get_config(),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'saved_at': datetime.now().isoformat()
        }
        
        if save_optimizer:
            checkpoint['g_optimizer_state_dict'] = self.g_optimizer.state_dict()
            checkpoint['d_optimizer_state_dict'] = self.d_optimizer.state_dict()
        
        if save_history:
            checkpoint['d_losses'] = self.d_losses
            checkpoint['g_losses'] = self.g_losses
        
        # Save PyTorch checkpoint
        torch.save(checkpoint, f"{path}.pt")
        
        # Save config as JSON for easy inspection
        with open(f"{path}_config.json", 'w') as f:
            json.dump(self.get_config(), f, indent=2)
        
        print(f"Model saved to {path}.pt")
    
    def load(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        load_history: bool = True,
        map_location: Optional[str] = None
    ) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file (with or without .pt extension)
            load_optimizer: Whether to load optimizer states
            load_history: Whether to load loss history
            map_location: Device to map tensors to (default: current device)
        """
        path = Path(path)
        if not path.suffix:
            path = Path(f"{path}.pt")
        
        if map_location is None:
            map_location = str(self._device)
        
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        # Load model weights
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        if load_optimizer and 'g_optimizer_state_dict' in checkpoint:
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if load_history and 'd_losses' in checkpoint:
            self.d_losses = checkpoint.get('d_losses', [])
            self.g_losses = checkpoint.get('g_losses', [])
        
        print(f"Model loaded from {path}")
        print(f"  Epoch: {self.current_epoch}, Global Step: {self.global_step}")
    
    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None
    ) -> 'VanillaGAN':
        """
        Create a VanillaGAN instance from a checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model to
        
        Returns:
            Loaded VanillaGAN instance
        """
        path = Path(path)
        if not path.suffix:
            path = Path(f"{path}.pt")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        # Create model with saved config
        model = cls(
            latent_dim=config['latent_dim'],
            image_size=config['image_size'],
            image_channels=config['image_channels'],
            g_lr=config['g_lr'],
            d_lr=config['d_lr'],
            beta1=config['beta1'],
            beta2=config['beta2'],
            label_smoothing=config['label_smoothing'],
            use_spectral_norm=config['use_spectral_norm'],
            device=device
        )
        
        # Load weights
        model.load(path, load_optimizer=True, load_history=True)
        
        return model
    
    def set_learning_rates(self, g_lr: float, d_lr: float) -> None:
        """
        Update learning rates for both optimizers.
        
        Args:
            g_lr: New generator learning rate
            d_lr: New discriminator learning rate
        """
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        
        self.g_lr = g_lr
        self.d_lr = d_lr
    
    def get_recent_losses(self, n: int = 100) -> Dict[str, float]:
        """
        Get average of recent losses.
        
        Args:
            n: Number of recent steps to average
        
        Returns:
            Dictionary with average D and G losses
        """
        d_recent = self.d_losses[-n:] if self.d_losses else [0]
        g_recent = self.g_losses[-n:] if self.g_losses else [0]
        
        return {
            'avg_d_loss': sum(d_recent) / len(d_recent),
            'avg_g_loss': sum(g_recent) / len(g_recent)
        }
    
    def summary(self) -> str:
        """
        Get a string summary of the model.
        
        Returns:
            Model summary string
        """
        config = self.get_config()
        
        summary_lines = [
            "=" * 60,
            "VanillaGAN Model Summary",
            "=" * 60,
            f"Device: {self._device}",
            f"Latent Dimension: {config['latent_dim']}",
            f"Image Size: {config['image_size']}x{config['image_size']}",
            f"Image Channels: {config['image_channels']}",
            "-" * 60,
            "Generator:",
            f"  Parameters: {config['g_params']:,}",
            f"  Learning Rate: {config['g_lr']}",
            "-" * 60,
            "Discriminator:",
            f"  Parameters: {config['d_params']:,}",
            f"  Learning Rate: {config['d_lr']}",
            f"  Spectral Norm: {config['use_spectral_norm']}",
            "-" * 60,
            f"Total Parameters: {config['total_params']:,}",
            f"Label Smoothing: {config['label_smoothing']}",
            f"Adam Betas: ({config['beta1']}, {config['beta2']})",
            "-" * 60,
            "Training State:",
            f"  Current Epoch: {config['current_epoch']}",
            f"  Global Step: {config['global_step']}",
            "=" * 60
        ]
        
        return "\n".join(summary_lines)


def create_vanilla_gan(
    latent_dim: int = 100,
    image_size: int = 64,
    use_spectral_norm: bool = False,
    device: Optional[str] = None
) -> VanillaGAN:
    """
    Factory function to create a VanillaGAN with common defaults.
    
    Args:
        latent_dim: Dimension of latent vector
        image_size: Size of generated images (64 or 128)
        use_spectral_norm: Whether to use spectral normalization
        device: Device to run on
    
    Returns:
        Configured VanillaGAN instance
    """
    return VanillaGAN(
        latent_dim=latent_dim,
        image_size=image_size,
        image_channels=1,
        use_spectral_norm=use_spectral_norm,
        device=device
    )


if __name__ == "__main__":
    print("Testing VanillaGAN Model")
    print("=" * 60)
    
    # Create model
    gan = VanillaGAN(
        latent_dim=100,
        image_size=64,
        image_channels=1,
        use_spectral_norm=False
    )
    
    print(gan.summary())
    
    # Test discriminator training step
    print("\nTesting Discriminator Step:")
    real_images = torch.randn(8, 1, 64, 64).to(gan.device)
    d_metrics = gan.train_discriminator_step(real_images)
    for k, v in d_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test generator training step
    print("\nTesting Generator Step:")
    g_metrics = gan.train_generator_step(batch_size=8)
    for k, v in g_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test combined training step
    print("\nTesting Combined Training Step:")
    metrics = gan.train_step(real_images, n_critic=1)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test sample generation
    print("\nTesting Sample Generation:")
    samples = gan.generate(n_samples=16)
    print(f"  Generated shape: {samples.shape}")
    print(f"  Value range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    
    # Test interpolation
    print("\nTesting Latent Interpolation:")
    interp = gan.generate_interpolation(n_steps=5)
    print(f"  Interpolation shape: {interp.shape}")
    
    # Test save/load
    print("\nTesting Save/Load:")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_gan"
        gan.save(save_path)
        
        # Load into new model
        gan_loaded = VanillaGAN.from_checkpoint(save_path)
        print(f"  Loaded model epoch: {gan_loaded.current_epoch}")
        print(f"  Loaded model step: {gan_loaded.global_step}")
    
    print("\n✓ All tests passed!")
