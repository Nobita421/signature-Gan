"""
Generator Network for Vanilla GAN - Signature Generation

This module implements the Generator network that transforms random latent vectors
into realistic signature images using transposed convolutions.

Architecture:
    Input: latent vector z ∈ R^100 (configurable) sampled from N(0, I)
    Output: 64×64×1 or 128×128×1 grayscale signature image
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class UpsampleBlock(nn.Module):
    """
    Upsampling block consisting of ConvTranspose2D + BatchNorm + ReLU.
    
    This block doubles the spatial dimensions while reducing channel depth.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        output_padding: Additional size added to one side of the output shape
        use_batch_norm: Whether to apply batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        use_batch_norm: bool = True
    ) -> None:
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=not use_batch_norm  # No bias if using batch norm
            )
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling block."""
        return self.block(x)


class Generator(nn.Module):
    """
    Generator network for Vanilla GAN signature generation.
    
    Transforms random latent vectors into signature images using a series
    of upsampling blocks (transposed convolutions).
    
    Architecture for 64×64 output:
        Dense → reshape to 4×4×256
        4×4×256 → 8×8×128 → 16×16×64 → 32×32×32 → 64×64×1
    
    Architecture for 128×128 output:
        Dense → reshape to 4×4×512
        4×4×512 → 8×8×256 → 16×16×128 → 32×32×64 → 64×64×32 → 128×128×1
    
    Args:
        latent_dim: Dimension of the latent vector z (default: 100)
        output_size: Output image size, either 64 or 128 (default: 64)
        output_channels: Number of output channels (default: 1 for grayscale)
        base_features: Base number of features to scale architecture (default: 256)
    
    Example:
        >>> generator = Generator(latent_dim=100, output_size=64)
        >>> z = torch.randn(16, 100)  # Batch of 16 latent vectors
        >>> fake_images = generator(z)
        >>> print(fake_images.shape)  # torch.Size([16, 1, 64, 64])
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        output_size: int = 64,
        output_channels: int = 1,
        base_features: int = 256
    ) -> None:
        super().__init__()
        
        if output_size not in [64, 128]:
            raise ValueError(f"output_size must be 64 or 128, got {output_size}")
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.output_channels = output_channels
        self.base_features = base_features
        
        # Initial projection from latent space to spatial representation
        # For 64x64: project to 4x4x256
        # For 128x128: project to 4x4x512
        self.init_size = 4
        if output_size == 64:
            self.init_channels = base_features  # 256
        else:  # 128x128
            self.init_channels = base_features * 2  # 512
        
        # Dense layer to project latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.init_channels * self.init_size * self.init_size),
            nn.BatchNorm1d(self.init_channels * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        # Build upsampling blocks based on output size
        if output_size == 64:
            # 4x4x256 → 8x8x128 → 16x16x64 → 32x32x32 → 64x64x32
            self.upsample_blocks = nn.Sequential(
                UpsampleBlock(256, 128),   # 4x4 → 8x8
                UpsampleBlock(128, 64),    # 8x8 → 16x16
                UpsampleBlock(64, 32),     # 16x16 → 32x32
                UpsampleBlock(32, 32),     # 32x32 → 64x64
            )
            final_in_channels = 32
        else:  # 128x128
            # 4x4x512 → 8x8x256 → 16x16x128 → 32x32x64 → 64x64x32 → 128x128x32
            self.upsample_blocks = nn.Sequential(
                UpsampleBlock(512, 256),   # 4x4 → 8x8
                UpsampleBlock(256, 128),   # 8x8 → 16x16
                UpsampleBlock(128, 64),    # 16x16 → 32x32
                UpsampleBlock(64, 32),     # 32x32 → 64x64
                UpsampleBlock(32, 32),     # 64x64 → 128x128
            )
            final_in_channels = 32
        
        # Final convolution to produce output image
        # Using Conv2D with kernel_size=3 to refine features
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                final_in_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using normal distribution with std=0.02.
        
        Following DCGAN paper recommendations for weight initialization.
        
        Args:
            module: PyTorch module to initialize
        """
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
        # Project and reshape: (batch, latent_dim) → (batch, channels, 4, 4)
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Upsample through transposed convolutions
        x = self.upsample_blocks(x)
        
        # Final convolution to output channels
        x = self.final_conv(x)
        
        return x
    
    def generate_latent(
        self,
        n_samples: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate random latent vectors sampled from N(0, I).
        
        Args:
            n_samples: Number of latent vectors to generate
            device: Device to place the tensors on (default: model's device)
        
        Returns:
            Latent vectors of shape (n_samples, latent_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        return torch.randn(n_samples, self.latent_dim, device=device)
    
    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_output_shape(self) -> Tuple[int, int, int]:
        """Return the output shape (channels, height, width)."""
        return (self.output_channels, self.output_size, self.output_size)


def create_generator(
    latent_dim: int = 100,
    output_size: int = 64,
    output_channels: int = 1
) -> Generator:
    """
    Factory function to create a Generator instance.
    
    Args:
        latent_dim: Dimension of the latent vector z
        output_size: Output image size (64 or 128)
        output_channels: Number of output channels
    
    Returns:
        Configured Generator instance
    """
    return Generator(
        latent_dim=latent_dim,
        output_size=output_size,
        output_channels=output_channels
    )


if __name__ == "__main__":
    # Test the generator
    print("Testing Generator Network")
    print("=" * 50)
    
    # Test 64x64 generator
    gen_64 = Generator(latent_dim=100, output_size=64)
    print(f"\n64x64 Generator:")
    print(f"  Parameters: {gen_64.get_num_params():,}")
    print(f"  Output shape: {gen_64.get_output_shape()}")
    
    z = torch.randn(4, 100)
    output = gen_64(z)
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test 128x128 generator
    gen_128 = Generator(latent_dim=100, output_size=128)
    print(f"\n128x128 Generator:")
    print(f"  Parameters: {gen_128.get_num_params():,}")
    print(f"  Output shape: {gen_128.get_output_shape()}")
    
    z = torch.randn(4, 100)
    output = gen_128(z)
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print("\n✓ All tests passed!")
