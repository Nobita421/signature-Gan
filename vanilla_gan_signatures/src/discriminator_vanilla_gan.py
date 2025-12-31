"""
Discriminator Network for Vanilla GAN - Signature Generation

This module implements the Discriminator network that classifies images
as real or fake using convolutional layers with downsampling.

Architecture:
    Input: 64×64×1 or 128×128×1 grayscale signature image
    Output: Probability that input is real (sigmoid)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple


class DownsampleBlock(nn.Module):
    """
    Downsampling block consisting of Conv2D + LeakyReLU + Dropout.
    
    This block halves the spatial dimensions while increasing channel depth.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        use_batch_norm: Whether to apply batch normalization
        use_spectral_norm: Whether to apply spectral normalization
        dropout: Dropout probability (0 to disable)
        leaky_slope: Negative slope for LeakyReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batch_norm: bool = False,
        use_spectral_norm: bool = False,
        dropout: float = 0.25,
        leaky_slope: float = 0.2
    ) -> None:
        super().__init__()
        
        # Create convolutional layer
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm
        )
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            conv = spectral_norm(conv)
        
        layers = [conv]
        
        # Batch normalization (not recommended with spectral norm)
        if use_batch_norm and not use_spectral_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
        
        # Dropout for regularization
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling block."""
        return self.block(x)


class Discriminator(nn.Module):
    """
    Discriminator network for Vanilla GAN signature classification.
    
    Classifies images as real or fake using convolutional layers with
    stride-2 downsampling.
    
    Architecture for 64×64 input:
        64×64×1 → 32×32×64 → 16×16×128 → 8×8×256 → 4×4×512 → Flatten → Dense(1)
    
    Architecture for 128×128 input:
        128×128×1 → 64×64×64 → 32×32×128 → 16×16×256 → 8×8×512 → 4×4×512 → Flatten → Dense(1)
    
    Args:
        input_size: Input image size, either 64 or 128 (default: 64)
        input_channels: Number of input channels (default: 1 for grayscale)
        use_spectral_norm: Whether to use spectral normalization for stability (default: False)
        dropout: Dropout probability (default: 0.25)
        leaky_slope: Negative slope for LeakyReLU (default: 0.2)
    
    Example:
        >>> discriminator = Discriminator(input_size=64)
        >>> images = torch.randn(16, 1, 64, 64)  # Batch of 16 images
        >>> predictions = discriminator(images)
        >>> print(predictions.shape)  # torch.Size([16, 1])
    """
    
    def __init__(
        self,
        input_size: int = 64,
        input_channels: int = 1,
        use_spectral_norm: bool = False,
        dropout: float = 0.25,
        leaky_slope: float = 0.2
    ) -> None:
        super().__init__()
        
        if input_size not in [64, 128]:
            raise ValueError(f"input_size must be 64 or 128, got {input_size}")
        
        self.input_size = input_size
        self.input_channels = input_channels
        self.use_spectral_norm = use_spectral_norm
        self.dropout = dropout
        self.leaky_slope = leaky_slope
        
        # Build downsampling blocks based on input size
        if input_size == 64:
            # 64x64x1 → 32x32x64 → 16x16x128 → 8x8x256 → 4x4x512
            self.conv_blocks = nn.Sequential(
                DownsampleBlock(
                    input_channels, 64,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 64x64 → 32x32
                DownsampleBlock(
                    64, 128,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 32x32 → 16x16
                DownsampleBlock(
                    128, 256,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 16x16 → 8x8
                DownsampleBlock(
                    256, 512,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 8x8 → 4x4
            )
            final_features = 512 * 4 * 4
        else:  # 128x128
            # 128x128x1 → 64x64x64 → 32x32x128 → 16x16x256 → 8x8x512 → 4x4x512
            self.conv_blocks = nn.Sequential(
                DownsampleBlock(
                    input_channels, 64,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 128x128 → 64x64
                DownsampleBlock(
                    64, 128,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 64x64 → 32x32
                DownsampleBlock(
                    128, 256,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 32x32 → 16x16
                DownsampleBlock(
                    256, 512,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 16x16 → 8x8
                DownsampleBlock(
                    512, 512,
                    use_spectral_norm=use_spectral_norm,
                    dropout=dropout,
                    leaky_slope=leaky_slope
                ),  # 8x8 → 4x4
            )
            final_features = 512 * 4 * 4
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Final classification layer
        fc = nn.Linear(final_features, 1)
        if use_spectral_norm:
            fc = spectral_norm(fc)
        
        self.classifier = nn.Sequential(
            fc,
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using normal distribution with std=0.02.
        
        Following DCGAN paper recommendations for weight initialization.
        Note: spectral_norm wraps the weight, so we need to handle it specially.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Conv2d):
            # Check if module has been wrapped by spectral_norm
            if hasattr(module, 'weight_orig'):
                nn.init.normal_(module.weight_orig, mean=0.0, std=0.02)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if hasattr(module, 'weight_orig'):
                nn.init.normal_(module.weight_orig, mean=0.0, std=0.02)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classify image as real or fake.
        
        Args:
            x: Input images of shape (batch_size, input_channels, input_size, input_size)
        
        Returns:
            Predictions of shape (batch_size, 1), values in [0, 1]
        """
        # Pass through convolutional blocks
        x = self.conv_blocks(x)
        
        # Flatten spatial dimensions
        x = self.flatten(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier (useful for feature matching loss).
        
        Args:
            x: Input images of shape (batch_size, input_channels, input_size, input_size)
        
        Returns:
            Feature vector of shape (batch_size, final_features)
        """
        x = self.conv_blocks(x)
        x = self.flatten(x)
        return x
    
    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Return the expected input shape (channels, height, width)."""
        return (self.input_channels, self.input_size, self.input_size)


class MinibatchDiscrimination(nn.Module):
    """
    Minibatch discrimination layer for improved training stability.
    
    Helps the discriminator look at multiple examples in combination,
    which can help prevent mode collapse.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        kernel_dims: Dimensions of the minibatch kernel
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_dims: int = 5
    ) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        
        self.T = nn.Parameter(
            torch.randn(in_features, out_features, kernel_dims) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with minibatch discrimination.
        
        Args:
            x: Input features of shape (batch_size, in_features)
        
        Returns:
            Output features of shape (batch_size, in_features + out_features)
        """
        # x: (batch_size, in_features)
        # T: (in_features, out_features, kernel_dims)
        
        # Compute M_i = x_i * T
        M = torch.einsum('bi,iok->bok', x, self.T)  # (batch_size, out_features, kernel_dims)
        
        # Compute L1 distance between samples
        M_expanded = M.unsqueeze(0)  # (1, batch_size, out_features, kernel_dims)
        M_transposed = M.unsqueeze(1)  # (batch_size, 1, out_features, kernel_dims)
        
        # L1 distance: |M_i - M_j|
        diff = torch.abs(M_expanded - M_transposed).sum(dim=3)  # (batch_size, batch_size, out_features)
        
        # c(x_i, x_j) = exp(-||M_i - M_j||_1)
        c = torch.exp(-diff)  # (batch_size, batch_size, out_features)
        
        # o(x_i) = sum_j c(x_i, x_j)
        o = c.sum(dim=1)  # (batch_size, out_features)
        
        # Concatenate with input
        return torch.cat([x, o], dim=1)


def create_discriminator(
    input_size: int = 64,
    input_channels: int = 1,
    use_spectral_norm: bool = False,
    dropout: float = 0.25
) -> Discriminator:
    """
    Factory function to create a Discriminator instance.
    
    Args:
        input_size: Input image size (64 or 128)
        input_channels: Number of input channels
        use_spectral_norm: Whether to use spectral normalization
        dropout: Dropout probability
    
    Returns:
        Configured Discriminator instance
    """
    return Discriminator(
        input_size=input_size,
        input_channels=input_channels,
        use_spectral_norm=use_spectral_norm,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the discriminator
    print("Testing Discriminator Network")
    print("=" * 50)
    
    # Test 64x64 discriminator without spectral norm
    disc_64 = Discriminator(input_size=64, use_spectral_norm=False)
    print(f"\n64x64 Discriminator (no spectral norm):")
    print(f"  Parameters: {disc_64.get_num_params():,}")
    print(f"  Input shape: {disc_64.get_input_shape()}")
    
    images = torch.randn(4, 1, 64, 64)
    output = disc_64(images)
    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test 64x64 discriminator with spectral norm
    disc_64_sn = Discriminator(input_size=64, use_spectral_norm=True)
    print(f"\n64x64 Discriminator (with spectral norm):")
    print(f"  Parameters: {disc_64_sn.get_num_params():,}")
    
    output_sn = disc_64_sn(images)
    print(f"  Output shape: {output_sn.shape}")
    print(f"  Output range: [{output_sn.min().item():.3f}, {output_sn.max().item():.3f}]")
    
    # Test 128x128 discriminator
    disc_128 = Discriminator(input_size=128, use_spectral_norm=False)
    print(f"\n128x128 Discriminator:")
    print(f"  Parameters: {disc_128.get_num_params():,}")
    print(f"  Input shape: {disc_128.get_input_shape()}")
    
    images_128 = torch.randn(4, 1, 128, 128)
    output_128 = disc_128(images_128)
    print(f"  Input shape: {images_128.shape}")
    print(f"  Output shape: {output_128.shape}")
    print(f"  Output range: [{output_128.min().item():.3f}, {output_128.max().item():.3f}]")
    
    # Test feature extraction
    features = disc_64.forward_features(images)
    print(f"\nFeature extraction test:")
    print(f"  Feature shape: {features.shape}")
    
    print("\n✓ All tests passed!")
