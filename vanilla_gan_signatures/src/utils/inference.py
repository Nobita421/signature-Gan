import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add src to path to allow importing generator_vanilla_gan
try:
    from generator_vanilla_gan import Generator
except ImportError:
    # Fallback for when running from different contexts
    sys.path.append(str(Path(__file__).parent.parent))
    from generator_vanilla_gan import Generator

DEFAULT_LATENT_DIM = 100
DEFAULT_IMAGE_SIZE = 64
DEFAULT_IMAGE_CHANNELS = 1

def infer_architecture_from_state_dict(state_dict: Dict[str, Any]) -> Tuple[int, int]:
    """
    Infer latent_dim and image_size from state dict layer shapes.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Tuple of (latent_dim, image_size)
    """
    latent_dim = DEFAULT_LATENT_DIM
    image_size = DEFAULT_IMAGE_SIZE
    
    # Try to infer latent_dim from first linear layer
    for key, tensor in state_dict.items():
        if 'fc' in key and 'weight' in key and len(tensor.shape) == 2:
            latent_dim = tensor.shape[1]
            break
    
    # Try to infer image_size from number of upsample blocks
    upsample_block_count = 0
    for key in state_dict.keys():
        if 'upsample_blocks' in key and '.0.weight' in key:
            block_idx = key.split('.')[1]
            try:
                upsample_block_count = max(upsample_block_count, int(block_idx) + 1)
            except ValueError:
                pass
    
    # 4 blocks = 64x64, 5 blocks = 128x128
    if upsample_block_count >= 5:
        image_size = 128
    else:
        image_size = 64
    
    return latent_dim, image_size

def load_generator(checkpoint_path: str, device: torch.device) -> Tuple[Generator, int]:
    """
    Load a trained generator from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto
        
    Returns:
        Tuple of (Generator model, latent_dim)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration and state dict
    if isinstance(checkpoint, dict):
        config = checkpoint.get('config', {})
        latent_dim = config.get('latent_dim', DEFAULT_LATENT_DIM)
        image_size = config.get('image_size', DEFAULT_IMAGE_SIZE)
        image_channels = config.get('image_channels', DEFAULT_IMAGE_CHANNELS)
        
        if 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            latent_dim, image_size = infer_architecture_from_state_dict(state_dict)
    else:
        state_dict = checkpoint
        latent_dim, image_size = infer_architecture_from_state_dict(state_dict)
        image_channels = DEFAULT_IMAGE_CHANNELS
    
    # Create and load generator
    generator = Generator(
        latent_dim=latent_dim,
        output_size=image_size,
        output_channels=image_channels
    )
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    
    return generator, latent_dim

def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image from [-1, 1] to PIL Image.
    
    Args:
        tensor: Image tensor of shape (C, H, W) in range [-1, 1]
        
    Returns:
        PIL Image object
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    img = tensor.numpy()
    
    # Handle grayscale (1, H, W) -> (H, W)
    if img.shape[0] == 1:
        img = img.squeeze(0)
    else:
        # RGB: (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    
    # Convert from [-1, 1] to [0, 255]
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    if img.ndim == 2:
        return Image.fromarray(img, mode='L')
    else:
        return Image.fromarray(img, mode='RGB')

def generate_signatures_batch(
    generator: Generator,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
    seed: Optional[int] = None,
    batch_size: int = 32,
    progress_callback: Optional[callable] = None,
    noise_scale: float = 1.0
) -> List[Image.Image]:
    """
    Generate signature images in batches.
    
    Args:
        generator: Trained generator model
        n_samples: Number of signatures to generate
        latent_dim: Dimension of latent vector
        device: Device to run inference on
        seed: Optional random seed for reproducibility
        batch_size: Batch size for generation
        progress_callback: Optional callback for progress updates
        noise_scale: Scale factor for the latent noise (diversity)
        
    Returns:
        List of PIL Image objects
    """
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    images: List[Image.Image] = []
    
    with torch.no_grad():
        num_batches = (n_samples + batch_size - 1) // batch_size
        generated_count = 0
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, n_samples - generated_count)
            
            # Generate latent vectors
            z = torch.randn(current_batch_size, latent_dim, device=device) * noise_scale
            
            # Generate images
            fake_images = generator(z)
            
            # Convert to PIL images
            for i in range(current_batch_size):
                pil_image = tensor_to_pil_image(fake_images[i])
                images.append(pil_image)
                generated_count += 1
            
            # Update progress
            if progress_callback is not None:
                progress_callback(generated_count / n_samples)
    
    return images
