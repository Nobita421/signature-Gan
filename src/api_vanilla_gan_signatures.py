"""
FastAPI REST API for Vanilla GAN Signature Generation.

Provides endpoints to generate synthetic signature images from a trained
Vanilla GAN generator model.

Endpoints:
    GET  /health   - Health check
    POST /generate - Generate signatures (returns ZIP or base64 JSON)
    GET  /info     - Model information

Usage:
    uvicorn api_vanilla_gan_signatures:app --host 0.0.0.0 --port 8000
    
    Or run directly:
    python api_vanilla_gan_signatures.py
"""

import io
import os
import sys
import base64
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from generator_vanilla_gan import Generator


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CHECKPOINT_PATH = str(
    Path(__file__).parent.parent / "checkpoints" / "G_final.pth"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_SAMPLES = 1000  # Maximum number of samples per request


# =============================================================================
# Global State
# =============================================================================

class ModelState:
    """Global state for the loaded generator model."""
    generator: Optional[Generator] = None
    device: torch.device = torch.device("cpu")
    latent_dim: int = 100
    image_size: int = 64
    image_channels: int = 1
    checkpoint_path: Optional[str] = None
    loaded_at: Optional[str] = None


model_state = ModelState()


# =============================================================================
# Model Loading
# =============================================================================

def load_generator_from_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Generator:
    """
    Load a trained generator from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        device: Device to load the model onto
        
    Returns:
        Generator model loaded with trained weights in eval mode
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint format is invalid
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model configuration from checkpoint if available
    if isinstance(checkpoint, dict):
        config = checkpoint.get('config', {})
        latent_dim = config.get('latent_dim', 100)
        image_size = config.get('image_size', 64)
        image_channels = config.get('image_channels', 1)
        
        # Get state dict
        if 'generator_state_dict' in checkpoint:
            state_dict = checkpoint['generator_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            latent_dim, image_size = _infer_architecture_from_state_dict(state_dict)
    else:
        state_dict = checkpoint
        latent_dim, image_size = _infer_architecture_from_state_dict(state_dict)
        image_channels = 1
    
    # Update global state
    model_state.latent_dim = latent_dim
    model_state.image_size = image_size
    model_state.image_channels = image_channels
    
    # Create generator with inferred/loaded configuration
    generator = Generator(
        latent_dim=latent_dim,
        output_size=image_size,
        output_channels=image_channels
    )
    
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    
    return generator


def _infer_architecture_from_state_dict(state_dict: Dict[str, Any]) -> tuple:
    """
    Infer latent_dim and image_size from state dict layer shapes.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Tuple of (latent_dim, image_size)
    """
    latent_dim = 100
    image_size = 64
    
    for key, tensor in state_dict.items():
        if 'fc' in key and 'weight' in key and len(tensor.shape) == 2:
            latent_dim = tensor.shape[1]
            break
    
    upsample_block_count = 0
    for key in state_dict.keys():
        if 'upsample_blocks' in key and '.0.weight' in key:
            block_idx = key.split('.')[1]
            try:
                upsample_block_count = max(upsample_block_count, int(block_idx) + 1)
            except ValueError:
                pass
    
    if upsample_block_count >= 5:
        image_size = 128
    else:
        image_size = 64
    
    return latent_dim, image_size


# =============================================================================
# Image Processing Utilities
# =============================================================================

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor image from [-1, 1] to [0, 255] uint8 numpy array.
    
    Args:
        tensor: Image tensor of shape (C, H, W) in range [-1, 1]
        
    Returns:
        Numpy array of shape (H, W) for grayscale or (H, W, C) for color
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    img = tensor.numpy()
    
    if img.shape[0] == 1:
        img = img.squeeze(0)
    else:
        img = np.transpose(img, (1, 2, 0))
    
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    img_array = tensor_to_image(tensor)
    if img_array.ndim == 2:
        return Image.fromarray(img_array, mode='L')
    return Image.fromarray(img_array, mode='RGB')


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_signatures_batch(
    n_samples: int,
    seed: Optional[int] = None,
    batch_size: int = 64
) -> List[Image.Image]:
    """
    Generate a batch of signature images.
    
    Args:
        n_samples: Number of signatures to generate
        seed: Optional random seed for reproducibility
        batch_size: Batch size for generation
        
    Returns:
        List of PIL Images
    """
    if model_state.generator is None:
        raise RuntimeError("Generator model not loaded")
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    images: List[Image.Image] = []
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < n_samples:
            current_batch_size = min(batch_size, n_samples - generated_count)
            
            z = torch.randn(
                current_batch_size,
                model_state.latent_dim,
                device=model_state.device
            )
            
            fake_images = model_state.generator(z)
            
            for i in range(current_batch_size):
                pil_img = tensor_to_pil(fake_images[i])
                images.append(pil_img)
                generated_count += 1
    
    return images


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    """Request model for signature generation."""
    n: int = Field(
        default=1,
        ge=1,
        le=MAX_SAMPLES,
        description=f"Number of signatures to generate (1-{MAX_SAMPLES})"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    format: str = Field(
        default="zip",
        description="Output format: 'zip' for ZIP file, 'base64' for JSON with base64 images"
    )


class GenerateBase64Response(BaseModel):
    """Response model for base64-encoded images."""
    success: bool
    n_generated: int
    seed: Optional[int]
    format: str = "base64"
    images: List[str] = Field(description="List of base64-encoded PNG images")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    latent_dim: int
    image_size: int
    image_channels: int
    checkpoint_path: Optional[str]
    loaded_at: Optional[str]
    device: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Loads the generator model on startup.
    """
    # Startup: Load model
    print("=" * 60)
    print("Starting Vanilla GAN Signature API")
    print("=" * 60)
    
    # Determine device
    model_state.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {model_state.device}")
    
    # Try to load checkpoint
    checkpoint_path = os.environ.get("GAN_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)
    
    if os.path.exists(checkpoint_path):
        try:
            model_state.generator = load_generator_from_checkpoint(
                checkpoint_path,
                model_state.device
            )
            model_state.checkpoint_path = checkpoint_path
            model_state.loaded_at = datetime.now().isoformat()
            print(f"✓ Generator loaded from: {checkpoint_path}")
            print(f"  - Latent dim: {model_state.latent_dim}")
            print(f"  - Image size: {model_state.image_size}x{model_state.image_size}")
            print(f"  - Channels: {model_state.image_channels}")
        except Exception as e:
            print(f"✗ Failed to load generator: {e}")
            print("  API will start but /generate endpoint will not work")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Set GAN_CHECKPOINT_PATH environment variable or place checkpoint at:")
        print(f"  {DEFAULT_CHECKPOINT_PATH}")
        print("  API will start but /generate endpoint will not work")
    
    print("=" * 60)
    print(f"API ready at http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print("=" * 60)
    
    yield
    
    # Shutdown: Cleanup
    print("Shutting down Vanilla GAN Signature API...")
    model_state.generator = None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Vanilla GAN Signature Generator API",
    description=(
        "REST API for generating synthetic signature images using a trained "
        "Vanilla GAN generator model."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and if the model is loaded."
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_state.generator is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get information about the loaded generator model."
)
async def get_model_info() -> ModelInfoResponse:
    """Get model information endpoint."""
    return ModelInfoResponse(
        latent_dim=model_state.latent_dim,
        image_size=model_state.image_size,
        image_channels=model_state.image_channels,
        checkpoint_path=model_state.checkpoint_path,
        loaded_at=model_state.loaded_at,
        device=str(model_state.device)
    )


@app.post(
    "/generate",
    summary="Generate Signatures",
    description=(
        "Generate synthetic signature images. "
        "Returns either a ZIP file or JSON with base64-encoded images."
    ),
    responses={
        200: {
            "description": "Generated signatures",
            "content": {
                "application/zip": {},
                "application/json": {"model": GenerateBase64Response}
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    }
)
async def generate_signatures(request: GenerateRequest) -> Union[StreamingResponse, JSONResponse]:
    """
    Generate synthetic signature images.
    
    Args:
        request: Generation request with n (number), seed (optional), and format
        
    Returns:
        ZIP file or JSON with base64-encoded images
    """
    # Check if model is loaded
    if model_state.generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator model not loaded. Please ensure checkpoint is available."
        )
    
    # Validate format
    if request.format not in ("zip", "base64"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{request.format}'. Use 'zip' or 'base64'."
        )
    
    try:
        # Generate images
        images = generate_signatures_batch(
            n_samples=request.n,
            seed=request.seed
        )
        
        if request.format == "base64":
            # Return JSON with base64-encoded images
            base64_images = [pil_to_base64(img) for img in images]
            return JSONResponse(
                content={
                    "success": True,
                    "n_generated": len(images),
                    "seed": request.seed,
                    "format": "base64",
                    "images": base64_images
                }
            )
        else:
            # Return ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for idx, img in enumerate(images):
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    zf.writestr(f"signature_{idx + 1:06d}.png", img_buffer.getvalue())
            
            zip_buffer.seek(0)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signatures_{request.n}_{timestamp}.zip"
            
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.post(
    "/generate/single",
    summary="Generate Single Signature",
    description="Generate a single signature image and return as PNG.",
    responses={
        200: {
            "description": "Generated signature PNG",
            "content": {"image/png": {}}
        },
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    }
)
async def generate_single_signature(
    seed: Optional[int] = Query(default=None, description="Random seed for reproducibility")
) -> Response:
    """
    Generate a single signature image.
    
    Args:
        seed: Optional random seed for reproducibility
        
    Returns:
        PNG image
    """
    if model_state.generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator model not loaded. Please ensure checkpoint is available."
        )
    
    try:
        images = generate_signatures_batch(n_samples=1, seed=seed)
        
        img_buffer = io.BytesIO()
        images[0].save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'inline; filename="signature.png"'}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment or use defaults
    host = os.environ.get("API_HOST", DEFAULT_HOST)
    port = int(os.environ.get("API_PORT", DEFAULT_PORT))
    
    print(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "api_vanilla_gan_signatures:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
