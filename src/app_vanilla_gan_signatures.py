"""
Streamlit Web Application for Signature Generation.

A web interface for generating synthetic signatures using a trained Vanilla GAN.

Features:
- Sidebar controls for generation parameters
- Cached model loading for fast startup
- Gallery display of generated signatures
- ZIP download of generated images
- Progress tracking during generation

Usage:
    streamlit run app_vanilla_gan_signatures.py
"""

import io
import os
import sys
import zipfile
import subprocess
import time
import json
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
from PIL import Image
import pandas as pd

try:
    # Lightweight helper for periodic reruns during long processes.
    # Avoids Streamlit's file-watcher (which can be unstable with PyTorch on Windows).
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from inference utils
from utils.inference import (
    load_generator as load_generator_inference,
    generate_signatures_batch,
    DEFAULT_LATENT_DIM,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_CHANNELS
)
from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_SAMPLE_DIR = Path(__file__).parent.parent / "samples"
DEFAULT_RUNS_DIR = Path(__file__).parent.parent / "runs"

# State file for persisting training run info across page reloads
TRAINING_STATE_FILE = Path(__file__).parent.parent / ".training_state.json"


# -----------------------------------------------------------------------------
# Manual test checklist
# -----------------------------------------------------------------------------
# 1) Load from checkpoints/ and runs/ (dropdown)
# 2) Enter a custom path outside project folders -> rejected by default
# 3) Enable unsafe mode -> custom path allowed (warning shown)
# 4) Generate a large batch, click Cancel mid-run -> stops within one batch
# 5) Refresh while training, on Linux/macOS PID check works and state clears if PID not running


def _save_training_state(run_dir: str, pid: int, stop_file: str) -> None:
    """Save training state to file for persistence across reloads."""
    try:
        state = {
            "run_dir": run_dir,
            "pid": pid,
            "stop_file": stop_file,
            "started_at": time.time(),
        }
        TRAINING_STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


def _is_pid_running(pid: int) -> bool:
    """Check if a process with given PID is running (cross-platform)."""
    if not pid or pid <= 0:
        return False
    # Prefer platform-native checks without extra deps.
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in result.stdout
        # Unix/macOS: signal 0 does not kill, only checks existence/permission.
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but we may not have permission to signal it.
        return True
    except Exception:
        return False


def _allowed_checkpoint_roots() -> List[Path]:
    return [DEFAULT_CHECKPOINT_DIR.resolve(), DEFAULT_RUNS_DIR.resolve()]


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path_resolved = path.resolve()
        root_resolved = root.resolve()
        path_resolved.relative_to(root_resolved)
        return True
    except Exception:
        return False


def _is_checkpoint_path_allowed(path: Path) -> bool:
    """Return True if path is within allowed checkpoint roots."""
    for root in _allowed_checkpoint_roots():
        if _is_under_root(path, root):
            return True
    return False


def _validate_checkpoint_path(user_path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Validate a user-provided checkpoint path exists and is a plausible checkpoint file."""
    try:
        p = Path(user_path).expanduser()
        if not p.exists() or not p.is_file():
            return None, "File not found"
        if p.suffix.lower() not in {".pt", ".pth"}:
            return None, "Not a .pt/.pth checkpoint file"
        return p.resolve(), None
    except Exception:
        return None, "Invalid path"


def _torch_load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
    trusted_for_loading: bool,
    weights_only: Optional[bool] = None,
) -> Any:
    """Load a checkpoint with guardrails.

    IMPORTANT: PyTorch checkpoints are pickle-based. This function must only be used
    for trusted paths (project checkpoints/runs, or explicitly enabled unsafe mode).
    """
    if not trusted_for_loading:
        raise ValueError(
            "Refusing to load an untrusted checkpoint. Enable Unsafe mode to proceed."
        )

    # Prefer weights_only=True where supported to reduce unpickling surface.
    if weights_only is None:
        return torch.load(str(checkpoint_path), map_location=map_location)
    try:
        return torch.load(
            str(checkpoint_path), map_location=map_location, weights_only=weights_only
        )
    except TypeError:
        # Older PyTorch: no weights_only arg.
        return torch.load(str(checkpoint_path), map_location=map_location)


def _load_training_state() -> Optional[dict]:
    """Load training state from file and verify process is running."""
    try:
        if not TRAINING_STATE_FILE.exists():
            return None
        
        state = json.loads(TRAINING_STATE_FILE.read_text(encoding="utf-8"))
        pid = state.get("pid")
        
        if pid and _is_pid_running(pid):
            state["is_running"] = True
            return state
        else:
            # Process not running, clean up state file
            _clear_training_state()
            return None
    except Exception:
        return None


def _clear_training_state() -> None:
    """Clear training state file."""
    try:
        if TRAINING_STATE_FILE.exists():
            TRAINING_STATE_FILE.unlink()
    except Exception:
        pass

# UI defaults
TRAINING_TERMINAL_TAIL_LINES = 250
TRAINING_RECENT_LOGS_LINES = 100
TRAINING_RECENT_LOGS_HEIGHT = 480
TRAINING_TERMINAL_HEIGHT = 400

# Default training hyperparameters (for session state persistence)
DEFAULT_TRAIN_CONFIG = {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 0.0002,
    "latent_dim": 100,
    "image_size": 64,
}


def _get_gpu_memory_info() -> Optional[dict]:
    """Get GPU memory usage if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "percent_used": (allocated / total) * 100 if total > 0 else 0,
        }
    except Exception:
        return None


def _parse_training_progress(log_text: str) -> dict:
    """Parse training progress from log text."""
    import re
    
    progress = {
        "current_epoch": 0,
        "total_epochs": 0,
        "current_batch": 0,
        "total_batches": 0,
        "g_loss": None,
        "d_loss": None,
        "d_real": None,
        "d_fake": None,
    }
    
    lines = log_text.strip().split("\n")
    
    # Process lines from end to find most recent data
    for line in reversed(lines[-100:]):
        # Match tqdm-style output: "Epoch 3/20:  60%|...| 12/20 [..., D=1.3189, G=0.9265, D(r)=0.436, D(f)=0.383]"
        epoch_match = re.search(r"Epoch\s*(\d+)/(\d+)", line)
        if epoch_match and progress["current_epoch"] == 0:
            progress["current_epoch"] = int(epoch_match.group(1))
            progress["total_epochs"] = int(epoch_match.group(2))
        
        # Match batch progress: "12/20" after the percentage
        batch_match = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
        if batch_match and progress["current_batch"] == 0:
            progress["current_batch"] = int(batch_match.group(1))
            progress["total_batches"] = int(batch_match.group(2))
        
        # Match G= and D= in tqdm postfix: "D=1.3189, G=0.9265"
        g_match = re.search(r"G=(\d+\.\d+)", line)
        if g_match and progress["g_loss"] is None:
            progress["g_loss"] = float(g_match.group(1))
        
        d_match = re.search(r"D=(\d+\.\d+)", line)
        if d_match and progress["d_loss"] is None:
            progress["d_loss"] = float(d_match.group(1))
        
        # Match D(r) and D(f): "D(r)=0.436, D(f)=0.383"
        dr_match = re.search(r"D\(r\)=(\d+\.\d+)", line)
        if dr_match and progress["d_real"] is None:
            progress["d_real"] = float(dr_match.group(1))
        
        df_match = re.search(r"D\(f\)=(\d+\.\d+)", line)
        if df_match and progress["d_fake"] is None:
            progress["d_fake"] = float(df_match.group(1))
        
        # Also match epoch summary format: "[Epoch 0003] G_loss: 0.8169 | D_loss: 1.3451"
        summary_g = re.search(r"G_loss:\s*(\d+\.\d+)", line)
        if summary_g and progress["g_loss"] is None:
            progress["g_loss"] = float(summary_g.group(1))
        
        summary_d = re.search(r"D_loss:\s*(\d+\.\d+)", line)
        if summary_d and progress["d_loss"] is None:
            progress["d_loss"] = float(summary_d.group(1))
        
        summary_dr = re.search(r"D\(real\):\s*(\d+\.\d+)", line)
        if summary_dr and progress["d_real"] is None:
            progress["d_real"] = float(summary_dr.group(1))
    
    return progress


def _check_loss_health(df: Optional[pd.DataFrame]) -> dict:
    """Analyze loss trends to detect training issues."""
    status = {
        "healthy": True,
        "warnings": [],
        "status_emoji": "✅",
    }
    
    if df is None or len(df) < 5:
        status["status_emoji"] = "⏳"
        return status
    
    recent = df.tail(10)
    
    # Check for NaN losses
    if recent["g_loss"].isna().any() or recent["d_loss"].isna().any():
        status["healthy"] = False
        status["warnings"].append("⚠️ NaN losses detected - training may have diverged")
        status["status_emoji"] = "🔴"
        return status
    
    # Check for exploding losses
    if recent["g_loss"].max() > 50 or recent["d_loss"].max() > 50:
        status["healthy"] = False
        status["warnings"].append("⚠️ Loss values very high - consider lowering learning rate")
        status["status_emoji"] = "🟠"
    
    # Check for mode collapse (D loss very low, G loss very high)
    if recent["d_loss"].mean() < 0.1 and recent["g_loss"].mean() > 5:
        status["healthy"] = False
        status["warnings"].append("⚠️ Possible mode collapse - discriminator too strong")
        status["status_emoji"] = "🟠"
    
    # Check for vanishing gradients (both losses stuck)
    if len(df) > 20:
        last_20 = df.tail(20)
        if last_20["g_loss"].std() < 0.01 and last_20["d_loss"].std() < 0.01:
            status["warnings"].append("⚠️ Losses appear stuck - training may have stalled")
            status["status_emoji"] = "🟡"
    
    return status


def _new_run_id(prefix: str = "train") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _list_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _count_images(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    try:
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)
    except Exception:
        return 0


def _tail_text_file(path: Path, max_chars: int = 60_000) -> str:
    """Read the tail of a text file without loading it all into memory."""
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_chars)
            f.seek(start)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        # If we started mid-line, drop the partial first line.
        if start > 0:
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1 :]
        return text
    except Exception:
        return ""


def _load_latest_metrics_json(log_dir: Path) -> Optional[Path]:
    """Find the latest metrics log file (JSON or CSV)."""
    if not log_dir.exists():
        return None
    # Try JSON first
    candidates = list(log_dir.glob("*_log.json"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _load_latest_metrics_csv(log_dir: Path) -> Optional[Path]:
    """Find the latest metrics CSV file."""
    if not log_dir.exists():
        return None
    candidates = list(log_dir.glob("*_metrics.csv"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _metrics_dataframe_from_logger_json(path: Path) -> Optional[pd.DataFrame]:
    """Load metrics from JSON log file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), list):
            return pd.DataFrame(payload["metrics"])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
    except Exception:
        return None
    return None


def _metrics_dataframe_from_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load metrics from CSV file."""
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception:
        return None


def _parse_metrics_from_log(log_text: str) -> Optional[pd.DataFrame]:
    """Parse epoch summaries from training log to build a metrics dataframe."""
    import re
    
    metrics = []
    lines = log_text.split('\n')
    
    for line in lines:
        # Match epoch summary: "[Epoch 0003] G_loss: 0.8169 | D_loss: 1.3451 | D(real): 0.5401 | D(fake): 0.4739"
        match = re.search(
            r'\[Epoch\s*(\d+)\]\s*G_loss:\s*([\d.]+)\s*\|\s*D_loss:\s*([\d.]+)\s*\|\s*D\(real\):\s*([\d.]+)\s*\|\s*D\(fake\):\s*([\d.]+)',
            line
        )
        if match:
            metrics.append({
                'epoch': int(match.group(1)),
                'g_loss': float(match.group(2)),
                'd_loss': float(match.group(3)),
                'd_real': float(match.group(4)),
                'd_fake': float(match.group(5)),
            })
    
    if metrics:
        return pd.DataFrame(metrics)
    return None


# -----------------------------------------------------------------------------
# Model Loading Functions
# -----------------------------------------------------------------------------

def _checkpoint_has_discriminator(path: str, trusted_for_loading: bool) -> bool:
    """Check whether a checkpoint dict contains discriminator weights.

    Note: PyTorch checkpoint loading is pickle-based. This function intentionally
    uses a guarded load and should only be called for trusted checkpoint paths.
    """
    try:
        p = Path(path)
        checkpoint = _torch_load_checkpoint(
            p,
            map_location="cpu",
            trusted_for_loading=trusted_for_loading,
            weights_only=True,
        )
        return isinstance(checkpoint, dict) and (
            "discriminator_state_dict" in checkpoint
        )
    except Exception:
        return False


@st.cache_data
def _cached_checkpoint_has_discriminator(path: str, trusted_for_loading: bool) -> bool:
    """Cached version of discriminator check to avoid repeated file reads."""
    return _checkpoint_has_discriminator(path, trusted_for_loading)


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """
    Find all available checkpoint files in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        List of paths to checkpoint files (.pth or .pt)
    """
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    # Search for .pth and .pt files recursively
    for ext in ['*.pth', '*.pt']:
        checkpoints.extend(checkpoint_dir.rglob(ext))
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoints


@st.cache_resource
def load_generator(checkpoint_path: str) -> Tuple[Generator, int, torch.device]:
    """
    Load a trained generator from checkpoint with caching.
    
    This function is cached using st.cache_resource to avoid reloading
    the model on every rerun.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (Generator model, latent_dim, device)
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use inference utility
    generator, latent_dim = load_generator_inference(checkpoint_path, device)
    
    return generator, latent_dim, device


@st.cache_data
def get_model_metadata(checkpoint_path: str, trusted_for_loading: bool) -> dict:
    """
    Extract metadata from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dictionary containing epoch, loss, and config
    """
    try:
        p = Path(checkpoint_path)
        checkpoint = _torch_load_checkpoint(
            p,
            map_location="cpu",
            trusted_for_loading=trusted_for_loading,
            weights_only=True,
        )
        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})
            # Keep metadata small/safe for display.
            if not isinstance(config, dict):
                config = {}
            return {
                "epoch": checkpoint.get("epoch", "Unknown"),
                "loss": checkpoint.get("best_g_loss", "N/A"),
                "config": config,
            }
    except Exception:
        return {}
    return {}


@st.cache_resource
def load_discriminator(
    checkpoint_path: str,
    device: torch.device,
    image_size: int = 64,
    trusted_for_loading: bool = True,
) -> Optional[Discriminator]:
    """
    Load a trained discriminator from checkpoint with caching.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model onto
        image_size: Input image size
        
    Returns:
        Discriminator model or None if not found in checkpoint
    """
    try:
        p = Path(checkpoint_path)
        checkpoint = _torch_load_checkpoint(
            p,
            map_location=device,
            trusted_for_loading=trusted_for_loading,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "discriminator_state_dict" not in checkpoint:
            return None

        discriminator = Discriminator(input_size=image_size)
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        discriminator.to(device)
        discriminator.eval()
        return discriminator
    except Exception:
        return None


def create_zip_archive_from_paths(
    image_paths: List[str],
    prefix: str = "signature",
    format: str = "PNG",
    quality: int = 95,
    selected_indices: Optional[List[int]] = None,
    filename_template: str = "{prefix}_{index:03d}",
) -> bytes:
    """Create a ZIP archive from image file paths (loads one-by-one)."""
    zip_buffer = io.BytesIO()
    if selected_indices is not None:
        items = [(idx, image_paths[idx]) for idx in selected_indices if 0 <= idx < len(image_paths)]
    else:
        items = list(enumerate(image_paths))

    ext = "jpg" if format.upper() == "JPEG" else "png"
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for new_idx, (orig_idx, path) in enumerate(items, start=1):
            img_buffer = io.BytesIO()
            with Image.open(path) as opened:
                img = opened.copy()

            if format.upper() == "JPEG":
                if img.mode == "RGBA":
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_buffer, format="JPEG", quality=quality)
            else:
                img.save(img_buffer, format="PNG")

            img_buffer.seek(0)
            filename = (
                filename_template.format(prefix=prefix, index=new_idx, total=len(items))
                + f".{ext}"
            )
            zip_file.writestr(filename, img_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def save_images_to_folder_from_paths(
    image_paths: List[str],
    output_dir: str,
    prefix: str = "signature",
    format: str = "PNG",
    quality: int = 95,
    selected_indices: Optional[List[int]] = None,
    filename_template: str = "{prefix}_{index:03d}",
) -> Tuple[int, str]:
    """Save/copy images from paths to a folder (loads one-by-one)."""
    os.makedirs(output_dir, exist_ok=True)
    if selected_indices is not None:
        items = [(idx, image_paths[idx]) for idx in selected_indices if 0 <= idx < len(image_paths)]
    else:
        items = list(enumerate(image_paths))

    ext = "jpg" if format.upper() == "JPEG" else "png"
    saved_count = 0
    for new_idx, (_, src_path) in enumerate(items, start=1):
        with Image.open(src_path) as opened:
            img = opened.copy()
        filename = (
            filename_template.format(prefix=prefix, index=new_idx, total=len(items))
            + f".{ext}"
        )
        dst_path = os.path.join(output_dir, filename)
        if format.upper() == "JPEG":
            if img.mode == "RGBA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dst_path, format="JPEG", quality=quality)
        else:
            img.save(dst_path, format="PNG")
        saved_count += 1

    return saved_count, output_dir


# -----------------------------------------------------------------------------
# Generation Functions
# -----------------------------------------------------------------------------

def create_zip_archive(
    images: List[Image.Image], 
    prefix: str = "signature",
    format: str = "PNG",
    quality: int = 95,
    selected_indices: Optional[List[int]] = None,
    filename_template: str = "{prefix}_{index:03d}"
) -> bytes:
    """
    Create a ZIP archive containing generated images.
    
    Args:
        images: List of PIL Image objects
        prefix: Filename prefix for images
        format: Image format ('PNG' or 'JPEG')
        quality: JPEG quality (1-100), ignored for PNG
        selected_indices: Optional list of indices to include (None = all)
        filename_template: Template for filenames. Supports {prefix}, {index}, {total}
        
    Returns:
        ZIP archive as bytes
    """
    zip_buffer = io.BytesIO()
    
    # Determine which images to include
    if selected_indices is not None:
        items = [(idx, images[idx]) for idx in selected_indices if 0 <= idx < len(images)]
    else:
        items = list(enumerate(images))
    
    ext = "jpg" if format.upper() == "JPEG" else "png"
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for new_idx, (orig_idx, img) in enumerate(items, start=1):
            img_buffer = io.BytesIO()
            
            # Convert RGBA to RGB for JPEG
            if format.upper() == "JPEG" and img.mode == "RGBA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            elif format.upper() == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")
            
            if format.upper() == "JPEG":
                img.save(img_buffer, format='JPEG', quality=quality)
            else:
                img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            filename = filename_template.format(
                prefix=prefix, 
                index=new_idx, 
                total=len(items)
            ) + f".{ext}"
            zip_file.writestr(filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def save_images_to_folder(
    images: List[Image.Image],
    output_dir: str,
    prefix: str = "signature",
    format: str = "PNG",
    quality: int = 95,
    selected_indices: Optional[List[int]] = None,
    filename_template: str = "{prefix}_{index:03d}"
) -> Tuple[int, str]:
    """
    Save generated images directly to a folder on disk.
    
    Args:
        images: List of PIL Image objects
        output_dir: Directory to save images to
        prefix: Filename prefix
        format: Image format ('PNG' or 'JPEG')
        quality: JPEG quality (1-100)
        selected_indices: Optional list of indices to include
        filename_template: Template for filenames
        
    Returns:
        Tuple of (number of saved images, output directory path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if selected_indices is not None:
        items = [(idx, images[idx]) for idx in selected_indices if 0 <= idx < len(images)]
    else:
        items = list(enumerate(images))
    
    ext = "jpg" if format.upper() == "JPEG" else "png"
    saved_count = 0
    
    for new_idx, (orig_idx, img) in enumerate(items, start=1):
        # Convert RGBA to RGB for JPEG
        if format.upper() == "JPEG" and img.mode == "RGBA":
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        elif format.upper() == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        
        filename = filename_template.format(
            prefix=prefix,
            index=new_idx,
            total=len(items)
        ) + f".{ext}"
        
        filepath = os.path.join(output_dir, filename)
        
        if format.upper() == "JPEG":
            img.save(filepath, format='JPEG', quality=quality)
        else:
            img.save(filepath, format='PNG')
        saved_count += 1
    
    return saved_count, output_dir


def create_contact_sheet(images: List[Image.Image], cols: int = 4) -> bytes:
    """
    Create a single contact sheet image from a list of images.
    
    Args:
        images: List of PIL Images
        cols: Number of columns in the grid
        
    Returns:
        PNG image bytes
    """
    if not images:
        return b""
    
    # Assume all images are same size
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    
    # Create white background sheet
    sheet = Image.new('RGBA', (w * cols, h * rows), (255, 255, 255, 0))
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        sheet.paste(img, (c * w, r * h), img)
        
    buf = io.BytesIO()
    sheet.save(buf, format='PNG')
    return buf.getvalue()


def process_images(images: List[Image.Image], threshold: int = 127, make_transparent: bool = False) -> List[Image.Image]:
    """
    Apply post-processing to images (thresholding and transparency).
    
    Args:
        images: List of PIL Images
        threshold: Binarization threshold (0-255)
        make_transparent: Whether to make white background transparent
        
    Returns:
        List of processed PIL Images
    """
    processed = []
    for img in images:
        # Convert to grayscale if not already
        img_gray = img.convert('L')
        
        # Binarize
        # Pixels < threshold become 0 (black), others 255 (white)
        # For signatures (black ink on white), we want dark pixels to stay dark
        fn = lambda x: 0 if x < threshold else 255
        img_binary = img_gray.point(fn, mode='1')
        
        if make_transparent:
            # Convert to RGBA
            img_rgba = img_binary.convert("RGBA")
            data = img_rgba.getdata()
            
            new_data = []
            for item in data:
                # If white (255, 255, 255), make transparent
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append((0, 0, 0, 255))  # Make ink pure black
            
            img_rgba.putdata(new_data)
            processed.append(img_rgba)
        else:
            processed.append(img_binary)
            
    return processed


# -----------------------------------------------------------------------------
# Page Renderers
# -----------------------------------------------------------------------------

def render_generation_page():
    st.markdown(
        """
        <div class="sg-hero">
          <div class="sg-hero-title">🚀 Generate Signatures</div>
          <div class="sg-hero-sub">Batch generation, quality filtering, post-processing, and latent morphing — all from your trained checkpoints.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar Controls for Generation
    st.sidebar.header("⚙️ Generation Settings")

    with st.sidebar.expander("🔐 Security Notes", expanded=False):
        st.markdown(
            """PyTorch checkpoints are **pickle-based** and can execute code when loaded.

For safety, this app only loads checkpoints from project folders by default:
- `checkpoints/`
- `runs/`

Only enable **Unsafe mode** for files you explicitly trust.
"""
        )
        unsafe_mode = st.checkbox(
            "Unsafe mode (allow checkpoints outside project folders)",
            value=False,
            key="unsafe_checkpoint_mode",
            help="Allows loading arbitrary local checkpoint paths. Only enable for trusted files.",
        )
    checkpoints = find_checkpoints(DEFAULT_CHECKPOINT_DIR)
    
    selected_checkpoint_path = None
    selected_model_name = "signature"

    # Also search in runs directory for checkpoints
    run_checkpoints = []
    if DEFAULT_RUNS_DIR.exists():
        run_checkpoints = list(DEFAULT_RUNS_DIR.rglob("*.pt"))
    all_checkpoints = checkpoints + run_checkpoints
    all_checkpoints = list(set(all_checkpoints))  # dedupe
    all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if all_checkpoints:
        checkpoint_options = {}
        for cp in all_checkpoints:
            # Create readable names
            try:
                if DEFAULT_RUNS_DIR in cp.parents:
                    # From runs folder: show run name + checkpoint name
                    run_name = cp.parent.parent.name if cp.parent.name == "checkpoints" else cp.parent.name
                    name = f"🏃 {run_name}/{cp.stem}"
                elif DEFAULT_CHECKPOINT_DIR in cp.parents or cp.parent == DEFAULT_CHECKPOINT_DIR:
                    rel_path = cp.relative_to(DEFAULT_CHECKPOINT_DIR)
                    if rel_path.parent != Path('.'):
                        name = f"📦 {rel_path.parent}/{cp.stem}"
                    else:
                        name = f"📦 {cp.stem}"
                else:
                    name = f"📄 {cp.stem}"
            except Exception:
                name = cp.stem
            checkpoint_options[name] = str(cp)
            
        selected_checkpoint_name = st.sidebar.selectbox(
            "📁 Select Model Checkpoint",
            options=list(checkpoint_options.keys()),
            help="📦 = checkpoints folder, 🏃 = training runs"
        )
        selected_checkpoint_path = checkpoint_options[selected_checkpoint_name]
        
        # Track checkpoint changes and clear morph state when changed
        if 'last_selected_checkpoint' not in st.session_state:
            st.session_state.last_selected_checkpoint = None
        
        if st.session_state.last_selected_checkpoint != selected_checkpoint_path:
            # Checkpoint changed - clear morph state
            st.session_state.morph_z1 = None
            st.session_state.morph_z2 = None
            st.session_state.morph_img1 = None
            st.session_state.morph_img2 = None
            st.session_state.last_selected_checkpoint = selected_checkpoint_path
        selected_model_name = selected_checkpoint_name.replace(" (", "_").replace(")", "").replace(" ", "_").replace("/", "_").replace("🏃 ", "").replace("📦 ", "").replace("📄 ", "")
    else:
        st.sidebar.error("⚠️ No checkpoints found!")
        st.sidebar.markdown("""
        **To get started:**
        1. Go to **Train** page to train a model
        2. Or place `.pt` files in `checkpoints/` folder
        """)
        custom_path = st.sidebar.text_input(
            "📁 Or enter custom path:",
            placeholder="C:/path/to/checkpoint.pt",
            help="By default, only project checkpoints/runs are allowed. Enable Unsafe mode to load others.",
        )
        selected_checkpoint_path = None
        if custom_path:
            cp, err = _validate_checkpoint_path(custom_path)
            if err:
                st.sidebar.error(err)
            else:
                if _is_checkpoint_path_allowed(cp) or unsafe_mode:
                    selected_checkpoint_path = str(cp)
                    selected_model_name = cp.stem
                else:
                    st.sidebar.error(
                        "For safety, custom checkpoints must be under the project `checkpoints/` or `runs/` folders. "
                        "Enable Unsafe mode to load a checkpoint from elsewhere."
                    )
            
    # Determine trust for any downstream torch.load usage
    trusted_for_loading = False
    if selected_checkpoint_path:
        try:
            trusted_for_loading = _is_checkpoint_path_allowed(Path(selected_checkpoint_path)) or unsafe_mode
        except Exception:
            trusted_for_loading = False

    # Metadata (guarded)
    if selected_checkpoint_path and os.path.exists(selected_checkpoint_path):
        if trusted_for_loading:
            metadata = get_model_metadata(selected_checkpoint_path, trusted_for_loading=trusted_for_loading)
        else:
            metadata = {}
        if metadata:
            with st.sidebar.expander("ℹ️ Model Card", expanded=True):
                st.markdown(f"**Epoch:** {metadata.get('epoch', 'N/A')}")
                loss = metadata.get('loss')
                if isinstance(loss, float):
                    st.markdown(f"**Best Loss:** {loss:.4f}")
                config = metadata.get('config', {})
                if config:
                    st.caption("Configuration:")
                    st.code(f"Latent Dim: {config.get('latent_dim')}\nImage Size: {config.get('image_size')}", language="yaml")

    st.sidebar.divider()
    
    # Quantity
    st.sidebar.subheader("📊 Quantity")
    n_signatures = st.sidebar.number_input("Number of Signatures", 1, 1000, 16, 1)
    
    # Advanced
    with st.sidebar.expander("🛠️ Advanced Options", expanded=False):
        noise_scale = st.slider(
            "🌊 Diversity (Noise Scale)", 0.1, 2.0, 1.0, 0.1,
            help="Controls variation in generated signatures. <1.0 = more similar/conservative, >1.0 = more diverse/experimental"
        )
        use_seed = st.checkbox("🎲 Use Fixed Seed (Reproducibility)", False, help="Enable to generate the same signatures every time")
        seed = st.number_input("Seed Value", 0, 2147483647, 42, help="Same seed = same output") if use_seed else None
            
    # Quality
    st.sidebar.subheader("💎 Quality Control")
    
    # Check discriminator availability BEFORE loading (using cached check)
    has_discriminator = False
    if selected_checkpoint_path and os.path.exists(selected_checkpoint_path) and trusted_for_loading:
        has_discriminator = _cached_checkpoint_has_discriminator(
            selected_checkpoint_path, trusted_for_loading=trusted_for_loading
        )
    
    # Show checkbox with warning indicator if discriminator unavailable
    quality_col1, quality_col2 = st.sidebar.columns([3, 1])
    with quality_col1:
        use_quality_filter = st.checkbox("Filter by Realism", False, help="Uses discriminator to score and keep only the best signatures")
    with quality_col2:
        if not has_discriminator and selected_checkpoint_path:
            st.markdown("⚠️", help="This checkpoint doesn't include discriminator weights. Quality filtering unavailable.")
    
    # Show expanded warning if user tries to enable filter without discriminator
    if use_quality_filter and not has_discriminator:
        st.sidebar.warning("⚠️ Quality filter unavailable: Use a `checkpoint_epoch_*.pt` or `checkpoint_best.pt` file.")
        use_quality_filter = False
    
    quality_ratio = st.sidebar.slider("Oversampling Ratio", 1.5, 5.0, 2.0, 0.5, help="Generate N× more, keep best ones") if use_quality_filter else 1.0

    # Lightweight guardrail warning
    total_samples_est = int(n_signatures * quality_ratio) if use_quality_filter else int(n_signatures)
    if (not torch.cuda.is_available()) and total_samples_est >= 500:
        st.sidebar.warning(
            "Large CPU generation requested. Consider reducing count or disabling quality oversampling."
        )
        
    # Post-processing
    st.sidebar.subheader("🎨 Post-Processing")
    apply_threshold = st.sidebar.checkbox("Clean Up (Binarize)", False, help="Convert to pure black & white")
    threshold_value = 127  # default
    if apply_threshold:
        threshold_value = st.sidebar.slider("Threshold Level", 50, 200, 127, 5, help="Lower = more ink, Higher = less ink")
        make_transparent = st.sidebar.checkbox("Transparent Background", False, help="Make white areas transparent (PNG)")
    else:
        make_transparent = False
    
    st.sidebar.divider()
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"🖥️ Running on: **{device_name}**")

    # Compact overview row (main area)
    ov1, ov2, ov3, ov4 = st.columns([2.2, 1, 1, 1])
    with ov1:
        st.markdown("**Selected checkpoint**")
        st.caption(selected_model_name if selected_checkpoint_path else "—")
    with ov2:
        st.markdown("**Device**")
        st.caption(device_name)
    with ov3:
        st.markdown("**Requested**")
        st.caption(f"{n_signatures} signatures")
    with ov4:
        st.markdown("**Quality filter**")
        st.caption("On" if use_quality_filter else "Off")

    # Main Logic
    if "generated_image_paths" not in st.session_state:
        st.session_state.generated_image_paths = []

    # Generation progress state (cooperative across reruns)
    if "gen_state" not in st.session_state:
        st.session_state.gen_state = "idle"  # idle|running|complete|cancelled
    if "gen_cancel_requested" not in st.session_state:
        st.session_state.gen_cancel_requested = False
    if "gen_output_dir" not in st.session_state:
        st.session_state.gen_output_dir = None
    if "gen_total_samples" not in st.session_state:
        st.session_state.gen_total_samples = 0
    if "gen_target_n" not in st.session_state:
        st.session_state.gen_target_n = 0
    if "gen_next_index" not in st.session_state:
        st.session_state.gen_next_index = 0
    if "gen_batch_size" not in st.session_state:
        st.session_state.gen_batch_size = 32
    if "gen_records" not in st.session_state:
        # For quality filtering: list of {path: str, score: float}
        st.session_state.gen_records = []
    if "gen_use_quality_filter" not in st.session_state:
        st.session_state.gen_use_quality_filter = False
    if "gen_apply_threshold" not in st.session_state:
        st.session_state.gen_apply_threshold = False
    if "gen_threshold_value" not in st.session_state:
        st.session_state.gen_threshold_value = 127
    if "gen_make_transparent" not in st.session_state:
        st.session_state.gen_make_transparent = False
    if "gen_noise_scale" not in st.session_state:
        st.session_state.gen_noise_scale = 1.0
    if "gen_seed" not in st.session_state:
        st.session_state.gen_seed = None
    
    model_loaded = False
    generator = None
    discriminator = None
    latent_dim = DEFAULT_LATENT_DIM
    device = torch.device('cpu')
    
    if selected_checkpoint_path and os.path.exists(selected_checkpoint_path):
        if not trusted_for_loading:
            st.error(
                "❌ Refusing to load checkpoint outside project folders. Enable Unsafe mode in the sidebar to proceed."
            )
        else:
            try:
                with st.spinner("Loading models..."):
                    generator, latent_dim, device = load_generator(selected_checkpoint_path)
                    if use_quality_filter:
                        discriminator = load_discriminator(
                            selected_checkpoint_path,
                            device,
                            trusted_for_loading=trusted_for_loading,
                        )
                model_loaded = True
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")

    # Readiness banner
    if model_loaded:
        st.markdown(
            f"""<div class="sg-banner ok">
            <div class="sg-banner-title">Ready</div>
            <div class="sg-banner-sub">Latent dim: <b>{latent_dim}</b> • Checkpoint: <b>{selected_model_name}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="sg-banner warn">
            <div class="sg-banner-title">Model not loaded</div>
            <div class="sg-banner-sub">Select a valid checkpoint in the sidebar to enable generation.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    
    tab_batch, tab_morph = st.tabs(["🚀 Batch Generation", "🧬 Morphing"])
    
    with tab_batch:
        st.markdown(
            """<div class="sg-section-title">Generation Console</div>
            <div class="sg-section-sub">Preview first, then generate your batch. Use cancel to stop early and keep partial results.</div>""",
            unsafe_allow_html=True,
        )
        # Preview + Generate buttons side by side
        btn_col1, btn_col2 = st.columns([1, 2])
        with btn_col1:
            preview_button = st.button("👁️ Preview (1 sample)", disabled=not model_loaded, use_container_width=True, help="Generate a single sample to check settings")
        with btn_col2:
            generate_button = st.button(f"🚀 Generate {n_signatures} Signatures", type="primary", disabled=not model_loaded, use_container_width=True)
        
        # Initialize preview state
        if 'preview_image' not in st.session_state:
            st.session_state.preview_image = None
        if 'preview_processed' not in st.session_state:
            st.session_state.preview_processed = None
        
        # Handle preview generation
        if preview_button and model_loaded and generator is not None:
            with st.spinner("Generating preview..."):
                preview_images = generate_signatures_batch(
                    generator=generator, n_samples=1, latent_dim=latent_dim,
                    device=device, seed=seed, batch_size=1, noise_scale=noise_scale
                )
                if preview_images:
                    st.session_state.preview_image = preview_images[0]
                    if apply_threshold:
                        processed = process_images([preview_images[0]], threshold=threshold_value, make_transparent=make_transparent)
                        st.session_state.preview_processed = processed[0]
                    else:
                        st.session_state.preview_processed = None
        
        # Display persistent preview
        if st.session_state.preview_image is not None:
            preview_header_col1, preview_header_col2 = st.columns([4, 1])
            with preview_header_col1:
                st.markdown("##### 👁️ Preview")
            with preview_header_col2:
                if st.button("🗑️ Clear", key="clear_preview", use_container_width=True):
                    st.session_state.preview_image = None
                    st.session_state.preview_processed = None
                    st.rerun()
            
            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.image(st.session_state.preview_image, caption="Raw Output", use_container_width=True)
            with col_proc:
                if apply_threshold:
                    # Re-process if threshold settings changed
                    processed = process_images([st.session_state.preview_image], threshold=threshold_value, make_transparent=make_transparent)
                    st.image(processed[0], caption=f"After Processing (threshold={threshold_value})", use_container_width=True)
                else:
                    st.caption("Enable 'Clean Up' in sidebar to see post-processing")
        
        # Start generation (initialize state and rerun)
        if generate_button and model_loaded and generator is not None:
            run_dir = DEFAULT_SAMPLE_DIR / _new_run_id("gen")
            run_dir.mkdir(parents=True, exist_ok=True)

            st.session_state.generated_image_paths = []
            st.session_state.selected_images = set()

            st.session_state.gen_state = "running"
            st.session_state.gen_cancel_requested = False
            st.session_state.gen_output_dir = str(run_dir)
            st.session_state.gen_records = []
            st.session_state.gen_next_index = 0
            st.session_state.gen_batch_size = 32
            st.session_state.gen_use_quality_filter = bool(use_quality_filter)
            st.session_state.gen_apply_threshold = bool(apply_threshold)
            st.session_state.gen_threshold_value = int(threshold_value)
            st.session_state.gen_make_transparent = bool(make_transparent)
            st.session_state.gen_noise_scale = float(noise_scale)
            st.session_state.gen_seed = int(seed) if seed is not None else None
            st.session_state.gen_target_n = int(n_signatures)
            st.session_state.gen_total_samples = int(n_signatures * quality_ratio) if use_quality_filter else int(n_signatures)

            st.rerun()

        # Cooperative generation step (one batch per rerun)
        if st.session_state.gen_state == "running":
            total_samples = int(st.session_state.gen_total_samples)
            next_index = int(st.session_state.gen_next_index)
            batch_size = int(st.session_state.gen_batch_size)

            progress = 0.0 if total_samples <= 0 else min(1.0, next_index / total_samples)
            progress_bar = st.progress(progress, text=f"Generating signatures... {next_index}/{total_samples}")

            status_col1, status_col2 = st.columns([3, 1])
            with status_col1:
                status_text = st.empty()
            with status_col2:
                if st.button("🛑 Cancel", key="cancel_generation_running", type="secondary", use_container_width=True):
                    st.session_state.gen_cancel_requested = True
                    st.rerun()

            if st.session_state.gen_cancel_requested:
                status_text.warning(
                    f"⚠️ Cancellation requested. Finalizing {next_index} generated samples..."
                )

            def _finalize_generation(state: str) -> None:
                use_q = bool(st.session_state.gen_use_quality_filter)
                if use_q:
                    records = list(st.session_state.gen_records)
                    records.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
                    target_n = int(st.session_state.gen_target_n)
                    selected = records[: min(target_n, len(records))]
                    selected_paths = [r["path"] for r in selected if "path" in r]
                    st.session_state.generated_image_paths = selected_paths

                    # Delete non-selected oversamples to save disk.
                    selected_set = set(selected_paths)
                    for r in records:
                        p = r.get("path")
                        if p and p not in selected_set:
                            try:
                                Path(p).unlink(missing_ok=True)
                            except Exception:
                                pass
                # Non-quality filter already appended paths to generated_image_paths
                st.session_state.gen_state = state

            try:
                # Stop within one batch when cancelled.
                if st.session_state.gen_cancel_requested:
                    _finalize_generation("cancelled")
                elif next_index >= total_samples:
                    _finalize_generation("complete")
                else:
                    current_batch_size = min(batch_size, total_samples - next_index)
                    batch_number = next_index // batch_size
                    base_seed = st.session_state.gen_seed
                    batch_seed = (base_seed + batch_number) if base_seed is not None else None
                    status_text.text(
                        f"🔄 Generating batch {batch_number + 1} ({current_batch_size} samples)..."
                    )

                    batch_images = generate_signatures_batch(
                        generator=generator,
                        n_samples=current_batch_size,
                        latent_dim=latent_dim,
                        device=device,
                        seed=batch_seed,
                        batch_size=current_batch_size,
                        noise_scale=float(st.session_state.gen_noise_scale),
                    )

                    if st.session_state.gen_apply_threshold:
                        batch_images = process_images(
                            batch_images,
                            threshold=int(st.session_state.gen_threshold_value),
                            make_transparent=bool(st.session_state.gen_make_transparent),
                        )

                    out_dir = Path(st.session_state.gen_output_dir)

                    # Optional scoring (stores only score + file path)
                    scores: Optional[List[float]] = None
                    if st.session_state.gen_use_quality_filter and discriminator is not None:
                        batch_tensors = []
                        for img in batch_images:
                            img_gray = img.convert("L")
                            t = torch.from_numpy(np.array(img_gray)).float().unsqueeze(0) / 127.5 - 1.0
                            batch_tensors.append(t)
                        batch_input = torch.stack(batch_tensors).to(device)
                        with torch.no_grad():
                            scores = discriminator(batch_input).cpu().numpy().flatten().tolist()

                    for i, img in enumerate(batch_images):
                        idx = next_index + i
                        file_path = out_dir / f"sample_{idx + 1:05d}.png"
                        img.save(str(file_path), format="PNG")
                        if scores is not None:
                            st.session_state.gen_records.append(
                                {"path": str(file_path), "score": float(scores[i])}
                            )
                        else:
                            st.session_state.generated_image_paths.append(str(file_path))

                    st.session_state.gen_next_index = next_index + current_batch_size
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                st.session_state.gen_state = "idle"

        if st.session_state.gen_state == "complete" and st.session_state.generated_image_paths:
            st.success(f"✅ Generated {len(st.session_state.generated_image_paths)} signatures successfully!")
        elif st.session_state.gen_state == "cancelled" and st.session_state.generated_image_paths:
            st.warning(f"⚠️ Generation cancelled. Kept {len(st.session_state.generated_image_paths)} signatures.")
        elif st.session_state.gen_state == "cancelled":
            st.warning("⚠️ Generation cancelled.")
        
        if st.session_state.generated_image_paths:
            st.divider()
            
            image_paths = st.session_state.generated_image_paths
            num_images = len(image_paths)
            
            # Initialize selection state
            if 'selected_images' not in st.session_state:
                st.session_state.selected_images = set()
            
            # Header with selection controls
            res_col1, res_col2, res_col3 = st.columns([2.2, 1, 1])
            with res_col1:
                st.markdown("<div class=\"sg-section-title\">📸 Generated Signatures Gallery</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class=\"sg-section-sub\">{num_images} images ready • Use Export Options to download, save, or build a contact sheet.</div>",
                    unsafe_allow_html=True,
                )
            with res_col2:
                selection_mode = st.checkbox("☑️ Selection Mode", value=False, help="Enable to select specific images for download")
            with res_col3:
                if selection_mode:
                    sel_subcol1, sel_subcol2 = st.columns(2)
                    with sel_subcol1:
                        if st.button("Select All", use_container_width=True, key="select_all"):
                            st.session_state.selected_images = set(range(num_images))
                            st.rerun()
                    with sel_subcol2:
                        if st.button("Clear All", use_container_width=True, key="clear_selection"):
                            st.session_state.selected_images = set()
                            st.rerun()
            
            # Export Options expander
            with st.expander("📤 Export Options", expanded=True):
                st.markdown(
                    """<div class="sg-section-sub" style="margin-top:-0.25rem;">
                    Choose format + naming, then download everything (or only your selection).
                    </div>""",
                    unsafe_allow_html=True,
                )

                exp_col1, exp_col2, exp_col3 = st.columns([1.2, 1.6, 1.2])
                
                with exp_col1:
                    export_format = st.selectbox(
                        "Format",
                        options=["PNG", "JPEG"],
                        index=0,
                        help="PNG for transparency support, JPEG for smaller file size"
                    )
                    if export_format == "JPEG":
                        jpeg_quality = st.slider("JPEG Quality", 50, 100, 90, 5)
                    else:
                        jpeg_quality = 95
                
                with exp_col2:
                    filename_prefix = st.text_input(
                        "Filename Prefix",
                        value=selected_model_name,
                        help="Prefix for generated filenames"
                    )
                    filename_template = st.text_input(
                        "Filename Template",
                        value="{prefix}_{index:03d}",
                        help="Template: {prefix}, {index}, {total}"
                    )
                
                with exp_col3:
                    st.markdown("**Quick Download**")
                    # Determine what to download
                    selected_list = sorted(st.session_state.selected_images) if selection_mode and st.session_state.selected_images else None
                    download_count = len(selected_list) if selected_list else num_images
                    
                    zip_data = create_zip_archive_from_paths(
                        image_paths,
                        prefix=filename_prefix,
                        format=export_format,
                        quality=jpeg_quality,
                        selected_indices=selected_list,
                        filename_template=filename_template
                    )
                    ext = "jpg" if export_format == "JPEG" else "png"
                    zip_filename = f"Signatures_{filename_prefix}.zip"
                    
                    btn_label = f"📥 Download {download_count} images" if selection_mode and selected_list else "📥 Download All"
                    st.download_button(
                        btn_label,
                        zip_data,
                        zip_filename,
                        "application/zip",
                        use_container_width=True,
                        type="primary"
                    )
                
                # Save to Folder section
                st.markdown("---")
                save_col1, save_col2 = st.columns([3, 1])
                with save_col1:
                    if "default_save_folder" not in st.session_state:
                        st.session_state.default_save_folder = str(
                            DEFAULT_SAMPLE_DIR / f"generated_{time.strftime('%Y%m%d_%H%M%S')}"
                        )
                    save_folder = st.text_input(
                        "💾 Save to Folder",
                        value=st.session_state.default_save_folder,
                        help="Enter a folder path to save images directly to disk"
                    )
                with save_col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    if st.button("💾 Save to Disk", use_container_width=True, type="secondary"):
                        try:
                            saved_count, saved_path = save_images_to_folder_from_paths(
                                image_paths,
                                save_folder,
                                prefix=filename_prefix,
                                format=export_format,
                                quality=jpeg_quality,
                                selected_indices=selected_list,
                                filename_template=filename_template
                            )
                            st.success(f"✅ Saved {saved_count} images to: `{saved_path}`")
                        except Exception as e:
                            st.error(f"❌ Failed to save: {str(e)}")
                
                # Contact sheet (no nested expander to avoid Streamlit nesting error)
                st.markdown("**📄 Contact Sheet**")
                sheet_cols = st.slider("Columns", 2, 10, 4, key="sheet_cols")
                sheet_src_paths = [
                    image_paths[i] for i in sorted(st.session_state.selected_images)
                ] if selection_mode and st.session_state.selected_images else image_paths
                sheet_images = []
                for p in sheet_src_paths:
                    try:
                        with Image.open(p) as opened:
                            sheet_images.append(opened.copy())
                    except Exception:
                        pass
                sheet_data = create_contact_sheet(sheet_images, cols=sheet_cols)
                if sheet_data:
                    st.download_button("📄 Download Contact Sheet", sheet_data, f"Contact_Sheet_{filename_prefix}.png", "image/png", use_container_width=True)
            
            # Selection info bar
            if selection_mode and st.session_state.selected_images:
                st.info(f"📌 Selected: {len(st.session_state.selected_images)} of {num_images} images")
            
            # Pagination controls
            IMAGES_PER_PAGE = 24
            total_pages = (num_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
            
            # Initialize page in session state
            if 'gallery_current_page' not in st.session_state:
                st.session_state.gallery_current_page = 1
            
            # Ensure page is within bounds
            if st.session_state.gallery_current_page > total_pages:
                st.session_state.gallery_current_page = 1
            
            current_page = st.session_state.gallery_current_page
            
            if total_pages > 1:
                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                
                with page_col1:
                    prev_disabled = current_page <= 1
                    if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True, key="prev_page"):
                        st.session_state.gallery_current_page = max(1, current_page - 1)
                        st.rerun()
                
                with page_col2:
                    start_img = (current_page - 1) * IMAGES_PER_PAGE + 1
                    end_img = min(current_page * IMAGES_PER_PAGE, num_images)
                    st.markdown(
                        f"<div style='text-align: center; padding: 8px;'>"
                        f"<strong>Page {current_page} of {total_pages}</strong><br/>"
                        f"<small>Images {start_img}-{end_img} of {num_images}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                with page_col3:
                    next_disabled = current_page >= total_pages
                    if st.button("Next ➡️", disabled=next_disabled, use_container_width=True, key="next_page"):
                        st.session_state.gallery_current_page = min(total_pages, current_page + 1)
                        st.rerun()
            
            # Get images for current page
            start_idx = (current_page - 1) * IMAGES_PER_PAGE
            end_idx = min(start_idx + IMAGES_PER_PAGE, num_images)
            display_paths = image_paths[start_idx:end_idx]
            
            # Display grid with selection or download
            num_cols = 6 if len(display_paths) > 12 else (4 if len(display_paths) > 4 else max(len(display_paths), 1))
            cols = st.columns(num_cols)
            for idx, img_path in enumerate(display_paths):
                actual_idx = start_idx + idx
                with cols[idx % num_cols]:
                    # Show selection indicator
                    is_selected = actual_idx in st.session_state.selected_images
                    if selection_mode and is_selected:
                        st.markdown("<div class=\"sg-selected-pill\">Selected ✓</div>", unsafe_allow_html=True)
                    
                    st.image(img_path, caption=f"#{actual_idx + 1}" + (" ✓" if is_selected else ""), use_container_width=True)
                    
                    if selection_mode:
                        # Toggle selection checkbox
                        if st.checkbox(
                            "Select", 
                            value=is_selected, 
                            key=f"sel_{actual_idx}",
                            label_visibility="collapsed"
                        ):
                            st.session_state.selected_images.add(actual_idx)
                        else:
                            st.session_state.selected_images.discard(actual_idx)
                    else:
                        # Individual download button
                        with Image.open(img_path) as opened:
                            img = opened.copy()
                        img_buf = io.BytesIO()
                        if export_format == "JPEG":
                            rgb_img = img.convert("RGB") if img.mode != "RGB" else img
                            rgb_img.save(img_buf, format="JPEG", quality=jpeg_quality)
                        else:
                            img.save(img_buf, format="PNG")
                        st.download_button(
                            "📥 Save", img_buf.getvalue(),
                            f"{filename_prefix}_{actual_idx + 1:03d}.{ext}",
                            f"image/{export_format.lower()}",
                            key=f"dl_{actual_idx}",
                            use_container_width=True,
                            type="secondary"
                        )

    with tab_morph:
        st.subheader("🧬 Latent Space Morphing")
        st.caption("Explore smooth transitions between two signature styles")
        
        if 'morph_z1' not in st.session_state:
            st.session_state.morph_z1 = None
            st.session_state.morph_z2 = None
            st.session_state.morph_img1 = None
            st.session_state.morph_img2 = None
            
        if st.button("🎲 Generate New Endpoint Signatures", disabled=not model_loaded, type="primary"):
            if model_loaded and generator is not None:
                z1 = torch.randn(1, latent_dim, 1, 1, device=device)
                z2 = torch.randn(1, latent_dim, 1, 1, device=device)
                st.session_state.morph_z1 = z1
                st.session_state.morph_z2 = z2
                # Pre-generate endpoint images
                with torch.no_grad():
                    t1 = generator(z1)
                    t1 = ((t1 + 1) / 2.0).clamp(0, 1).cpu().squeeze().numpy()
                    st.session_state.morph_img1 = Image.fromarray((t1 * 255).astype(np.uint8))
                    
                    t2 = generator(z2)
                    t2 = ((t2 + 1) / 2.0).clamp(0, 1).cpu().squeeze().numpy()
                    st.session_state.morph_img2 = Image.fromarray((t2 * 255).astype(np.uint8))
        
        if st.session_state.morph_z1 is not None and model_loaded and generator is not None:
            # Show endpoints
            st.markdown("#### 🔀 Endpoints")
            ep_col1, ep_col2 = st.columns(2)
            with ep_col1:
                if st.session_state.morph_img1:
                    disp_img1 = process_images([st.session_state.morph_img1], threshold=threshold_value, make_transparent=make_transparent)[0] if apply_threshold else st.session_state.morph_img1
                    st.image(disp_img1, caption="Start (α=0.0)", use_container_width=True)
            with ep_col2:
                if st.session_state.morph_img2:
                    disp_img2 = process_images([st.session_state.morph_img2], threshold=threshold_value, make_transparent=make_transparent)[0] if apply_threshold else st.session_state.morph_img2
                    st.image(disp_img2, caption="End (α=1.0)", use_container_width=True)
            
            st.markdown("#### 🎚️ Interpolation")
            alpha = st.slider("Blend Factor (α)", 0.0, 1.0, 0.5, 0.01, help="0.0 = Start signature, 1.0 = End signature")
            
            z = (1 - alpha) * st.session_state.morph_z1 + alpha * st.session_state.morph_z2
            with torch.no_grad():
                img_tensor = generator(z)
                img_tensor = (img_tensor + 1) / 2.0
                img_tensor = img_tensor.clamp(0, 1)
                img_np = img_tensor.cpu().squeeze().numpy()
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                if apply_threshold:
                    img_pil = process_images([img_pil], threshold=threshold_value, make_transparent=make_transparent)[0]
            
            m_col1, m_col2, m_col3 = st.columns([1, 2, 1])
            with m_col2:
                st.image(img_pil, caption=f"Interpolated (α={alpha:.2f})", use_container_width=True)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button("📥 Download This Frame", buf.getvalue(), f"morph_alpha_{alpha:.2f}.png", "image/png", use_container_width=True)
            
            # Export animation strip
            with st.expander("🎬 Export Morph Sequence"):
                n_frames = st.slider("Number of frames", 5, 30, 10)
                if st.button("Generate Sequence"):
                    frames = []
                    for i in range(n_frames):
                        a = i / (n_frames - 1)
                        z_interp = (1 - a) * st.session_state.morph_z1 + a * st.session_state.morph_z2
                        with torch.no_grad():
                            t = generator(z_interp)
                            t = ((t + 1) / 2.0).clamp(0, 1).cpu().squeeze().numpy()
                            frame = Image.fromarray((t * 255).astype(np.uint8))
                            if apply_threshold:
                                frame = process_images([frame], threshold=threshold_value, make_transparent=make_transparent)[0]
                            frames.append(frame)
                    
                    # Create horizontal strip
                    strip_width = frames[0].width * n_frames
                    strip = Image.new('RGBA' if make_transparent else 'L', (strip_width, frames[0].height), (255, 255, 255, 0) if make_transparent else 255)
                    for i, frame in enumerate(frames):
                        if frame.mode != strip.mode:
                            frame = frame.convert(strip.mode)
                        strip.paste(frame, (i * frames[0].width, 0))
                    
                    st.image(strip, caption=f"Morph sequence ({n_frames} frames)", use_container_width=True)
                    strip_buf = io.BytesIO()
                    strip.save(strip_buf, format="PNG")
                    st.download_button("📥 Download Strip", strip_buf.getvalue(), "morph_sequence.png", "image/png")
        else:
            st.info("👆 Click 'Generate New Endpoint Signatures' to start morphing")

def render_preprocessing_page():
    st.header("🧹 Preprocess Signatures")
    st.markdown("Prepare your raw signature images for training by resizing, binarizing, and normalizing them.")
    
    with st.form("preprocess_form"):
        col1, col2 = st.columns(2)
        with col1:
            input_dir = st.text_input("Input Directory (Raw Images)", str(DEFAULT_DATA_DIR / "signatures/train"))
        with col2:
            output_dir = st.text_input("Output Directory (Processed)", str(DEFAULT_DATA_DIR / "processed"))
            
        size = st.selectbox("Target Image Size", [64, 128], index=0)
        
        st.markdown("### Options")
        c1, c2 = st.columns(2)
        with c1:
            disable_validation = st.checkbox("Disable Validation (Recommended for sparse signatures)", value=True)
        with c2:
            verbose_logging = st.checkbox("Verbose Logging", value=True)
            
        submitted = st.form_submit_button("Start Preprocessing", type="primary")
        
    if submitted:
        result = None
        with st.status("Preprocessing in progress...", expanded=True) as status:
            st.write("🚀 Launching preprocessing script...")
            
            # Ensure paths are absolute
            abs_input = str(Path(input_dir).resolve())
            abs_output = str(Path(output_dir).resolve())
            
            cmd = [
                sys.executable, 
                "src/preprocess_signatures.py", 
                abs_input, 
                abs_output, 
                "--size", str(size)
            ]
            
            if disable_validation:
                cmd.append("--no-validate")
            
            if verbose_logging:
                cmd.append("--verbose")
            
            # Set up environment to ensure imports work
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent)
            
            try:
                # Run with explicit cwd and env
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    cwd=str(Path(__file__).parent.parent), # Run from project root
                    env=env
                )
                
                if result.returncode == 0:
                    status.update(label="Preprocessing Complete!", state="complete", expanded=False)
                else:
                    status.update(label="Preprocessing Failed", state="error", expanded=False)
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Failed to run script: {e}")

        # Display results outside the status container to avoid nesting errors
        if result:
            if result.returncode == 0:
                st.success("✅ Preprocessing finished successfully!")
                with st.expander("View Logs"):
                    st.code(result.stderr)
            else:
                st.error("❌ Preprocessing failed.")
                st.markdown("### Error Output:")
                st.code(result.stderr)
                st.markdown("### Standard Output:")
                st.code(result.stdout)

def render_training_page():
    st.header("🏋️ Train GAN Model")
    
    # Initialize session state for persistent settings
    if "train_config" not in st.session_state:
        st.session_state.train_config = DEFAULT_TRAIN_CONFIG.copy()
    
    # Device info banner
    device_col1, device_col2 = st.columns([3, 1])
    with device_col1:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info = _get_gpu_memory_info()
            if gpu_info:
                st.success(f"🎮 **GPU:** {gpu_name} | Memory: {gpu_info['allocated_gb']:.1f}/{gpu_info['total_gb']:.1f} GB ({gpu_info['percent_used']:.0f}% used)")
            else:
                st.success(f"🎮 **GPU:** {gpu_name}")
        else:
            st.warning("⚠️ **No GPU detected** - Training will be slow on CPU")
    with device_col2:
        if st.button("🔄 Refresh GPU Info", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Training Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            data_dir = st.text_input("Data Directory", str(DEFAULT_DATA_DIR / "processed"),
                                     help="Path to folder containing processed training images")
            epochs = st.number_input("Epochs", 1, 1000, st.session_state.train_config["epochs"],
                                     help="Total number of training epochs. More epochs = longer training but potentially better results")
            batch_size = st.number_input("Batch Size", 16, 256, st.session_state.train_config["batch_size"],
                                         help="Number of images per training batch. Larger = faster but uses more GPU memory")
        with col2:
            lr = st.number_input("Learning Rate", 0.0001, 0.01, st.session_state.train_config["learning_rate"], format="%.5f",
                                 help="Controls how fast the model learns. Too high = unstable, too low = slow convergence")
            latent_dim = st.number_input("Latent Dimension", 64, 512, st.session_state.train_config["latent_dim"],
                                         help="Size of random noise vector. Higher = more variety potential but harder to train")
            image_size = st.selectbox("Image Size", [64, 128], 
                                      index=0 if st.session_state.train_config["image_size"] == 64 else 1,
                                      help="Output image resolution. 128 takes longer to train")
        
        # Persist settings to session state
        st.session_state.train_config.update({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "latent_dim": latent_dim,
            "image_size": image_size,
        })

        st.markdown("### Run")
        # Generate a default run name only once per session to prevent it from changing on every rerun
        if "default_run_name" not in st.session_state:
            st.session_state.default_run_name = _new_run_id(prefix="train")
            
        run_name = st.text_input("Run Name", value=st.session_state.default_run_name)
        resume_training = st.checkbox("Resume from checkpoint", value=False)

        run_dirs = _list_runs(DEFAULT_RUNS_DIR)
        run_dir_options = {"(new run)": None}
        for p in run_dirs[:20]:
            run_dir_options[p.name] = p

        selected_existing_run_name = st.selectbox(
            "Select Existing Run (for monitoring/resume)",
            options=list(run_dir_options.keys()),
            index=0,
            disabled=not resume_training,
        )
        selected_existing_run_dir = run_dir_options.get(selected_existing_run_name)

        resume_checkpoint_path: Optional[str] = None
        if resume_training and selected_existing_run_dir is not None:
            ckpt_dir = selected_existing_run_dir / "checkpoints"
            ckpts = []
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                resume_checkpoint_path = st.selectbox(
                    "Resume Checkpoint",
                    options=[str(p) for p in ckpts],
                )
            else:
                st.warning("No checkpoints found in the selected run.")

        # Quick validation
        img_count = _count_images(Path(data_dir))
        if img_count == 0:
            st.warning("No training images found in Data Directory.")
        else:
            st.caption(f"Found {img_count} images.")

    # Process Management - Initialize session state
    if 'training_process' not in st.session_state:
        st.session_state.training_process = None
    if 'training_run_dir' not in st.session_state:
        st.session_state.training_run_dir = None
    if 'training_stop_file' not in st.session_state:
        st.session_state.training_stop_file = None
    if 'training_stop_requested' not in st.session_state:
        st.session_state.training_stop_requested = False
    if 'training_pid' not in st.session_state:
        st.session_state.training_pid = None
    
    # Check if there's an active training (either in session or from saved state)
    saved_state = _load_training_state()
    
    # Restore from saved state if session was lost but training is still running
    if saved_state and saved_state.get("is_running"):
        st.session_state.training_run_dir = saved_state.get("run_dir")
        st.session_state.training_stop_file = saved_state.get("stop_file")
        st.session_state.training_pid = saved_state.get("pid")
        
    col_btn1, col_btn2 = st.columns([1, 4])
    
    with col_btn1:
        if st.session_state.training_process is None:
            start_disabled = img_count == 0
            if st.button("🚀 Start Training", type="primary", disabled=start_disabled):
                project_root = Path(__file__).parent.parent

                run_dir = DEFAULT_RUNS_DIR / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "logs").mkdir(parents=True, exist_ok=True)

                stop_file = run_dir / "stop.request"
                try:
                    if stop_file.exists():
                        stop_file.unlink()
                except Exception:
                    pass

                cmd = [
                    sys.executable,
                    "src/train_vanilla_gan_signatures.py",
                    "--data_dir",
                    str(Path(data_dir).resolve()),
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    str(batch_size),
                    "--g_lr",
                    str(lr),
                    "--d_lr",
                    str(lr),
                    "--latent_dim",
                    str(latent_dim),
                    "--image_size",
                    str(image_size),
                    "--run_dir",
                    str(run_dir.resolve()),
                    "--stop_file",
                    str(stop_file.resolve()),
                ]

                if resume_training and resume_checkpoint_path:
                    cmd.extend(["--resume_from", resume_checkpoint_path])

                log_path = run_dir / "logs" / "training_output.log"
                st.session_state.training_stop_requested = False

                # Set environment to disable Python buffering for real-time output
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                
                # Open log file in unbuffered mode and keep handle in session state
                log_file = open(log_path, "w", encoding="utf-8", buffering=1)
                
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(project_root),
                    text=True,
                    bufsize=1,  # Line buffered
                    env=env,
                )
                
                # Store file handle to keep it open
                st.session_state.training_log_file = log_file

                st.session_state.training_process = process
                st.session_state.training_run_dir = str(run_dir)
                st.session_state.training_stop_file = str(stop_file)
                st.session_state.training_pid = process.pid
                
                # Save state immediately for persistence
                _save_training_state(str(run_dir), process.pid, str(stop_file))
                st.rerun()
        else:
            if st.button("🛑 Stop Training", type="secondary"):
                # Cooperative stop: create stop file; training loop checks and exits safely.
                stop_file = st.session_state.get('training_stop_file')
                if stop_file:
                    try:
                        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
                        Path(stop_file).write_text("stop requested", encoding="utf-8")
                        st.session_state.training_stop_requested = True
                    except Exception:
                        pass
                st.rerun()

    # Monitoring - Determine if training is active
    # Check both subprocess object and PID-based detection
    proc = st.session_state.training_process
    pid = st.session_state.get('training_pid')
    
    # Training is running if we have a real process that hasn't finished, OR if PID is still active
    is_running = False
    if proc is not None and hasattr(proc, 'poll'):
        try:
            is_running = proc.poll() is None
        except Exception:
            pass
    
    if not is_running and pid:
        is_running = _is_pid_running(pid)
    
    active_run_dir = st.session_state.get('training_run_dir')
    if not active_run_dir and selected_existing_run_dir is not None:
        active_run_dir = str(selected_existing_run_dir)

    run_dir_path = Path(active_run_dir) if active_run_dir else None
    run_log_dir = (run_dir_path / "logs") if run_dir_path else None
    run_sample_dir = (run_dir_path / "samples") if run_dir_path else None

    if is_running:
        # AUTO-REFRESH - Use streamlit_autorefresh component (this works reliably)
        if st_autorefresh is not None:
            try:
                # This will cause the page to rerun every 2 seconds
                count = st_autorefresh(interval=2000, limit=None, key="train_autorefresh")
            except Exception as e:
                st.error(f"Auto-refresh error: {e}")
        else:
            # Fallback message if autorefresh not available
            st.warning("⚠️ Auto-refresh not available. Click Refresh button manually.")
        
        # Progress header with status
        display_pid = pid or (proc.pid if proc and hasattr(proc, 'pid') else 'N/A')
        prog_col1, prog_col2, prog_col3 = st.columns([2, 1, 1])
        with prog_col1:
            st.info(f"🏃 Training in progress (PID: {display_pid})")
        with prog_col2:
            if st.session_state.get('training_stop_requested'):
                st.warning("⏸️ Stop requested...")
        with prog_col3:
            if st.button("🔄 Refresh", key="manual_refresh"):
                st.rerun()
        
        # Read log data
        # Determine log path - add debugging
        if run_log_dir is not None and run_log_dir.exists():
            log_path = run_log_dir / "training_output.log"
        else:
            log_path = DEFAULT_LOG_DIR / "training_output.log"
        
        # Read terminal text - force fresh read each time
        if log_path.exists():
            terminal_text = _tail_text_file(log_path)
            if not terminal_text or terminal_text.strip() == "":
                terminal_text = f"Log file exists but is empty: {log_path}"
        else:
            terminal_text = f"Log file not found: {log_path}\nRun dir: {run_log_dir}"
        
        # Parse progress from logs
        progress_info = _parse_training_progress(terminal_text)
        
        # Load metrics - try JSON first, then CSV, then parse from log
        df = None
        metrics_source = None
        if run_log_dir is not None and run_log_dir.exists():
            # Try JSON first
            json_path = _load_latest_metrics_json(run_log_dir)
            if json_path:
                df = _metrics_dataframe_from_logger_json(json_path)
                if df is not None:
                    metrics_source = json_path.name
            
            # If JSON didn't work, try CSV
            if df is None:
                csv_path = _load_latest_metrics_csv(run_log_dir)
                if csv_path:
                    df = _metrics_dataframe_from_csv(csv_path)
                    if df is not None:
                        metrics_source = csv_path.name
        
        # If no metrics file, try to parse from training log
        if df is None and terminal_text:
            df = _parse_metrics_from_log(terminal_text)
            if df is not None:
                metrics_source = "training_output.log (parsed)"
        
        health_status = _check_loss_health(df)
        
        st.markdown("---")
        
        # Progress bar and key metrics
        st.subheader(f"📈 Training Progress {health_status['status_emoji']}")
        
        # Calculate progress percentage
        total_epochs = progress_info["total_epochs"] or epochs
        current_epoch = progress_info["current_epoch"] or 0
        progress_pct = current_epoch / total_epochs if total_epochs > 0 else 0
        
        # Progress bar
        st.progress(progress_pct, text=f"Epoch {current_epoch}/{total_epochs} ({progress_pct*100:.1f}%)")
        
        # Key metrics row - use parsed progress first, fallback to df
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        with metric_col1:
            st.metric("Epoch", f"{current_epoch}/{total_epochs}")
        with metric_col2:
            g_loss_val = progress_info.get("g_loss") or (df["g_loss"].iloc[-1] if df is not None and len(df) > 0 and "g_loss" in df.columns else None)
            st.metric("G Loss", f"{g_loss_val:.4f}" if g_loss_val else "—")
        with metric_col3:
            d_loss_val = progress_info.get("d_loss") or (df["d_loss"].iloc[-1] if df is not None and len(df) > 0 and "d_loss" in df.columns else None)
            st.metric("D Loss", f"{d_loss_val:.4f}" if d_loss_val else "—")
        with metric_col4:
            d_real_val = progress_info.get("d_real") or (df["d_real"].iloc[-1] if df is not None and len(df) > 0 and "d_real" in df.columns else None)
            st.metric("D(Real)", f"{d_real_val:.2f}" if d_real_val else "—")
        with metric_col5:
            # GPU memory if available
            gpu_info = _get_gpu_memory_info()
            if gpu_info:
                st.metric("GPU Mem", f"{gpu_info['percent_used']:.0f}%")
            else:
                st.metric("GPU", "CPU")
        
        # Health warnings
        if health_status["warnings"]:
            for warn in health_status["warnings"]:
                st.warning(warn)
        
        # Two column layout: Charts + Terminal
        chart_col, log_col = st.columns([1, 1])
        
        with chart_col:
            st.markdown("#### 📊 Loss Curves")
            if df is not None and 'epoch' in df.columns and {'g_loss', 'd_loss'}.issubset(df.columns):
                st.line_chart(df.set_index('epoch')[['g_loss', 'd_loss']], height=300)
                if metrics_source:
                    st.caption(f"📈 Data from: {metrics_source} ({len(df)} epochs)")
            else:
                st.info("⏳ Waiting for metrics data...")
                # Debug info
                if run_log_dir:
                    st.caption(f"Looking in: {run_log_dir}")
                    if run_log_dir.exists():
                        files = list(run_log_dir.glob("*"))
                        st.caption(f"Files found: {[f.name for f in files[:5]]}")
            
            # Sample images
            st.markdown("#### 🖼️ Latest Sample")
            sample_dir = run_sample_dir if run_sample_dir else DEFAULT_SAMPLE_DIR
            sample_files = list(sample_dir.glob("*.png")) if sample_dir.exists() else []
            if sample_files:
                latest_sample = max(sample_files, key=lambda x: x.stat().st_mtime)
                st.image(str(latest_sample), caption=latest_sample.name, use_container_width=True)
            else:
                st.caption("No samples generated yet...")
        
        with log_col:
            st.markdown("#### 📜 Live Terminal")
            
            # Show last 150 lines to keep it manageable but scrollable
            terminal_lines = terminal_text.split('\n')
            display_lines = terminal_lines[-150:] if len(terminal_lines) > 150 else terminal_lines
            display_text = '\n'.join(display_lines) if display_lines else "Waiting for output..."
            
            # Terminal with auto-scroll to bottom using JavaScript
            st.markdown(
                f"""
                <div id="live-terminal" style="
                    background-color: #0e1117;
                    border: 1px solid #262730;
                    border-radius: 4px;
                    padding: 10px;
                    height: {TRAINING_TERMINAL_HEIGHT}px;
                    overflow-y: auto;
                    font-family: 'Consolas', 'Monaco', 'Source Code Pro', monospace;
                    font-size: 12px;
                    line-height: 1.4;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    color: #d4d4d4;
                ">
{html.escape(display_text)}
                </div>
                <script>
                    // Auto-scroll terminal to bottom
                    var terminal = document.getElementById('live-terminal');
                    if (terminal) {{
                        terminal.scrollTop = terminal.scrollHeight;
                    }}
                </script>
                """,
                unsafe_allow_html=True
            )
            
            # Show line count and path info
            st.caption(f"📄 {log_path.name} | Lines: {len(terminal_lines)}")
            
            # Full logs in expander for copying
            with st.expander("📋 Copy Full Logs"):
                st.code(terminal_text[-15000:], language=None)

        # Check if training just finished (process no longer running)
        if not is_running and st.session_state.training_run_dir:
            # Training just finished - close log file and clear state
            log_file_handle = st.session_state.get('training_log_file')
            if log_file_handle:
                try:
                    log_file_handle.close()
                except Exception:
                    pass
                st.session_state.training_log_file = None
            
            st.session_state.training_process = None
            st.session_state.training_pid = None
            old_run_dir = st.session_state.training_run_dir
            st.session_state.training_run_dir = None
            _clear_training_state()
            st.balloons()
            st.success(f"✅ Training completed! Results saved to: {old_run_dir}")
            st.rerun()

    else:
        # Static view when not training - show run history browser
        st.markdown("---")
        
        # Run history selector
        st.subheader("📂 Browse Training Runs")
        all_runs = _list_runs(DEFAULT_RUNS_DIR)
        
        if all_runs:
            run_options = {p.name: p for p in all_runs[:30]}
            selected_run_name = st.selectbox(
                "Select a run to view",
                options=list(run_options.keys()),
                help="Browse previous training runs to view their metrics and samples"
            )
            browse_run_dir = run_options.get(selected_run_name)
            browse_log_dir = browse_run_dir / "logs" if browse_run_dir else None
            browse_sample_dir = browse_run_dir / "samples" if browse_run_dir else None
        else:
            st.info("No training runs found. Start a new training run above.")
            browse_log_dir = DEFAULT_LOG_DIR
            browse_sample_dir = DEFAULT_SAMPLE_DIR
        
        # Show metrics for selected run
        st.subheader("📈 Training History")
        
        chart_col, sample_col = st.columns([1, 1])
        
        with chart_col:
            st.markdown("#### 📊 Loss Curves")
            latest_metrics_path = _load_latest_metrics_json(browse_log_dir) if browse_log_dir else _load_latest_metrics_json(DEFAULT_LOG_DIR)
            if latest_metrics_path is not None:
                df = _metrics_dataframe_from_logger_json(latest_metrics_path)
                if df is not None and 'epoch' in df.columns and {'g_loss', 'd_loss'}.issubset(df.columns):
                    st.line_chart(df.set_index('epoch')[['g_loss', 'd_loss']], height=300)
                    
                    # Summary stats
                    st.markdown("#### 📋 Final Metrics")
                    latest = df.iloc[-1]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Epochs", f"{int(latest.get('epoch', 0))}")
                    m2.metric("Final G Loss", f"{float(latest.get('g_loss', 0)):.4f}")
                    m3.metric("Final D Loss", f"{float(latest.get('d_loss', 0)):.4f}")
                    
                    # Health check
                    health = _check_loss_health(df)
                    if health["warnings"]:
                        st.warning("⚠️ " + " | ".join(health["warnings"]))
                    else:
                        st.success("✅ Training completed normally")
                else:
                    st.info("No metrics data available for this run.")
            else:
                st.info("No metrics found. Start a training run to see progress here.")
        
        with sample_col:
            st.markdown("#### 🖼️ Generated Samples")
            sample_dir = browse_sample_dir if browse_sample_dir else DEFAULT_SAMPLE_DIR
            sample_files = sorted(list(sample_dir.glob("*.png")), key=lambda x: x.stat().st_mtime, reverse=True) if sample_dir.exists() else []
            
            if sample_files:
                # Show latest sample prominently
                latest_sample = sample_files[0]
                st.image(str(latest_sample), caption=f"Latest: {latest_sample.name}", use_container_width=True)
                
                # Show sample gallery
                if len(sample_files) > 1:
                    with st.expander(f"📁 View all {len(sample_files)} samples"):
                        sample_cols = st.columns(4)
                        for i, sf in enumerate(sample_files[:12]):
                            with sample_cols[i % 4]:
                                st.image(str(sf), caption=sf.name, use_container_width=True)
            else:
                st.info("No samples generated yet for this run.")
        
        # Logs viewer
        with st.expander("📜 View Training Logs"):
            if browse_log_dir is not None:
                log_path = browse_log_dir / "training_output.log"
            else:
                log_path = DEFAULT_LOG_DIR / "training_output.log"
            
            if log_path.exists():
                terminal_text = _tail_text_file(log_path) or ""
                lines = terminal_text.splitlines()
                tail = [ln.rstrip("\n") for ln in lines[-TRAINING_RECENT_LOGS_LINES:]] if lines else ["No logs available"]
                st.dataframe(
                    pd.DataFrame({"log": tail}),
                    hide_index=True,
                    use_container_width=True,
                    height=300,
                )
            else:
                st.info("No training logs found for this run.")

def render_about_page():
    st.header("ℹ️ About Vanilla GAN Signatures")
    st.markdown("""
    ### 🧠 The Technology
    This application uses a **Generative Adversarial Network (GAN)** to synthesize realistic handwritten signatures.
    
    ### 🔄 Workflow
    1. **Preprocess:** Convert raw images to a standardized format (64x64, binarized).
    2. **Train:** Run the GAN training loop to learn the data distribution.
    3. **Generate:** Use the trained Generator to create infinite synthetic samples.
    
    ### 🛠️ Features
    - **Batch Generation:** Create hundreds of unique variations instantly.
    - **Quality Control:** Uses the Discriminator to score and filter the best signatures.
    - **Morphing:** Explore the "latent space" between two signatures.
    
    ---
    *Built with PyTorch & Streamlit*
    """)

# -----------------------------------------------------------------------------
# Main App Structure
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Signature Generator Suite",
        page_icon="✍️",
        layout="wide"
    )

    # Custom CSS (global)
    st.markdown(
        """
                <style>
                    /* Layout + typography */
                    .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; }
                    .stApp { background: var(--background-color); }

                    /* Hero */
                    .sg-hero {
                        padding: 1.0rem 1.1rem;
                        border-radius: 14px;
                        border: 1px solid rgba(255,255,255,0.08);
                        background: rgba(255,255,255,0.03);
                        margin-bottom: 0.8rem;
                    }
                    .sg-hero-title { font-size: 1.65rem; font-weight: 750; line-height: 1.1; }
                    .sg-hero-sub { margin-top: 0.35rem; opacity: 0.85; font-size: 0.95rem; }

                    /* Section titles */
                    .sg-section-title { font-size: 1.15rem; font-weight: 700; margin-top: 0.35rem; }
                    .sg-section-sub { opacity: 0.78; margin-top: 0.2rem; margin-bottom: 0.6rem; }

                    /* Banners */
                    .sg-banner {
                        padding: 0.75rem 0.9rem;
                        border-radius: 12px;
                        border: 1px solid rgba(255,255,255,0.08);
                        background: rgba(255,255,255,0.03);
                        margin: 0.75rem 0 0.25rem 0;
                    }
                    .sg-banner.ok { border-left: 4px solid var(--primary-color); }
                    .sg-banner.warn { border-left: 4px solid rgba(255,255,255,0.25); }
                    .sg-banner-title { font-weight: 700; }
                    .sg-banner-sub { opacity: 0.82; font-size: 0.92rem; margin-top: 0.15rem; }

                    /* Images */
                    div[data-testid="stImage"] {
                        background-color: #ffffff;
                        padding: 12px;
                        border: 1px solid rgba(0,0,0,0.08);
                        border-radius: 12px;
                        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
                    }
                    div[data-testid="stImage"] img { border-radius: 8px; }

                    /* Expanders */
                    details[data-testid="stExpander"] {
                        border-radius: 12px;
                        border: 1px solid rgba(255,255,255,0.08);
                        background: rgba(255,255,255,0.02);
                    }

                    /* Tabs: slightly bolder active state */
                    button[data-baseweb="tab"] { font-weight: 600; }
                    button[data-baseweb="tab"][aria-selected="true"] {
                        border-bottom: 2px solid var(--primary-color) !important;
                    }

                    /* Buttons: rounder + consistent height */
                    button[kind], div[data-testid="stDownloadButton"] button {
                        border-radius: 12px !important;
                    }
                    div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
                        min-height: 42px;
                    }

                    /* Selected pill */
                    .sg-selected-pill {
                        display: inline-block;
                        padding: 0.15rem 0.5rem;
                        border-radius: 999px;
                        font-size: 0.78rem;
                        font-weight: 650;
                        margin-bottom: 0.35rem;
                        border: 1px solid rgba(255,255,255,0.12);
                        background: rgba(255,255,255,0.05);
                    }
                </style>
        """,
        unsafe_allow_html=True,
    )

    # Navigation
    with st.sidebar:
        st.title("✍️ Signature GAN")
        page = st.radio("Navigation", ["Generate", "Preprocess", "Train", "About"])
        st.divider()

    if page == "Generate":
        render_generation_page()
    elif page == "Preprocess":
        render_preprocessing_page()
    elif page == "Train":
        render_training_page()
    elif page == "About":
        render_about_page()

if __name__ == "__main__":
    main()
