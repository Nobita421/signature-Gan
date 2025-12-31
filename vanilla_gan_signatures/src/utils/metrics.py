"""GAN evaluation metrics for signature generation."""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import List, Dict, Optional, Union
from collections import defaultdict

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def get_inception_model(device: torch.device) -> nn.Module:
    """Load InceptionV3 model for FID calculation."""
    if not INCEPTION_AVAILABLE:
        raise ImportError("torchvision required for FID calculation")
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def extract_features(images: torch.Tensor, model: nn.Module, device: torch.device) -> np.ndarray:
    """Extract InceptionV3 features from images."""
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    features = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32].to(device)
            feat = model(batch).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)


def calculate_fid(real_images: torch.Tensor, fake_images: torch.Tensor, 
                  device: Optional[torch.device] = None) -> float:
    """
    Calculate Fréchet Inception Distance between real and fake images.
    
    Args:
        real_images: Real images tensor (N, C, H, W)
        fake_images: Generated images tensor (N, C, H, W)
        device: Computation device
    
    Returns:
        FID score (lower is better)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_inception_model(device)
    
    real_features = extract_features(real_images, model, device)
    fake_features = extract_features(fake_images, model, device)
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def calculate_lpips_diversity(images_list: List[torch.Tensor], 
                              device: Optional[torch.device] = None) -> float:
    """
    Calculate LPIPS diversity score between image pairs.
    
    Args:
        images_list: List of image tensors to compare
        device: Computation device
    
    Returns:
        Average LPIPS distance (higher means more diverse)
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("lpips package required: pip install lpips")
    
    if len(images_list) < 2:
        return 0.0
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    distances = []
    with torch.no_grad():
        for i in range(len(images_list)):
            for j in range(i + 1, min(i + 10, len(images_list))):
                img1 = images_list[i].to(device)
                img2 = images_list[j].to(device)
                if img1.dim() == 3:
                    img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
                if img1.shape[1] == 1:
                    img1, img2 = img1.repeat(1, 3, 1, 1), img2.repeat(1, 3, 1, 1)
                dist = lpips_fn(img1, img2).item()
                distances.append(dist)
    
    return float(np.mean(distances)) if distances else 0.0


def calculate_stroke_density(images: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Analyze stroke density distribution in signature images.
    
    Args:
        images: Image tensor (N, C, H, W), values in [0, 1] or [-1, 1]
        threshold: Threshold for stroke detection
    
    Returns:
        Dict with mean, std, min, max stroke density
    """
    if images.min() < 0:
        images = (images + 1) / 2
    
    if images.shape[1] > 1:
        images = images.mean(dim=1, keepdim=True)
    
    stroke_pixels = (images < threshold).float()
    densities = stroke_pixels.view(images.shape[0], -1).mean(dim=1).cpu().numpy()
    
    return {
        'mean': float(np.mean(densities)),
        'std': float(np.std(densities)),
        'min': float(np.min(densities)),
        'max': float(np.max(densities))
    }


def calculate_foreground_ratio(images: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate foreground pixel percentage in images.
    
    Args:
        images: Image tensor (N, C, H, W), values in [0, 1] or [-1, 1]
        threshold: Threshold for foreground detection
    
    Returns:
        Dict with mean, std foreground ratio
    """
    if images.min() < 0:
        images = (images + 1) / 2
    
    if images.shape[1] > 1:
        images = images.mean(dim=1, keepdim=True)
    
    foreground = (images < threshold).float()
    ratios = foreground.view(images.shape[0], -1).mean(dim=1).cpu().numpy()
    
    return {
        'mean': float(np.mean(ratios)),
        'std': float(np.std(ratios)),
        'percentiles': {
            '25': float(np.percentile(ratios, 25)),
            '50': float(np.percentile(ratios, 50)),
            '75': float(np.percentile(ratios, 75))
        }
    }


class MetricsTracker:
    """Track GAN training metrics over epochs."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.epoch_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def add(self, name: str, value: Union[float, torch.Tensor]) -> None:
        """Add a metric value."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.epoch_metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """Get average of metric for current epoch."""
        values = self.epoch_metrics.get(name, [])
        return float(np.mean(values)) if values else 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages for all tracked metrics."""
        return {name: self.get_average(name) for name in self.epoch_metrics}
    
    def reset(self) -> None:
        """Reset epoch metrics and store averages in history."""
        for name, values in self.epoch_metrics.items():
            if values:
                self.metrics[name].append(float(np.mean(values)))
        self.epoch_metrics.clear()
    
    def get_history(self, name: str) -> List[float]:
        """Get full history of a metric across epochs."""
        return self.metrics.get(name, [])
    
    def get_last(self, name: str, default: float = 0.0) -> float:
        """Get last recorded average for a metric."""
        history = self.metrics.get(name, [])
        return history[-1] if history else default
