"""
Signature Verification Model Evaluation.

This module evaluates trained Siamese networks for signature verification,
computing comprehensive metrics and generating ROC/DET curve visualizations.
Supports comparison between baseline and augmented (GAN-enhanced) models.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    det_curve,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =============================================================================
# Model Architecture (must match training)
# =============================================================================


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting features from 64x64 signature images.
    
    Architecture:
        Conv2d(1,32) -> Conv2d(32,64) -> Conv2d(64,128) -> FC -> embedding
    """
    
    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Initialize the CNN encoder.
        
        Args:
            embedding_dim: Dimension of the output embedding vector.
        """
        super(CNNEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 3 pooling layers: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64).
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese Network for signature verification.
    
    Uses twin CNN encoders with shared weights to compare two signatures
    and determine if they belong to the same person.
    """
    
    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Initialize the Siamese network.
        
        Args:
            embedding_dim: Dimension of the embedding vectors.
        """
        super(SiameseNetwork, self).__init__()
        
        self.encoder = CNNEncoder(embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        
        # Classifier for BCE loss approach
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64).
            
        Returns:
            Embedding tensor.
        """
        return self.encoder(x)
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of images.
        
        Args:
            x1: First image tensor of shape (batch_size, 1, 64, 64).
            x2: Second image tensor of shape (batch_size, 1, 64, 64).
            
        Returns:
            Tuple of (embedding1, embedding2, similarity_score).
        """
        # Get embeddings for both images
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        
        # Compute absolute difference for classification
        diff = torch.abs(embedding1 - embedding2)
        similarity = self.classifier(diff)
        
        return embedding1, embedding2, similarity


# =============================================================================
# Test Dataset
# =============================================================================


class SignatureTestDataset(Dataset):
    """
    Test dataset for signature verification evaluation.
    
    Supports two directory structures:
        1. Organized by user: test_dir/user_id/*.png
        2. Flat with naming convention: test_dir/userID_sigNum.png
        
    Generates genuine (same user) and forgery (different user) pairs.
    """
    
    def __init__(
        self,
        test_dir: str,
        transform: Optional[transforms.Compose] = None,
        pairs_per_user: int = 20,
        seed: int = 42
    ) -> None:
        """
        Initialize the test dataset.
        
        Args:
            test_dir: Directory containing test signature images.
            transform: Image transformations to apply.
            pairs_per_user: Number of pairs to generate per user.
            seed: Random seed for reproducibility.
        """
        self.test_dir = Path(test_dir)
        self.pairs_per_user = pairs_per_user
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Load user signatures
        self.user_signatures: Dict[str, List[Path]] = {}
        self._load_signatures()
        
        # Generate test pairs
        self.pairs: List[Tuple[Path, Path, int]] = []
        self._generate_pairs()
        
    def _load_signatures(self) -> None:
        """Load signatures from test directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Check for subdirectories (user-based organization)
        subdirs = [d for d in self.test_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            for user_dir in subdirs:
                user_id = user_dir.name
                user_images = [
                    f for f in user_dir.iterdir() 
                    if f.suffix.lower() in image_extensions
                ]
                if len(user_images) >= 2:
                    self.user_signatures[user_id] = user_images
        else:
            # Flat structure: group by filename prefix
            all_images = [
                f for f in self.test_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            for img_path in all_images:
                filename = img_path.stem
                parts = filename.split('_')
                user_id = parts[0] if parts else filename
                
                if user_id not in self.user_signatures:
                    self.user_signatures[user_id] = []
                self.user_signatures[user_id].append(img_path)
            
            # Filter users with less than 2 signatures
            self.user_signatures = {
                k: v for k, v in self.user_signatures.items() 
                if len(v) >= 2
            }
        
        print(f"[Test Dataset] Loaded {len(self.user_signatures)} users")
        
    def _generate_pairs(self) -> None:
        """Generate genuine and forgery pairs for testing."""
        user_ids = list(self.user_signatures.keys())
        
        if len(user_ids) < 2:
            print("WARNING: Need at least 2 users to generate forgery pairs")
            return
        
        for user_id in user_ids:
            user_sigs = self.user_signatures[user_id]
            
            # Generate genuine pairs (same user) - label=1
            num_genuine = min(self.pairs_per_user, len(user_sigs) * (len(user_sigs) - 1) // 2)
            for _ in range(num_genuine):
                if len(user_sigs) >= 2:
                    indices = np.random.choice(len(user_sigs), 2, replace=False)
                    sig1, sig2 = user_sigs[indices[0]], user_sigs[indices[1]]
                    self.pairs.append((sig1, sig2, 1))  # Genuine pair
            
            # Generate forgery pairs (different users) - label=0
            other_users = [u for u in user_ids if u != user_id]
            for _ in range(self.pairs_per_user):
                other_user = np.random.choice(other_users)
                sig1 = user_sigs[np.random.randint(len(user_sigs))]
                other_sigs = self.user_signatures[other_user]
                sig2 = other_sigs[np.random.randint(len(other_sigs))]
                self.pairs.append((sig1, sig2, 0))  # Forgery pair
        
        # Shuffle pairs
        np.random.shuffle(self.pairs)
        
        genuine_count = sum(1 for _, _, label in self.pairs if label == 1)
        forgery_count = len(self.pairs) - genuine_count
        print(f"[Test Dataset] Generated {len(self.pairs)} pairs: "
              f"{genuine_count} genuine, {forgery_count} forgery")
        
    def __len__(self) -> int:
        """Return the number of test pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a test pair.
        
        Args:
            idx: Index of the pair.
            
        Returns:
            Tuple of (image1, image2, label).
        """
        sig1_path, sig2_path, label = self.pairs[idx]
        
        img1 = Image.open(sig1_path).convert('L')
        img2 = Image.open(sig2_path).convert('L')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# =============================================================================
# Model Loading
# =============================================================================


def load_model(
    checkpoint_path: str,
    device: torch.device,
    embedding_dim: Optional[int] = None
) -> Tuple[SiameseNetwork, Dict[str, Any]]:
    """
    Load a trained Siamese model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth file).
        device: Device to load the model on.
        embedding_dim: Override embedding dimension (optional).
        
    Returns:
        Tuple of (model, checkpoint_metadata).
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract embedding dimension
    emb_dim = embedding_dim or checkpoint.get('embedding_dim', 128)
    
    # Create and load model
    model = SiameseNetwork(embedding_dim=emb_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract metadata
    metadata = {
        'embedding_dim': emb_dim,
        'val_accuracy': checkpoint.get('val_accuracy', None),
        'epoch': checkpoint.get('epoch', None),
        'includes_synthetic': checkpoint.get('includes_synthetic', False),
        'checkpoint_path': str(checkpoint_path)
    }
    
    print(f"[Model] Loaded from: {checkpoint_path}")
    print(f"[Model] Embedding dim: {emb_dim}, "
          f"Val accuracy: {metadata['val_accuracy']:.4f if metadata['val_accuracy'] else 'N/A'}")
    
    return model, metadata


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_verification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive signature verification metrics.
    
    Args:
        y_true: Ground truth labels (1 for genuine, 0 for forgery).
        y_scores: Predicted similarity scores (0 to 1).
        y_pred: Predicted binary labels.
        threshold: Decision threshold for binary classification.
        
    Returns:
        Dictionary containing all computed metrics.
    """
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # FAR (False Acceptance Rate): Forgeries accepted as genuine
    # FAR = FP / (FP + TN) = FP / Total Negatives (forgeries)
    total_forgeries = fp + tn
    far = fp / total_forgeries if total_forgeries > 0 else 0.0
    
    # FRR (False Rejection Rate): Genuine signatures rejected
    # FRR = FN / (FN + TP) = FN / Total Positives (genuine)
    total_genuine = fn + tp
    frr = fn / total_genuine if total_genuine > 0 else 0.0
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC-AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # EER (Equal Error Rate)
    # Find threshold where FAR = FRR
    fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    eer_threshold = roc_thresholds[eer_threshold_idx] if len(roc_thresholds) > eer_threshold_idx else threshold
    
    metrics = {
        'accuracy': float(accuracy),
        'far': float(far),
        'frr': float(frr),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_genuine': int(total_genuine),
        'total_forgeries': int(total_forgeries),
        'threshold': float(threshold)
    }
    
    return metrics


def compute_eer_from_scores(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from scores.
    
    Args:
        y_true: Ground truth labels.
        y_scores: Predicted similarity scores.
        
    Returns:
        Tuple of (EER value, EER threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the point where FAR = FRR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx] if len(thresholds) > eer_idx else 0.5
    
    return float(eer), float(eer_threshold)


# =============================================================================
# Model Evaluation
# =============================================================================


def evaluate_model(
    model: SiameseNetwork,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a Siamese model on test data.
    
    Args:
        model: The Siamese network model.
        dataloader: Test data loader.
        device: Device to evaluate on.
        threshold: Decision threshold for binary classification.
        
    Returns:
        Tuple of (metrics_dict, y_true, y_scores, y_pred).
    """
    model.eval()
    
    all_labels: List[float] = []
    all_scores: List[float] = []
    
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Get similarity scores
            _, _, similarity = model(img1, img2)
            
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(similarity.squeeze().cpu().numpy().tolist())
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = compute_verification_metrics(y_true, y_scores, y_pred, threshold)
    
    return metrics, y_true, y_scores, y_pred


# =============================================================================
# Visualization
# =============================================================================


def plot_roc_curve(
    results: Dict[str, Dict[str, Any]],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot ROC curves for all evaluated models.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
        save_path: Path to save the plot.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (model_name, data) in enumerate(results.items()):
        y_true = data['y_true']
        y_scores = data['y_scores']
        metrics = data['metrics']
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = metrics['roc_auc']
        
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        plt.plot(
            fpr, tpr,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f'{model_name} (AUC = {roc_auc:.4f})'
        )
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12)
    plt.title('ROC Curve - Signature Verification', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] ROC curve saved to: {save_path}")


def plot_det_curve(
    results: Dict[str, Dict[str, Any]],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot DET (Detection Error Tradeoff) curves for all evaluated models.
    
    DET curves show FRR vs FAR on a log scale, making it easier to
    compare systems at low error rates.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
        save_path: Path to save the plot.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (model_name, data) in enumerate(results.items()):
        y_true = data['y_true']
        y_scores = data['y_scores']
        metrics = data['metrics']
        
        fpr, fnr, _ = det_curve(y_true, y_scores)
        eer = metrics['eer']
        
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        plt.plot(
            fpr, fnr,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f'{model_name} (EER = {eer:.4f})'
        )
    
    # Plot EER line (diagonal where FAR = FRR)
    plt.plot([0.001, 1], [0.001, 1], 'k--', linewidth=1, label='EER Line')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.001, 1])
    plt.ylim([0.001, 1])
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title('DET Curve - Signature Verification', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] DET curve saved to: {save_path}")


def plot_score_distribution(
    results: Dict[str, Dict[str, Any]],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot score distributions for genuine and forgery pairs.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
        save_path: Path to save the plot.
        figsize: Figure size.
    """
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(figsize[0], figsize[1]))
    
    if num_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results.items()):
        ax = axes[idx]
        y_true = data['y_true']
        y_scores = data['y_scores']
        metrics = data['metrics']
        
        # Separate scores by class
        genuine_scores = y_scores[y_true == 1]
        forgery_scores = y_scores[y_true == 0]
        
        # Plot histograms
        ax.hist(genuine_scores, bins=30, alpha=0.7, color='#2ecc71', 
                label='Genuine', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(forgery_scores, bins=30, alpha=0.7, color='#e74c3c', 
                label='Forgery', density=True, edgecolor='black', linewidth=0.5)
        
        # Plot threshold line
        threshold = metrics.get('eer_threshold', 0.5)
        ax.axvline(x=threshold, color='#3498db', linestyle='--', linewidth=2, 
                   label=f'EER Threshold ({threshold:.3f})')
        
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{model_name}\nEER: {metrics["eer"]:.4f}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] Score distribution saved to: {save_path}")


def plot_comparison_bar_chart(
    results: Dict[str, Dict[str, Any]],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot bar chart comparing key metrics across models.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
        save_path: Path to save the plot.
        figsize: Figure size.
    """
    metrics_to_plot = ['accuracy', 'far', 'frr', 'eer', 'roc_auc', 'f1_score']
    metric_labels = ['Accuracy', 'FAR', 'FRR', 'EER', 'ROC-AUC', 'F1 Score']
    
    model_names = list(results.keys())
    num_models = len(model_names)
    num_metrics = len(metrics_to_plot)
    
    x = np.arange(num_metrics)
    width = 0.35 if num_models == 2 else 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    for idx, model_name in enumerate(model_names):
        metrics = results[model_name]['metrics']
        values = [metrics[m] for m in metrics_to_plot]
        
        offset = (idx - (num_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model_name, 
                      color=colors[idx % len(colors)], edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Comparison - Signature Verification Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] Comparison bar chart saved to: {save_path}")


# =============================================================================
# Report Generation
# =============================================================================


def generate_evaluation_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report in JSON format.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
        output_path: Path to save the JSON report.
        
    Returns:
        The complete report dictionary.
    """
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'num_models_evaluated': len(results),
        'models': {}
    }
    
    for model_name, data in results.items():
        model_report = {
            'model_metadata': data.get('metadata', {}),
            'metrics': data['metrics'],
            'num_test_samples': len(data['y_true']),
            'genuine_samples': int(np.sum(data['y_true'] == 1)),
            'forgery_samples': int(np.sum(data['y_true'] == 0)),
        }
        report['models'][model_name] = model_report
    
    # Add comparison summary if multiple models
    if len(results) > 1:
        comparison = {}
        metrics_to_compare = ['accuracy', 'far', 'frr', 'eer', 'roc_auc', 'f1_score']
        
        for metric in metrics_to_compare:
            values = {name: data['metrics'][metric] for name, data in results.items()}
            best_model = max(values.keys(), key=lambda k: values[k]) if metric in ['accuracy', 'roc_auc', 'f1_score'] else min(values.keys(), key=lambda k: values[k])
            
            comparison[metric] = {
                'values': values,
                'best_model': best_model,
                'improvement': None
            }
            
            # Calculate improvement if comparing baseline vs augmented
            if 'Baseline' in values and 'Augmented' in values:
                baseline_val = values['Baseline']
                augmented_val = values['Augmented']
                if metric in ['accuracy', 'roc_auc', 'f1_score']:
                    improvement = ((augmented_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                else:
                    improvement = ((baseline_val - augmented_val) / baseline_val * 100) if baseline_val != 0 else 0
                comparison[metric]['improvement'] = f"{improvement:+.2f}%"
        
        report['comparison_summary'] = comparison
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[Report] Evaluation report saved to: {output_path}")
    
    return report


def print_evaluation_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a formatted summary of evaluation results to console.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
    """
    print("\n" + "=" * 70)
    print("SIGNATURE VERIFICATION EVALUATION SUMMARY")
    print("=" * 70)
    
    for model_name, data in results.items():
        metrics = data['metrics']
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  FAR:          {metrics['far']:.4f}")
        print(f"  FRR:          {metrics['frr']:.4f}")
        print(f"  EER:          {metrics['eer']:.4f}")
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"  F1 Score:     {metrics['f1_score']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.4f}")
        print(f"  EER Threshold:{metrics['eer_threshold']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}")
        print(f"    FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    
    # Comparison if multiple models
    if len(results) > 1:
        print("\n" + "-" * 70)
        print("MODEL COMPARISON")
        print("-" * 70)
        
        model_names = list(results.keys())
        metrics_to_compare = ['accuracy', 'far', 'frr', 'eer', 'roc_auc']
        
        # Header
        header = f"{'Metric':<15}"
        for name in model_names:
            header += f"{name:<15}"
        header += "Winner"
        print(header)
        print("-" * len(header))
        
        for metric in metrics_to_compare:
            row = f"{metric.upper():<15}"
            values = []
            for name in model_names:
                val = results[name]['metrics'][metric]
                values.append(val)
                row += f"{val:<15.4f}"
            
            # Determine winner
            if metric in ['accuracy', 'roc_auc']:
                winner_idx = np.argmax(values)
            else:
                winner_idx = np.argmin(values)
            row += model_names[winner_idx]
            
            print(row)
    
    print("\n" + "=" * 70)


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================


def evaluate_signature_verifier(
    baseline_model_path: Optional[str],
    augmented_model_path: Optional[str],
    test_dir: str,
    output_dir: str,
    batch_size: int = 32,
    pairs_per_user: int = 20,
    threshold: float = 0.5,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete signature verification evaluation pipeline.
    
    Args:
        baseline_model_path: Path to baseline model checkpoint.
        augmented_model_path: Path to augmented (GAN-enhanced) model checkpoint.
        test_dir: Directory containing test signature images.
        output_dir: Directory to save evaluation outputs.
        batch_size: Batch size for evaluation.
        pairs_per_user: Number of pairs to generate per user.
        threshold: Decision threshold for binary classification.
        device: Device to run evaluation on.
        
    Returns:
        Complete evaluation results dictionary.
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"[Setup] Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup test dataset
    print("\n[Setup] Loading test dataset...")
    test_dataset = SignatureTestDataset(
        test_dir=test_dir,
        pairs_per_user=pairs_per_user
    )
    
    if len(test_dataset) == 0:
        raise ValueError(f"No test pairs generated from: {test_dir}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate models
    results: Dict[str, Dict[str, Any]] = {}
    
    # Evaluate baseline model
    if baseline_model_path:
        print("\n" + "=" * 50)
        print("Evaluating BASELINE Model")
        print("=" * 50)
        
        baseline_model, baseline_metadata = load_model(baseline_model_path, device)
        metrics, y_true, y_scores, y_pred = evaluate_model(
            baseline_model, test_loader, device, threshold
        )
        
        results['Baseline'] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_scores': y_scores,
            'y_pred': y_pred,
            'metadata': baseline_metadata
        }
    
    # Evaluate augmented model
    if augmented_model_path:
        print("\n" + "=" * 50)
        print("Evaluating AUGMENTED Model")
        print("=" * 50)
        
        augmented_model, augmented_metadata = load_model(augmented_model_path, device)
        metrics, y_true, y_scores, y_pred = evaluate_model(
            augmented_model, test_loader, device, threshold
        )
        
        results['Augmented'] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_scores': y_scores,
            'y_pred': y_pred,
            'metadata': augmented_metadata
        }
    
    if not results:
        raise ValueError("At least one model path must be provided for evaluation.")
    
    # Print summary
    print_evaluation_summary(results)
    
    # Generate plots
    print("\n[Plots] Generating visualizations...")
    plot_roc_curve(results, output_path / 'roc_curve.png')
    plot_det_curve(results, output_path / 'det_curve.png')
    plot_score_distribution(results, output_path / 'score_distribution.png')
    
    if len(results) > 1:
        plot_comparison_bar_chart(results, output_path / 'comparison_metrics.png')
    
    # Generate report
    report = generate_evaluation_report(results, output_path / 'evaluation_report.json')
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Output directory: {output_path}")
    print(f"  - ROC curve: roc_curve.png")
    print(f"  - DET curve: det_curve.png")
    print(f"  - Score distribution: score_distribution.png")
    if len(results) > 1:
        print(f"  - Comparison chart: comparison_metrics.png")
    print(f"  - Report: evaluation_report.json")
    
    return report


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Siamese network models for signature verification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--baseline_model',
        type=str,
        default=None,
        help='Path to baseline Siamese model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--augmented_model',
        type=str,
        default=None,
        help='Path to augmented (GAN-enhanced) Siamese model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--test_dir',
        type=str,
        required=True,
        help='Directory containing test signature images (organized by user)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation outputs (plots, report)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--pairs_per_user',
        type=int,
        default=20,
        help='Number of genuine/forgery pairs to generate per user'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold for binary classification'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run evaluation on (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.baseline_model and not args.augmented_model:
        parser.error("At least one of --baseline_model or --augmented_model must be provided")
    
    print("=" * 70)
    print("SIGNATURE VERIFICATION MODEL EVALUATION")
    print("=" * 70)
    print(f"Baseline model:  {args.baseline_model or 'Not provided'}")
    print(f"Augmented model: {args.augmented_model or 'Not provided'}")
    print(f"Test directory:  {args.test_dir}")
    print(f"Output directory:{args.output_dir}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Pairs per user:  {args.pairs_per_user}")
    print(f"Threshold:       {args.threshold}")
    print("=" * 70)
    
    evaluate_signature_verifier(
        baseline_model_path=args.baseline_model,
        augmented_model_path=args.augmented_model,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        pairs_per_user=args.pairs_per_user,
        threshold=args.threshold,
        device=args.device
    )


if __name__ == '__main__':
    main()
