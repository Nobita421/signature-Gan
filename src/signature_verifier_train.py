"""
Signature Verification Model Training using Siamese Network.

This module implements a Siamese network for signature verification,
training on pairs of signatures to distinguish genuine from forged signatures.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.
    
    Loss = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    
    Where:
        Y = 0 for similar pairs, 1 for dissimilar pairs
        D = Euclidean distance between embeddings
    """
    
    def __init__(self, margin: float = 2.0) -> None:
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor, 
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embedding tensor.
            embedding2: Second embedding tensor.
            label: Label tensor (1 for same, 0 for different).
            
        Returns:
            Scalar loss tensor.
        """
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        # label=1 means same pair, label=0 means different pair
        loss = (label) * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(
                   torch.clamp(self.margin - euclidean_distance, min=0.0), 2
               )
        
        return loss.mean()


class SignaturePairDataset(Dataset):
    """
    Dataset for creating signature pairs for Siamese network training.
    
    Creates:
        - Genuine-genuine pairs (label=1): Same user signatures
        - Genuine-random pairs (label=0): Different user signatures
    """
    
    def __init__(
        self,
        data_dir: str,
        synthetic_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        pairs_per_user: int = 10
    ) -> None:
        """
        Initialize the signature pair dataset.
        
        Args:
            data_dir: Directory containing real signature images organized by user.
            synthetic_dir: Optional directory containing synthetic signatures.
            transform: Image transformations to apply.
            pairs_per_user: Number of pairs to generate per user.
        """
        self.data_dir = Path(data_dir)
        self.synthetic_dir = Path(synthetic_dir) if synthetic_dir else None
        self.pairs_per_user = pairs_per_user
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load user signatures
        self.user_signatures: Dict[str, List[Path]] = {}
        self._load_signatures()
        
        # Generate pairs
        self.pairs: List[Tuple[Path, Path, int]] = []
        self._generate_pairs()
        
    def _load_signatures(self) -> None:
        """Load signatures from data directory organized by user."""
        # Handle flat directory structure (all images in one folder)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Check if directory has subdirectories (user-based organization)
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # User-based organization: each subdirectory is a user
            for user_dir in subdirs:
                user_id = user_dir.name
                user_images = [
                    f for f in user_dir.iterdir() 
                    if f.suffix.lower() in image_extensions
                ]
                if len(user_images) >= 2:
                    self.user_signatures[user_id] = user_images
        else:
            # Flat structure: group by filename prefix (e.g., "user001_sig1.png")
            all_images = [
                f for f in self.data_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            # Try to extract user IDs from filenames
            for img_path in all_images:
                # Assume format: userID_signatureNum.ext or similar
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
        
        # Add synthetic signatures if provided
        if self.synthetic_dir and self.synthetic_dir.exists():
            synthetic_images = [
                f for f in self.synthetic_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            if synthetic_images:
                # Add synthetic signatures as a separate "synthetic" user
                # or distribute among existing users
                self.user_signatures['_synthetic_'] = synthetic_images
        
        print(f"Loaded {len(self.user_signatures)} users with signatures")
        for user_id, sigs in self.user_signatures.items():
            print(f"  {user_id}: {len(sigs)} signatures")
            
    def _generate_pairs(self) -> None:
        """Generate positive and negative pairs."""
        user_ids = list(self.user_signatures.keys())
        
        for user_id in user_ids:
            user_sigs = self.user_signatures[user_id]
            
            # Skip synthetic-only comparisons for more meaningful training
            if user_id == '_synthetic_':
                continue
                
            # Generate positive pairs (same user)
            for _ in range(self.pairs_per_user):
                if len(user_sigs) >= 2:
                    sig1, sig2 = random.sample(user_sigs, 2)
                    self.pairs.append((sig1, sig2, 1))  # label=1 for same
            
            # Generate negative pairs (different users)
            other_users = [u for u in user_ids if u != user_id]
            for _ in range(self.pairs_per_user):
                if other_users:
                    other_user = random.choice(other_users)
                    sig1 = random.choice(user_sigs)
                    sig2 = random.choice(self.user_signatures[other_user])
                    self.pairs.append((sig1, sig2, 0))  # label=0 for different
        
        # Shuffle pairs
        random.shuffle(self.pairs)
        print(f"Generated {len(self.pairs)} pairs")
        
    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of signatures and their label.
        
        Args:
            idx: Index of the pair.
            
        Returns:
            Tuple of (image1, image2, label).
        """
        sig1_path, sig2_path, label = self.pairs[idx]
        
        # Load images
        img1 = Image.open(sig1_path).convert('L')
        img2 = Image.open(sig2_path).convert('L')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def train_epoch(
    model: SiameseNetwork,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_bce: nn.BCELoss,
    criterion_contrastive: ContrastiveLoss,
    device: torch.device,
    use_contrastive: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The Siamese network model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion_bce: BCE loss function.
        criterion_contrastive: Contrastive loss function.
        device: Device to train on.
        use_contrastive: Whether to use contrastive loss in addition to BCE.
        
    Returns:
        Dictionary with training metrics.
    """
    model.train()
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_contrastive_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        emb1, emb2, similarity = model(img1, img2)
        
        # Compute losses
        bce_loss = criterion_bce(similarity.squeeze(), labels)
        
        if use_contrastive:
            contrastive_loss = criterion_contrastive(emb1, emb2, labels)
            loss = bce_loss + 0.5 * contrastive_loss
            total_contrastive_loss += contrastive_loss.item()
        else:
            loss = bce_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        
        # Accuracy
        predictions = (similarity.squeeze() > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches if use_contrastive else 0.0,
        'accuracy': correct / total if total > 0 else 0.0
    }
    
    return metrics


def evaluate(
    model: SiameseNetwork,
    dataloader: DataLoader,
    criterion_bce: nn.BCELoss,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Args:
        model: The Siamese network model.
        dataloader: Evaluation data loader.
        criterion_bce: BCE loss function.
        device: Device to evaluate on.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            
            _, _, similarity = model(img1, img2)
            
            loss = criterion_bce(similarity.squeeze(), labels)
            total_loss += loss.item()
            
            predictions = (similarity.squeeze() > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': correct / total if total > 0 else 0.0
    }
    
    return metrics


def train_model(
    data_dir: str,
    synthetic_dir: Optional[str],
    epochs: int,
    model_output: str,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    embedding_dim: int = 128,
    device: Optional[str] = None
) -> Dict[str, str]:
    """
    Train signature verification models.
    
    Trains two models:
        1. Baseline model: trained on real signatures only
        2. Augmented model: trained on real + synthetic signatures (if provided)
    
    Args:
        data_dir: Directory containing real signature images.
        synthetic_dir: Optional directory containing synthetic signatures.
        epochs: Number of training epochs.
        model_output: Output directory for saved models.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimizer.
        embedding_dim: Dimension of embedding vectors.
        device: Device to train on (cuda/cpu).
        
    Returns:
        Dictionary with paths to saved models.
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Training on device: {device}")
    
    # Create output directory
    output_path = Path(model_output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define transforms with data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    saved_models = {}
    
    # ===== Train Baseline Model (Real signatures only) =====
    print("\n" + "="*60)
    print("Training BASELINE model (real signatures only)")
    print("="*60)
    
    baseline_dataset = SignaturePairDataset(
        data_dir=data_dir,
        synthetic_dir=None,  # No synthetic data for baseline
        transform=train_transform,
        pairs_per_user=20
    )
    
    if len(baseline_dataset) == 0:
        print("WARNING: No training pairs generated for baseline model.")
        print("Please ensure data directory contains signature images organized by user.")
    else:
        # Split into train/val
        train_size = int(0.8 * len(baseline_dataset))
        val_size = len(baseline_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            baseline_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Initialize model
        baseline_model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
        optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion_bce = nn.BCELoss()
        criterion_contrastive = ContrastiveLoss(margin=2.0)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            train_metrics = train_epoch(
                baseline_model, train_loader, optimizer,
                criterion_bce, criterion_contrastive, device
            )
            val_metrics = evaluate(baseline_model, val_loader, criterion_bce, device)
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                baseline_path = output_path / 'baseline_siamese_model.pth'
                torch.save({
                    'model_state_dict': baseline_model.state_dict(),
                    'embedding_dim': embedding_dim,
                    'val_accuracy': best_val_acc,
                    'epoch': epoch + 1
                }, baseline_path)
                print(f"  -> Saved best baseline model (val_acc: {best_val_acc:.4f})")
        
        saved_models['baseline'] = str(baseline_path)
        print(f"\nBaseline model saved to: {baseline_path}")
    
    # ===== Train Augmented Model (Real + Synthetic signatures) =====
    if synthetic_dir and Path(synthetic_dir).exists():
        print("\n" + "="*60)
        print("Training AUGMENTED model (real + synthetic signatures)")
        print("="*60)
        
        augmented_dataset = SignaturePairDataset(
            data_dir=data_dir,
            synthetic_dir=synthetic_dir,
            transform=train_transform,
            pairs_per_user=20
        )
        
        if len(augmented_dataset) == 0:
            print("WARNING: No training pairs generated for augmented model.")
        else:
            # Split into train/val
            train_size = int(0.8 * len(augmented_dataset))
            val_size = len(augmented_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                augmented_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            
            # Initialize model
            augmented_model = SiameseNetwork(embedding_dim=embedding_dim).to(device)
            optimizer = torch.optim.Adam(augmented_model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
            best_val_acc = 0.0
            
            for epoch in range(epochs):
                train_metrics = train_epoch(
                    augmented_model, train_loader, optimizer,
                    criterion_bce, criterion_contrastive, device
                )
                val_metrics = evaluate(augmented_model, val_loader, criterion_bce, device)
                scheduler.step()
                
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    augmented_path = output_path / 'augmented_siamese_model.pth'
                    torch.save({
                        'model_state_dict': augmented_model.state_dict(),
                        'embedding_dim': embedding_dim,
                        'val_accuracy': best_val_acc,
                        'epoch': epoch + 1,
                        'includes_synthetic': True
                    }, augmented_path)
                    print(f"  -> Saved best augmented model (val_acc: {best_val_acc:.4f})")
            
            saved_models['augmented'] = str(augmented_path)
            print(f"\nAugmented model saved to: {augmented_path}")
    else:
        print("\nNo synthetic directory provided or directory doesn't exist.")
        print("Skipping augmented model training.")
    
    return saved_models


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train Siamese network for signature verification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing real signature images (organized by user)'
    )
    
    parser.add_argument(
        '--synthetic_dir',
        type=str,
        default=None,
        help='Optional directory containing synthetic/GAN-generated signatures'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--model_output',
        type=str,
        default='./models',
        help='Output directory for saved models'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='Dimension of embedding vectors'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to train on (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Signature Verification Model Training")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Synthetic directory: {args.synthetic_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Model output: {args.model_output}")
    print("="*60)
    
    saved_models = train_model(
        data_dir=args.data_dir,
        synthetic_dir=args.synthetic_dir,
        epochs=args.epochs,
        model_output=args.model_output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        device=args.device
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Saved models:")
    for model_name, model_path in saved_models.items():
        print(f"  - {model_name}: {model_path}")
    
    return saved_models


if __name__ == '__main__':
    main()
