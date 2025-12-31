# Vanilla GAN Architecture for Signature Generation

This document describes the neural network architecture used in the Vanilla GAN Signature project.

---

## Overview

The Vanilla GAN consists of two neural networks trained in an adversarial manner:
- **Generator (G)**: Creates synthetic signatures from random noise
- **Discriminator (D)**: Distinguishes real signatures from generated ones

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐
│ Random Noise │ ──► │  Generator  │ ──► │ Fake Signature│
│   z ~ N(0,1) │     │      G      │     │               │
└─────────────┘     └─────────────┘     └───────┬───────┘
                                                │
                    ┌─────────────────────────────┤
                    │                             │
                    ▼                             ▼
            ┌───────────────┐             ┌───────────────┐
            │ Real Signature│             │ Fake Signature│
            └───────┬───────┘             └───────┬───────┘
                    │                             │
                    └──────────┬──────────────────┘
                               ▼
                    ┌─────────────────┐
                    │  Discriminator  │
                    │        D        │
                    └────────┬────────┘
                             │
                             ▼
                    Real (1) or Fake (0)
```

---

## Generator Architecture

The Generator transforms a latent vector into a 64×64 grayscale signature image.

### Architecture Details

```
Input: Latent vector z ∈ R^100 (sampled from N(0,1))

Layer 1: Dense
├── Input: 100
├── Output: 256 × 4 × 4 = 4096
├── Reshape: (256, 4, 4)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 2: Upsample Block
├── ConvTranspose2d: 256 → 128, kernel=4, stride=2, padding=1
├── Output: (128, 8, 8)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 3: Upsample Block
├── ConvTranspose2d: 128 → 64, kernel=4, stride=2, padding=1
├── Output: (64, 16, 16)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 4: Upsample Block
├── ConvTranspose2d: 64 → 32, kernel=4, stride=2, padding=1
├── Output: (32, 32, 32)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 5: Output Block
├── ConvTranspose2d: 32 → 1, kernel=4, stride=2, padding=1
├── Output: (1, 64, 64)
└── Activation: Tanh

Output: Signature image ∈ [-1, 1]^(64×64)
```

### Generator Code Structure

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64):
        # Project and reshape
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Upsample blocks
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 4→8
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 8→16
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 16→32
        self.conv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)     # 32→64
        
        # Normalization
        self.bn1-4 = nn.BatchNorm2d(...)
        
    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        x = leaky_relu(self.bn1(self.conv1(x)))
        x = leaky_relu(self.bn2(self.conv2(x)))
        x = leaky_relu(self.bn3(self.conv3(x)))
        x = tanh(self.conv4(x))
        return x
```

---

## Discriminator Architecture

The Discriminator classifies images as real or fake.

### Architecture Details

```
Input: Signature image ∈ R^(1×64×64)

Layer 1: Downsample Block
├── Conv2d: 1 → 64, kernel=4, stride=2, padding=1
├── Output: (64, 32, 32)
└── Activation: LeakyReLU(0.2)

Layer 2: Downsample Block
├── Conv2d: 64 → 128, kernel=4, stride=2, padding=1
├── Output: (128, 16, 16)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 3: Downsample Block
├── Conv2d: 128 → 256, kernel=4, stride=2, padding=1
├── Output: (256, 8, 8)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 4: Downsample Block
├── Conv2d: 256 → 512, kernel=4, stride=2, padding=1
├── Output: (512, 4, 4)
└── Activation: LeakyReLU(0.2) + BatchNorm

Layer 5: Output Block
├── Conv2d: 512 → 1, kernel=4, stride=1, padding=0
├── Output: (1, 1, 1)
└── Activation: Sigmoid

Output: Probability ∈ [0, 1] (real=1, fake=0)
```

### Discriminator Code Structure

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64):
        # Downsample blocks
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)      # 64→32
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)    # 32→16
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)   # 16→8
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)   # 8→4
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)     # 4→1
        
        # Normalization (no BN on first layer)
        self.bn2-4 = nn.BatchNorm2d(...)
        
    def forward(self, img):
        x = leaky_relu(self.conv1(img))
        x = leaky_relu(self.bn2(self.conv2(x)))
        x = leaky_relu(self.bn3(self.conv3(x)))
        x = leaky_relu(self.bn4(self.conv4(x)))
        x = sigmoid(self.conv5(x))
        return x.view(-1, 1)
```

---

## Loss Functions

### Binary Cross-Entropy Loss

The standard GAN uses Binary Cross-Entropy (BCE) loss:

```
L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]
L_G = -E[log(D(G(z)))]
```

Where:
- `x` = real signature
- `z` = random latent vector
- `D(x)` = discriminator output for real image
- `G(z)` = generated signature
- `D(G(z))` = discriminator output for fake image

### Label Smoothing

We apply **one-sided label smoothing** to improve training stability:

```python
# Real labels: 1.0 → 0.9 (smoothed)
real_labels = torch.full((batch_size, 1), 0.9)

# Fake labels: 0.0 → 0.0 (no smoothing)
fake_labels = torch.zeros((batch_size, 1))
```

**Benefits of Label Smoothing:**
- Prevents discriminator from becoming too confident
- Reduces overfitting to training data
- Provides smoother gradients for generator training

### Implementation

```python
criterion = nn.BCELoss()

# Discriminator loss
real_loss = criterion(D(real_images), real_labels * 0.9)  # Label smoothing
fake_loss = criterion(D(G(z)), fake_labels)
d_loss = (real_loss + fake_loss) / 2

# Generator loss
g_loss = criterion(D(G(z)), real_labels)  # Fool discriminator
```

---

## Training Procedure

### Algorithm

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # ─────────────────────────────────────
        # 1. Train Discriminator
        # ─────────────────────────────────────
        
        # Real images
        real_images = batch
        real_labels = 0.9 * ones  # Label smoothing
        
        # Fake images
        z = sample_noise(batch_size, latent_dim)
        fake_images = G(z).detach()
        fake_labels = zeros
        
        # Discriminator loss
        d_real = D(real_images)
        d_fake = D(fake_images)
        d_loss = BCE(d_real, real_labels) + BCE(d_fake, fake_labels)
        
        # Update discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ─────────────────────────────────────
        # 2. Train Generator
        # ─────────────────────────────────────
        
        # Generate new fakes
        z = sample_noise(batch_size, latent_dim)
        fake_images = G(z)
        
        # Generator wants D to output 1 (real)
        g_loss = BCE(D(fake_images), ones)
        
        # Update generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Latent dimension | 100 | Input noise vector size |
| Batch size | 64 | Adjust based on GPU memory |
| Learning rate (G) | 0.0002 | Adam optimizer |
| Learning rate (D) | 0.0002 | Adam optimizer |
| Beta1 (Adam) | 0.5 | Momentum term |
| Beta2 (Adam) | 0.999 | RMSprop term |
| Epochs | 200 | May need more for convergence |
| Label smoothing | 0.9 | Real labels only |
| LeakyReLU slope | 0.2 | Negative slope |

### Training Tips

1. **Balance D and G**: Monitor both losses; neither should dominate
2. **Gradient clipping**: Clip gradients if training becomes unstable
3. **Learning rate scheduling**: Reduce LR if loss plateaus
4. **Checkpoint frequently**: Save models every 10-20 epochs
5. **Visual inspection**: Generate samples periodically to assess quality

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         GENERATOR                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  z ∈ R^100                                                       │
│      │                                                           │
│      ▼                                                           │
│  ┌───────────┐                                                   │
│  │  Dense    │ 100 → 4096                                        │
│  │  Reshape  │ → (256, 4, 4)                                     │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ ConvT 4×4 │ (256, 4, 4) → (128, 8, 8)                        │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ ConvT 4×4 │ (128, 8, 8) → (64, 16, 16)                       │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ ConvT 4×4 │ (64, 16, 16) → (32, 32, 32)                      │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ ConvT 4×4 │ (32, 32, 32) → (1, 64, 64)                       │
│  └─────┬─────┘                                                   │
│        │ Tanh                                                    │
│        ▼                                                         │
│  Output: (1, 64, 64)                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       DISCRIMINATOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: (1, 64, 64)                                              │
│      │                                                           │
│      ▼                                                           │
│  ┌───────────┐                                                   │
│  │ Conv 4×4  │ (1, 64, 64) → (64, 32, 32)                       │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU                                               │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ Conv 4×4  │ (64, 32, 32) → (128, 16, 16)                     │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ Conv 4×4  │ (128, 16, 16) → (256, 8, 8)                      │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ Conv 4×4  │ (256, 8, 8) → (512, 4, 4)                        │
│  └─────┬─────┘                                                   │
│        │ LeakyReLU + BN                                          │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │ Conv 4×4  │ (512, 4, 4) → (1, 1, 1)                          │
│  └─────┬─────┘                                                   │
│        │ Sigmoid                                                 │
│        ▼                                                         │
│  Output: P(real) ∈ [0, 1]                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

1. Goodfellow et al., "Generative Adversarial Nets" (2014)
2. Radford et al., "Unsupervised Representation Learning with DCGANs" (2016)
3. Salimans et al., "Improved Techniques for Training GANs" (2016)
