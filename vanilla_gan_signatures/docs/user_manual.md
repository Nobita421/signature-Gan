# User Manual - Vanilla GAN Signature Generator

A comprehensive guide for using the Vanilla GAN Signature Generation system.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Preprocessing Data](#2-preprocessing-data)
3. [Training the GAN](#3-training-the-gan)
4. [Generating Signatures](#4-generating-signatures)
5. [Using the Web UI](#5-using-the-web-ui)
6. [Using the API](#6-using-the-api)
7. [Evaluating Results](#7-evaluating-results)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM

### Setup Steps

```bash
# Clone or navigate to the project directory
cd vanilla_gan_signatures

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Dependencies Overview

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| torchvision | Image transformations |
| numpy | Numerical operations |
| Pillow | Image processing |
| matplotlib | Visualization |
| flask | Web API server |
| gradio | Web UI interface |
| tqdm | Progress bars |

---

## 2. Preprocessing Data

### Supported Formats

- PNG, JPG, JPEG, BMP, TIFF
- Grayscale or RGB (auto-converted)

### Data Directory Structure

```
data/
└── signatures/
    └── train/
        ├── signature_001.png
        ├── signature_002.png
        └── ...
```

### Running Preprocessing

```bash
# Basic preprocessing
python src/preprocess_signatures.py --input_dir data/raw --output_dir data/signatures/train

# With custom options
python src/preprocess_signatures.py \
    --input_dir data/raw \
    --output_dir data/signatures/train \
    --img_size 64 \
    --threshold 128 \
    --invert
```

### Preprocessing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | Required | Source image directory |
| `--output_dir` | Required | Processed output directory |
| `--img_size` | 64 | Output image size (square) |
| `--threshold` | 128 | Binarization threshold |
| `--invert` | False | Invert colors |
| `--augment` | False | Apply data augmentation |

### Data Augmentation

Enable augmentation to increase dataset size:

```bash
python src/preprocess_signatures.py --augment --aug_factor 5
```

Augmentation techniques:
- Random rotation (±10°)
- Random scaling (0.9-1.1)
- Elastic deformation
- Noise injection

---

## 3. Training the GAN

### Quick Start

```bash
# Train with default settings
python src/train_vanilla_gan_signatures.py
```

### Custom Training

```bash
python src/train_vanilla_gan_signatures.py \
    --data_dir data/signatures/train \
    --epochs 200 \
    --batch_size 64 \
    --latent_dim 100 \
    --lr 0.0002 \
    --checkpoint_dir checkpoints \
    --sample_interval 10
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_dir` | data/signatures/train | Training data path |
| `--epochs` | 200 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--latent_dim` | 100 | Latent vector dimension |
| `--lr` | 0.0002 | Learning rate |
| `--beta1` | 0.5 | Adam beta1 parameter |
| `--checkpoint_dir` | checkpoints | Model save directory |
| `--sample_interval` | 10 | Epochs between samples |
| `--save_interval` | 20 | Epochs between checkpoints |
| `--resume` | None | Resume from checkpoint |

### Monitoring Training

Training outputs:
- **Console**: Loss values every batch
- **Logs**: Detailed logs in `logs/` directory
- **Samples**: Generated images in `samples/` directory
- **Checkpoints**: Model weights in `checkpoints/` directory

```bash
# View training logs
tail -f logs/training.log

# Monitor GPU usage
nvidia-smi -l 1
```

### Resume Training

```bash
python src/train_vanilla_gan_signatures.py --resume checkpoints/vanilla_gan_epoch_100.pth
```

---

## 4. Generating Signatures

### Command Line Generation

```bash
# Generate single signature
python src/generate_signatures.py --checkpoint checkpoints/vanilla_gan_epoch_200.pth

# Generate multiple signatures
python src/generate_signatures.py \
    --checkpoint checkpoints/vanilla_gan_epoch_200.pth \
    --num_samples 100 \
    --output_dir generated_signatures

# Generate with specific seed (reproducible)
python src/generate_signatures.py \
    --checkpoint checkpoints/vanilla_gan_epoch_200.pth \
    --seed 42 \
    --num_samples 10
```

### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | Required | Model checkpoint path |
| `--num_samples` | 1 | Number of signatures to generate |
| `--output_dir` | generated | Output directory |
| `--seed` | None | Random seed for reproducibility |
| `--latent_dim` | 100 | Must match training config |
| `--format` | png | Output format (png/jpg) |

### Latent Space Exploration

```bash
# Interpolate between two random points
python src/generate_signatures.py \
    --checkpoint checkpoints/vanilla_gan_epoch_200.pth \
    --interpolate \
    --num_steps 10
```

---

## 5. Using the Web UI

### Starting the Web Interface

```bash
python src/app_vanilla_gan_signatures.py
```

The UI will be available at: `http://localhost:7860`

### Web UI Features

1. **Generate Tab**
   - Slider for number of samples (1-16)
   - Random seed input (optional)
   - Generate button
   - Gallery view of results
   - Download individual/all images

2. **Settings Tab**
   - Model selection dropdown
   - Output format selection
   - Image size display

3. **Gallery Tab**
   - Browse previously generated signatures
   - Filter by date/quality
   - Batch download

### UI Usage Tips

- Use seeds for reproducible generation
- Generate in batches of 4-8 for quick preview
- Check the "High Quality" option for final outputs

---

## 6. Using the API

### Starting the API Server

```bash
python src/api_vanilla_gan_signatures.py --port 5000
```

API available at: `http://localhost:5000`

### API Endpoints

#### Health Check
```bash
GET /health

Response: {"status": "healthy", "model_loaded": true}
```

#### Generate Signatures
```bash
POST /generate
Content-Type: application/json

{
    "num_samples": 5,
    "seed": 42,
    "return_base64": true
}

Response:
{
    "success": true,
    "signatures": ["base64_image_1", "base64_image_2", ...],
    "seed_used": 42
}
```

#### Generate and Save
```bash
POST /generate/save
Content-Type: application/json

{
    "num_samples": 10,
    "output_dir": "api_generated"
}

Response:
{
    "success": true,
    "files": ["api_generated/sig_001.png", ...]
}
```

### Python Client Example

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Generate signatures
response = requests.post(
    "http://localhost:5000/generate",
    json={"num_samples": 5, "return_base64": True}
)

data = response.json()

# Decode and display
for i, b64_img in enumerate(data["signatures"]):
    img_data = base64.b64decode(b64_img)
    img = Image.open(BytesIO(img_data))
    img.save(f"signature_{i}.png")
```

### cURL Examples

```bash
# Generate 5 signatures
curl -X POST http://localhost:5000/generate \
    -H "Content-Type: application/json" \
    -d '{"num_samples": 5}'

# Generate with seed
curl -X POST http://localhost:5000/generate \
    -H "Content-Type: application/json" \
    -d '{"num_samples": 1, "seed": 12345}'
```

---

## 7. Evaluating Results

### Running Evaluation

```bash
python src/evaluate_vanilla_gan_signatures.py \
    --checkpoint checkpoints/vanilla_gan_epoch_200.pth \
    --real_dir data/signatures/train \
    --num_samples 1000
```

### Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| FID Score | Fréchet Inception Distance | < 50 |
| IS Score | Inception Score | > 2.0 |
| SSIM | Structural Similarity | > 0.3 |
| Diversity | Intra-class variance | > 0.1 |

### Evaluation Output

```
=====================================
EVALUATION RESULTS
=====================================
Model: vanilla_gan_epoch_200.pth
Samples: 1000

FID Score: 45.32
Inception Score: 2.45 ± 0.15
Mean SSIM: 0.42
Diversity Score: 0.23

Quality Assessment: GOOD
=====================================
```

### Visual Evaluation

```bash
# Generate comparison grid
python src/evaluate_vanilla_gan_signatures.py \
    --checkpoint checkpoints/vanilla_gan_epoch_200.pth \
    --visualize \
    --output figures/evaluation_grid.png
```

### Signature Verification Test

```bash
# Test against signature verifier
python src/signature_verifier_eval.py \
    --generated_dir generated_signatures \
    --real_dir data/signatures/train
```

---

## 8. Troubleshooting

### Common Issues

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size
```bash
python src/train_vanilla_gan_signatures.py --batch_size 32
```

#### Mode Collapse
**Symptoms**: All generated signatures look identical
**Solutions**:
- Reduce learning rate: `--lr 0.0001`
- Add noise to discriminator inputs
- Use label smoothing
- Train longer

#### Training Instability
**Symptoms**: Loss values oscillate wildly or go to NaN
**Solutions**:
- Check data preprocessing (values in [-1, 1])
- Reduce learning rate
- Clip gradients: `--grad_clip 1.0`
- Use spectral normalization

#### Poor Quality Generations
**Solutions**:
- Train for more epochs
- Increase dataset size
- Check data quality
- Adjust hyperparameters

### Getting Help

1. Check logs in `logs/` directory
2. Review training curves in TensorBoard
3. Open an issue on GitHub with:
   - Error message
   - Command used
   - System specifications

### Performance Tips

| Tip | Impact |
|-----|--------|
| Use GPU | 10-50x faster training |
| Increase batch size | Better gradients |
| Use SSD storage | Faster data loading |
| Pin memory | Faster CPU-GPU transfer |
| Use mixed precision | 2x faster, less memory |

---

## Quick Reference

### Essential Commands

```bash
# Preprocess data
python src/preprocess_signatures.py --input_dir raw --output_dir processed

# Train model
python src/train_vanilla_gan_signatures.py --epochs 200

# Generate signatures
python src/generate_signatures.py --checkpoint checkpoints/best.pth --num_samples 10

# Start Web UI
python src/app_vanilla_gan_signatures.py

# Start API
python src/api_vanilla_gan_signatures.py

# Evaluate model
python src/evaluate_vanilla_gan_signatures.py --checkpoint checkpoints/best.pth
```

---

## Support

For additional help:
- Check the [README.md](../README.md)
- Review [architecture.md](architecture.md) for technical details
- See [future_work.md](future_work.md) for planned features
