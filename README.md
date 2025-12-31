# Vanilla GAN Signature Generation

## Project Overview

This project implements a **Vanilla Generative Adversarial Network (GAN)** for synthetic handwritten signature generation. The model learns to generate realistic-looking handwritten signatures from random noise vectors, enabling applications in data augmentation, signature synthesis research, and understanding of generative models.

## What is a Vanilla GAN?

A Vanilla GAN consists of two neural networks trained in an adversarial manner:

- **Generator (G)**: Takes random noise as input and generates synthetic signature images
- **Discriminator (D)**: Distinguishes between real signatures and generated (fake) signatures

The two networks compete against each other: the Generator tries to create increasingly realistic signatures to fool the Discriminator, while the Discriminator learns to better identify fake signatures. This adversarial training leads to the Generator producing high-quality synthetic signatures.

## Project Structure

```
vanilla_gan_signatures/
├── src/                    # Source code
│   ├── __init__.py
│   └── utils/              # Utility functions
│       └── __init__.py
├── data/                   # Data directory
│   └── signatures/
│       └── train/          # Training signature images
├── checkpoints/            # Saved model checkpoints
├── samples/                # Generated signature samples
├── figures/                # Training plots and visualizations
├── logs/                   # Training logs
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone or navigate to this project directory
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Place your signature images in the `data/signatures/train/` directory. Images should be:
- Grayscale or RGB format
- Consistent dimensions (will be resized during training)
- PNG or JPG format

### Training

To train the model, run the training script:

```bash
# Run training
python src/train_vanilla_gan_signatures.py --data_dir data/signatures/train --epochs 200
```

**Common Arguments:**

- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 64)
- `--image_size`: Target image size (default: 64)
- `--resume`: Resume from the latest checkpoint

### Generation (Web Interface)

The easiest way to generate signatures is using the included Streamlit web interface.

#### Option 1: Using the Batch Script (Windows)

Double-click `run_app.bat` in the root directory.

#### Option 2: Command Line

```bash
streamlit run src/app_vanilla_gan_signatures.py
```

The interface allows you to:

- Generate single or batch signatures
- Adjust the "noise" seed for reproducibility
- Download generated signatures as a ZIP file

## Features

- **Vanilla GAN Architecture**: Classic GAN implementation for educational and research purposes
- **Signature-Focused**: Optimized for handwritten signature generation
- **Evaluation Metrics**: Includes FID and LPIPS for quality assessment
- **Visualization Tools**: Training progress plots and sample generation
- **Web Interface**: Streamlit app for interactive signature generation

## Dependencies

- PyTorch 2.0+ for deep learning
- OpenCV and Pillow for image processing
- Matplotlib for visualization
- Streamlit and FastAPI for web interfaces
- LPIPS and PyTorch-FID for evaluation metrics

## License

This project is for educational and research purposes.

## Acknowledgments

Based on the original GAN paper by Goodfellow et al. (2014): "Generative Adversarial Nets"
