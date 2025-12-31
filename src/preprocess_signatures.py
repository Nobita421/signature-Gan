"""
Signature Image Preprocessing Module for Vanilla GAN.

This module provides functions for preprocessing signature images including:
- Grayscale conversion
- Binarization and normalization
- Contour-based cropping to remove margins
- Resizing to fixed dimensions
- Pixel value normalization to [-1, 1] range
- Noise removal and quality filtering

Author: Vanilla GAN Signatures Project
Date: December 2024
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Union
import logging

import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_TARGET_SIZE: Tuple[int, int] = (64, 64)
DEFAULT_MARGIN: int = 5
DEFAULT_BINARY_THRESHOLD: int = 127
MIN_CONTOUR_AREA_RATIO: float = 0.001  # Minimum contour area as ratio of image
MAX_NOISE_RATIO: float = 0.95  # Maximum white pixel ratio (too empty)
MIN_INK_RATIO: float = 0.01  # Minimum ink (dark) pixel ratio


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        image: Input image as numpy array (BGR or RGB format).
        
    Returns:
        Grayscale image as numpy array.
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA - convert to RGB first
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


def binarize_image(
    image: np.ndarray,
    threshold: int = DEFAULT_BINARY_THRESHOLD,
    adaptive: bool = True
) -> np.ndarray:
    """
    Binarize a grayscale image.
    
    Args:
        image: Grayscale image as numpy array.
        threshold: Threshold value for simple binarization (0-255).
        adaptive: If True, use adaptive thresholding for better results.
        
    Returns:
        Binarized image as numpy array (0 or 255 values).
    """
    if adaptive:
        # Adaptive Gaussian thresholding works better for signatures
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
    else:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensity using histogram equalization.
    
    Args:
        image: Grayscale image as numpy array.
        
    Returns:
        Intensity-normalized image.
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(image)
    return normalized


def find_signature_bbox(
    image: np.ndarray,
    margin: int = DEFAULT_MARGIN
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bounding box around the signature using contour detection.
    
    Args:
        image: Grayscale or binary image as numpy array.
        margin: Padding to add around the detected bounding box.
        
    Returns:
        Tuple of (x, y, width, height) or None if no signature found.
    """
    # Ensure binary image (invert so signature is white)
    if image.max() > 1:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        binary = (image < 0.5).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Filter small contours (noise)
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * MIN_CONTOUR_AREA_RATIO
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        return None
    
    # Get bounding box that encompasses all valid contours
    all_points = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add margin
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    return (x, y, w, h)


def crop_signature(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    margin: int = DEFAULT_MARGIN
) -> np.ndarray:
    """
    Crop the image around the signature.
    
    Args:
        image: Input image as numpy array.
        bbox: Optional pre-computed bounding box (x, y, w, h).
        margin: Margin to add if computing bbox.
        
    Returns:
        Cropped image containing the signature.
    """
    if bbox is None:
        bbox = find_signature_bbox(image, margin)
    
    if bbox is None:
        # Return original if no signature detected
        logger.warning("No signature detected, returning original image")
        return image
    
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    pad_color: int = 255
) -> np.ndarray:
    """
    Resize image to target size while preserving aspect ratio.
    
    The image is scaled to fit within the target dimensions and then
    centered with padding.
    
    Args:
        image: Input image as numpy array.
        target_size: Target (width, height) tuple.
        pad_color: Color value for padding (255 for white, 0 for black).
        
    Returns:
        Resized and padded image.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with padding color
    if len(image.shape) == 3:
        canvas = np.full((target_h, target_w, image.shape[2]), pad_color, dtype=np.uint8)
    else:
        canvas = np.full((target_h, target_w), pad_color, dtype=np.uint8)
    
    # Center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def normalize_pixels(
    image: np.ndarray,
    output_range: Tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """
    Normalize pixel values to specified range.
    
    Args:
        image: Input image as numpy array (0-255 range).
        output_range: Target (min, max) range for normalization.
        
    Returns:
        Normalized image as float32 array.
    """
    min_val, max_val = output_range
    
    # Convert to float and normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Scale to output range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def denormalize_pixels(
    image: np.ndarray,
    input_range: Tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """
    Denormalize pixel values from [-1, 1] back to [0, 255].
    
    Args:
        image: Normalized image as numpy array.
        input_range: Current (min, max) range of the image.
        
    Returns:
        Denormalized image as uint8 array (0-255 range).
    """
    min_val, max_val = input_range
    
    # Scale to [0, 1]
    denormalized = (image - min_val) / (max_val - min_val)
    
    # Scale to [0, 255]
    denormalized = (denormalized * 255).clip(0, 255).astype(np.uint8)
    
    return denormalized


def remove_noise(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Remove noise from signature image using morphological operations.
    
    Args:
        image: Grayscale image as numpy array.
        kernel_size: Size of the morphological kernel.
        
    Returns:
        Denoised image.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    
    return opened


def is_valid_signature(
    image: np.ndarray,
    min_ink_ratio: float = MIN_INK_RATIO,
    max_noise_ratio: float = MAX_NOISE_RATIO
) -> bool:
    """
    Check if the image contains a valid signature.
    
    Filters out:
    - Nearly empty images (too much white space)
    - Extremely noisy images
    - Images with too little ink
    
    Args:
        image: Grayscale image as numpy array (0-255).
        min_ink_ratio: Minimum ratio of dark pixels required.
        max_noise_ratio: Maximum ratio of white pixels allowed.
        
    Returns:
        True if the image appears to contain a valid signature.
    """
    # Binarize for analysis
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate pixel ratios
    total_pixels = image.size
    white_pixels = np.sum(binary == 255)
    dark_pixels = total_pixels - white_pixels
    
    white_ratio = white_pixels / total_pixels
    ink_ratio = dark_pixels / total_pixels
    
    # Check validity
    if white_ratio > max_noise_ratio:
        logger.debug(f"Image too empty: {white_ratio:.2%} white")
        return False
    
    if ink_ratio < min_ink_ratio:
        logger.debug(f"Too little ink: {ink_ratio:.2%} ink")
        return False
    
    return True


def center_signature(image: np.ndarray) -> np.ndarray:
    """
    Center the signature within the image using center of mass.
    
    Args:
        image: Grayscale image with signature.
        
    Returns:
        Image with centered signature.
    """
    # Invert so signature is white for moment calculation
    inverted = 255 - image
    
    # Calculate moments
    moments = cv2.moments(inverted)
    
    if moments['m00'] == 0:
        return image
    
    # Calculate center of mass
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    # Calculate shift needed to center
    h, w = image.shape[:2]
    shift_x = w // 2 - cx
    shift_y = h // 2 - cy
    
    # Create translation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Apply translation
    centered = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        borderValue=255  # White background
    )
    
    return centered


def preprocess_single_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    binarize: bool = False,
    normalize: bool = True,
    remove_margin: bool = True,
    center: bool = True,
    denoise: bool = True,
    validate: bool = True
) -> Optional[np.ndarray]:
    """
    Preprocess a single signature image.
    
    Complete preprocessing pipeline:
    1. Load and convert to grayscale
    2. Optionally denoise
    3. Optionally remove margins (crop)
    4. Resize with aspect ratio preservation
    5. Optionally center the signature
    6. Optionally binarize
    7. Optionally normalize to [-1, 1]
    
    Args:
        image_path: Path to the input image.
        target_size: Target (width, height) for output image.
        binarize: If True, output binary image.
        normalize: If True, normalize pixels to [-1, 1].
        remove_margin: If True, crop to signature bounding box.
        center: If True, center the signature.
        denoise: If True, apply noise removal.
        validate: If True, reject invalid signatures.
        
    Returns:
        Preprocessed image as numpy array, or None if invalid.
    """
    # Load image
    image = cv2.imread(str(image_path))
    
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    
    # Denoise
    if denoise:
        gray = remove_noise(gray)
    
    # Validate
    if validate and not is_valid_signature(gray):
        logger.warning(f"Invalid signature detected: {image_path}")
        return None
    
    # Crop to signature region
    if remove_margin:
        gray = crop_signature(gray)
    
    # Resize with aspect ratio preservation
    resized = resize_with_aspect_ratio(gray, target_size)
    
    # Center the signature
    if center:
        resized = center_signature(resized)
    
    # Binarize if requested
    if binarize:
        resized = binarize_image(resized)
    else:
        # Apply intensity normalization
        resized = normalize_intensity(resized)
    
    # Normalize pixel values
    if normalize:
        resized = normalize_pixels(resized, (-1.0, 1.0))
    
    return resized


def preprocess_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    binarize: bool = False,
    remove_margin: bool = True,
    center: bool = True,
    denoise: bool = True,
    validate: bool = True,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
) -> Tuple[int, int, List[str]]:
    """
    Preprocess a batch of signature images.
    
    Args:
        input_dir: Directory containing input images.
        output_dir: Directory to save preprocessed images.
        target_size: Target (width, height) for output images.
        binarize: If True, output binary images.
        remove_margin: If True, crop to signature bounding box.
        center: If True, center the signatures.
        denoise: If True, apply noise removal.
        validate: If True, reject invalid signatures.
        extensions: Tuple of valid image file extensions.
        
    Returns:
        Tuple of (success_count, fail_count, list_of_failed_files).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        try:
            # Preprocess (don't normalize for saving)
            result = preprocess_single_image(
                img_file,
                target_size=target_size,
                binarize=binarize,
                normalize=False,  # Save as uint8
                remove_margin=remove_margin,
                center=center,
                denoise=denoise,
                validate=validate
            )
            
            if result is not None:
                # Save preprocessed image
                output_file = output_path / f"{img_file.stem}_processed.png"
                cv2.imwrite(str(output_file), result)
                success_count += 1
                logger.debug(f"Processed: {img_file.name}")
            else:
                fail_count += 1
                failed_files.append(str(img_file))
                
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            fail_count += 1
            failed_files.append(str(img_file))
    
    logger.info(f"Preprocessing complete: {success_count} succeeded, {fail_count} failed")
    
    return success_count, fail_count, failed_files


def save_preprocessed_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    is_normalized: bool = True
) -> bool:
    """
    Save a preprocessed image to disk.
    
    Args:
        image: Preprocessed image as numpy array.
        output_path: Path to save the image.
        is_normalized: If True, denormalize before saving.
        
    Returns:
        True if save was successful.
    """
    try:
        if is_normalized:
            image = denormalize_pixels(image)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        return True
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


def create_preprocessing_config(
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    binarize: bool = False,
    remove_margin: bool = True,
    center: bool = True,
    denoise: bool = True,
    validate: bool = True
) -> dict:
    """
    Create a preprocessing configuration dictionary.
    
    Args:
        target_size: Target image dimensions.
        binarize: Whether to binarize images.
        remove_margin: Whether to crop margins.
        center: Whether to center signatures.
        denoise: Whether to apply denoising.
        validate: Whether to validate signatures.
        
    Returns:
        Configuration dictionary.
    """
    return {
        'target_size': target_size,
        'binarize': binarize,
        'remove_margin': remove_margin,
        'center': center,
        'denoise': denoise,
        'validate': validate
    }


def main():
    """CLI interface for batch preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess signature images for GAN training'
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing input signature images'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save preprocessed images'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=64,
        choices=[64, 128],
        help='Target image size (64 or 128, default: 64)'
    )
    
    parser.add_argument(
        '--binarize',
        action='store_true',
        help='Binarize images (black and white only)'
    )
    
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='Disable margin cropping'
    )
    
    parser.add_argument(
        '--no-center',
        action='store_true',
        help='Disable signature centering'
    )
    
    parser.add_argument(
        '--no-denoise',
        action='store_true',
        help='Disable noise removal'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Disable signature validation (include all images)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run batch preprocessing
    target_size = (args.size, args.size)
    
    success, fail, failed_files = preprocess_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=target_size,
        binarize=args.binarize,
        remove_margin=not args.no_crop,
        center=not args.no_center,
        denoise=not args.no_denoise,
        validate=not args.no_validate
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Preprocessing Complete")
    print(f"{'='*50}")
    print(f"Successfully processed: {success}")
    print(f"Failed: {fail}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


if __name__ == '__main__':
    main()
