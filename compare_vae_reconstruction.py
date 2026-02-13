#!/usr/bin/env python3
"""Compare original images with VAE reconstructions.

This script loads images, encodes them with a VAE, decodes them back,
and creates a visual comparison showing:
- Column 1: Original Image (with optional zoom on a region)
- Column 2: VAE Reconstruction 
- Column 3: Difference/Error Map (amplified)

Usage:
    python compare_vae_reconstruction.py --image_dir /path/to/images --vae_path stabilityai/sd-vae-ft-mse
    python compare_vae_reconstruction.py --image_path single_image.jpg --vae_path stabilityai/sd-vae-ft-ema --zoom
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Tuple

from diffusers.models import AutoencoderKL
from torchvision.transforms import Normalize, Resize, CenterCrop
from torchvision.transforms.functional import to_tensor


def load_vae(vae_path: str, device: str = 'cuda') -> AutoencoderKL:
    """Load VAE model from path.
    
    Args:
        vae_path: Path to VAE weights (local or HuggingFace model ID)
        device: Device to load model on
        
    Returns:
        Loaded VAE model
    """
    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully")
    return vae


def load_and_preprocess_image(
    image_path: str, 
    resolution: int = 512
) -> Tuple[torch.Tensor, Image.Image]:
    """Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        resolution: Target resolution for processing
        
    Returns:
        Tuple of (preprocessed tensor [-1,1], original PIL image)
    """
    # Load image
    pil_image = Image.open(image_path).convert('RGB')
    original_image = pil_image.copy()
    
    # Resize and crop to target resolution
    resize = Resize(resolution)
    center_crop = CenterCrop(resolution)
    
    pil_image = resize(pil_image)
    pil_image = center_crop(pil_image)
    
    # Convert to tensor and normalize to [-1, 1]
    image_tensor = to_tensor(pil_image)  # [0, 1]
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    normalized_tensor = normalize(image_tensor)  # [-1, 1]
    
    return normalized_tensor, pil_image


def get_zoom_region(
    image: Union[torch.Tensor, np.ndarray],
    zoom_factor: float = 2.0,
    center: Tuple[int, int] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Extract a zoomed-in region from an image.
    
    Args:
        image: Image tensor [C, H, W] or array [H, W, C]
        zoom_factor: How much to zoom (2.0 means 2x zoom)
        center: (y, x) center point for zoom, None for image center
        
    Returns:
        Zoomed image region
    """
    if isinstance(image, torch.Tensor):
        c, h, w = image.shape
        is_tensor = True
    else:
        h, w, c = image.shape
        is_tensor = False
    
    # Calculate crop size
    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)
    
    # Calculate crop coordinates
    if center is None:
        # Default to center of image
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center
    
    y1 = max(0, cy - crop_h // 2)
    y2 = min(h, y1 + crop_h)
    x1 = max(0, cx - crop_w // 2)
    x2 = min(w, x1 + crop_w)
    
    # Extract region
    if is_tensor:
        region = image[:, y1:y2, x1:x2]
    else:
        region = image[y1:y2, x1:x2, :]
    
    return region


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization.
    
    Args:
        tensor: Image tensor [C, H, W] in [-1, 1]
        
    Returns:
        Numpy array [H, W, C] in [0, 255]
    """
    # Move to CPU and convert to numpy
    array = tensor.detach().cpu().numpy()
    
    # Transpose from [C, H, W] to [H, W, C]
    array = np.transpose(array, (1, 2, 0))
    
    # Denormalize from [-1, 1] to [0, 1]
    array = (array + 1) / 2
    
    # Clip and convert to [0, 255]
    array = np.clip(array, 0, 1)
    array = (array * 255).astype(np.uint8)
    
    return array


@torch.no_grad()
def encode_decode_image(
    vae: AutoencoderKL,
    image_tensor: torch.Tensor,
    device: str = 'cuda'
) -> torch.Tensor:
    """Encode and decode an image through the VAE.
    
    Args:
        vae: VAE model
        image_tensor: Image tensor [C, H, W] in [-1, 1]
        device: Device to use
        
    Returns:
        Reconstructed image tensor [C, H, W] in [-1, 1]
    """
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    # Encode
    latent_dist = vae.encode(image_batch).latent_dist
    latent = latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    
    # Decode
    latent = latent / vae.config.scaling_factor
    reconstructed = vae.decode(latent).sample
    
    # Remove batch dimension
    reconstructed = reconstructed.squeeze(0)
    
    return reconstructed


def create_comparison_figure(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    zoom: bool = False,
    zoom_factor: float = 2.0,
    zoom_center: Tuple[int, int] = None,
    diff_amplification: float = 5.0,
    save_path: str = None,
    title: str = None
):
    """Create a 3-column comparison figure.
    
    Args:
        original: Original image tensor [C, H, W] in [-1, 1]
        reconstructed: Reconstructed image tensor [C, H, W] in [-1, 1]
        zoom: Whether to zoom in on the original image
        zoom_factor: Zoom factor for original image
        zoom_center: Center point (y, x) for zoom
        diff_amplification: Factor to amplify difference map
        save_path: Path to save figure
        title: Figure title
    """
    # Convert tensors to numpy arrays
    original_np = tensor_to_numpy(original)
    reconstructed_np = tensor_to_numpy(reconstructed)
    
    # Apply zoom to original if requested
    if zoom:
        original_np = get_zoom_region(original_np, zoom_factor, zoom_center)
        # Also zoom reconstruction to match
        reconstructed_tensor_zoomed = get_zoom_region(reconstructed, zoom_factor, zoom_center)
        reconstructed_np = tensor_to_numpy(reconstructed_tensor_zoomed)
    
    # Calculate difference map
    diff_map = original_np.astype(float) - reconstructed_np.astype(float)
    
    # Amplify difference
    diff_map_amplified = diff_map * diff_amplification
    
    # Convert to grayscale for better visualization
    diff_gray = np.mean(np.abs(diff_map_amplified), axis=2)
    
    # Clip to valid range
    diff_gray = np.clip(diff_gray, 0, 255).astype(np.uint8)
    
    # Calculate MSE and PSNR
    mse = np.mean((original_np.astype(float) - reconstructed_np.astype(float)) ** 2)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Column 1: Original
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image' + (' (Zoomed)' if zoom else ''), fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Column 2: Reconstruction
    axes[1].imshow(reconstructed_np)
    axes[1].set_title('VAE Reconstruction', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Column 3: Difference map
    im = axes[2].imshow(diff_gray, cmap='hot', vmin=0, vmax=255)
    axes[2].set_title(f'Error Map (×{diff_amplification})', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar for difference map
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add metrics as figure title
    if title is None:
        title = f'VAE Reconstruction Comparison (MSE: {mse:.2f}, PSNR: {psnr:.2f} dB)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def find_images(image_dir: str, extensions: List[str] = None) -> List[str]:
    """Find all images in a directory.
    
    Args:
        image_dir: Directory to search
        extensions: List of valid extensions (default: common image formats)
        
    Returns:
        List of image paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_dir).glob(f'**/*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'**/*{ext.upper()}'))
    
    return sorted([str(p) for p in image_paths])


def main():
    parser = argparse.ArgumentParser(
        description='Compare original images with VAE reconstructions'
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image_path', 
        type=str,
        help='Path to a single image file'
    )
    input_group.add_argument(
        '--image_dir',
        type=str,
        help='Path to directory containing images'
    )
    
    # VAE options
    parser.add_argument(
        '--vae_path',
        type=str,
        default='stabilityai/sd-vae-ft-mse',
        help='Path to VAE weights (local path or HuggingFace model ID)'
    )
    
    # Processing options
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Resolution to process images at'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    
    # Visualization options
    parser.add_argument(
        '--zoom',
        action='store_true',
        help='Zoom in on a region of the original image'
    )
    parser.add_argument(
        '--zoom_factor',
        type=float,
        default=2.0,
        help='Zoom factor (e.g., 2.0 for 2x zoom)'
    )
    parser.add_argument(
        '--zoom_center',
        type=int,
        nargs=2,
        metavar=('Y', 'X'),
        help='Center point for zoom (y, x). Default: image center'
    )
    parser.add_argument(
        '--diff_amplification',
        type=float,
        default=5.0,
        help='Factor to amplify the difference map for better visibility'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vae_comparison_output',
        help='Directory to save comparison figures'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to process (for directories)'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Do not display figures interactively (only save)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VAE
    vae = load_vae(args.vae_path, args.device)
    
    # Get list of images to process
    if args.image_path:
        image_paths = [args.image_path]
    else:
        image_paths = find_images(args.image_dir)
        if args.max_images:
            image_paths = image_paths[:args.max_images]
    
    print(f"\nFound {len(image_paths)} image(s) to process")
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing [{i+1}/{len(image_paths)}]: {image_path}")
        
        try:
            # Load and preprocess image
            image_tensor, pil_image = load_and_preprocess_image(
                image_path, 
                args.resolution
            )
            
            # Encode and decode through VAE
            reconstructed = encode_decode_image(vae, image_tensor, args.device)
            
            # Create comparison figure
            image_name = Path(image_path).stem
            save_path = os.path.join(
                args.output_dir, 
                f'{image_name}_comparison.png'
            )
            
            # Prepare title
            title = f'VAE Reconstruction: {image_name}'
            
            # Set matplotlib to non-interactive mode if no_display
            if args.no_display:
                plt.ioff()
            
            # Create figure
            create_comparison_figure(
                original=image_tensor,
                reconstructed=reconstructed,
                zoom=args.zoom,
                zoom_factor=args.zoom_factor,
                zoom_center=tuple(args.zoom_center) if args.zoom_center else None,
                diff_amplification=args.diff_amplification,
                save_path=save_path,
                title=title
            )
            
            # Close figure to free memory
            if args.no_display or len(image_paths) > 1:
                plt.close()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\n✓ Processing complete! Figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
