#!/usr/bin/env python3
"""Generate samples for FID calculation.

Generates a specified number of images per class using a trained checkpoint.

Usage:
  python generate_fid_samples.py --ckpt /path/to/checkpoint.ckpt --out /path/to/output --num_classes 20 --samples_per_class 1000 --batch_size 50 --num_steps 100
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.lightning_model import LightningModel


def to_image(x):
    """Convert tensor to PIL Image.
    
    Args:
        x: tensor [C,H,W] in [-1,1]
    
    Returns:
        PIL Image
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, -1, 1)
    x = ((x + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(x)


def load_model(ckpt, device, config_path=None):
    """Load model from checkpoint.
    
    Args:
        ckpt: Path to checkpoint file
        device: Device to load model on
        config_path: Optional path to config YAML file
    
    Returns:
        Loaded LightningModel
    """
    print(f"Loading checkpoint: {ckpt}")
    model = None
    
    # Try direct Lightning load
    try:
        model: LightningModel = LightningModel.load_from_checkpoint(ckpt)
        print("Loaded LightningModel via load_from_checkpoint")
    except TypeError as e:
        print(f"Direct load failed ({e}). Trying to instantiate model from config...")
        # require config yaml path in CKPT directory or via env
        import yaml, importlib, glob, inspect
        
        # Try to find config next to checkpoint (search for YAML files) if not provided
        if config_path is None:
            ckpt_dir = os.path.dirname(ckpt)
            yaml_candidates = glob.glob(os.path.join(ckpt_dir, "*.yaml")) + glob.glob(os.path.join(ckpt_dir, "*.yml"))
            if yaml_candidates:
                # prefer filenames containing 'config' or 'cfg'
                for p in yaml_candidates:
                    if 'config' in os.path.basename(p) or 'cfg' in os.path.basename(p):
                        config_path = p
                        break
                if config_path is None:
                    config_path = yaml_candidates[0]
            if config_path is None:
                raise RuntimeError("Cannot find config next to checkpoint. Rerun script with --config /path/to/config.yaml")
        
        print(f"Using config: {config_path}")

        conf = yaml.safe_load(open(config_path))
        model_conf = conf.get('model', {})

        def resolve_python_path(val):
            """If val is a string like 'module.attr', import and return attr."""
            if not isinstance(val, str) or '.' not in val:
                return val
            module_name, attr_name = val.rsplit('.', 1)
            try:
                mod = importlib.import_module(module_name)
                attr = getattr(mod, attr_name)
                if inspect.isclass(attr):
                    try:
                        return attr()
                    except Exception:
                        return attr
                else:
                    return attr
            except Exception:
                return val

        def instantiate_node(node):
            if isinstance(node, dict) and 'class_path' in node:
                class_path = node['class_path']
                init_args = node.get('init_args', {}) or {}
                # recursively instantiate nested class args
                for k, v in list(init_args.items()):
                    if isinstance(v, dict) and 'class_path' in v:
                        init_args[k] = instantiate_node(v)
                    elif isinstance(v, str) and '.' in v:
                        init_args[k] = resolve_python_path(v)
                module_name, class_name = class_path.rsplit('.', 1)
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name)
                return cls(**init_args)
            # if it's a plain string referencing a python path, resolve
            if isinstance(node, str) and '.' in node:
                return resolve_python_path(node)
            return node

        vae = instantiate_node(model_conf['vae'])
        denoiser = instantiate_node(model_conf['denoiser'])
        conditioner = instantiate_node(model_conf['conditioner'])
        # Skip diffusion_trainer for inference (it contains perceptual losses we don't need)
        # diffusion_trainer = instantiate_node(model_conf['diffusion_trainer'])
        diffusion_sampler = instantiate_node(model_conf['diffusion_sampler'])

        # Create a minimal model without trainer (set to None for inference)
        model = LightningModel(vae, conditioner, denoiser, None, diffusion_sampler)

        # load checkpoint state dict
        ckpt_data = torch.load(ckpt, map_location='cpu')
        sd = ckpt_data.get('state_dict', ckpt_data)
        model.load_state_dict(sd, strict=False)
        print("Instantiated model from config and loaded state_dict (non-strict)")

    model.to(device)
    model.eval()
    return model


def generate_fid_samples(
    ckpt,
    out_dir,
    num_classes=20,
    samples_per_class=1000,
    batch_size=50,
    use_ema=True,
    device=None,
    cfg_scale=None,
    seed=42,
    num_steps=None,
    config_path=None
):
    """Generate samples for FID calculation.
    
    Args:
        ckpt: Path to checkpoint file
        out_dir: Output directory for generated images
        num_classes: Number of classes
        samples_per_class: Number of samples to generate per class
        batch_size: Batch size for generation
        use_ema: Whether to use EMA model
        device: Device to use
        cfg_scale: Classifier-free guidance scale (optional)
        seed: Random seed
        num_steps: Override number of sampling steps (NFE)
        config_path: Optional path to config YAML file
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load model
    model = load_model(ckpt, device, config_path)
    
    # Choose network (EMA or raw)
    net = model.ema_denoiser if use_ema else model.denoiser
    net.to(device)
    net.eval()
    
    sampler = model.diffusion_sampler
    
    # Override num_steps if specified
    if num_steps is not None:
        original_num_steps = getattr(sampler, 'num_steps', None)
        sampler.num_steps = num_steps
        print(f"Overriding sampler num_steps: {original_num_steps} -> {num_steps}")
    
    # Get latent shape
    C = model.denoiser.in_channels
    
    # Try to get spatial size from denoiser
    if hasattr(model.denoiser, 'input_size'):
        S = model.denoiser.input_size
    else:
        # For SD VAE, downsampling factor is typically 8
        # Infer from VAE model config if available
        print("Denoiser doesn't have input_size attribute, inferring from VAE...")
        vae_downsampling = 8  # Default for SD-VAE
        
        # Try to get it from VAE model config
        if hasattr(model.vae, 'model') and hasattr(model.vae.model, 'config'):
            vae_config = model.vae.model.config
            if hasattr(vae_config, 'block_out_channels'):
                # SD-VAE has downsampling factor = 2^(len(block_out_channels) - 1)
                vae_downsampling = 2 ** (len(vae_config.block_out_channels) - 1)
        
        # Assume 256x256 images (standard for ImageNet-like datasets)
        # You can override this with a parameter if needed
        image_size = 256
        S = image_size // vae_downsampling
        print(f"Inferred latent spatial size: {S} (image_size={image_size}, downsampling={vae_downsampling})")
    
    latent_shape = (C, S, S)
    
    print(f"Generating {num_classes * samples_per_class} images total ({samples_per_class} per class)")
    print(f"Latent shape: {latent_shape}")
    print(f"Batch size: {batch_size}")
    print(f"Using {'EMA' if use_ema else 'raw'} denoiser")
    if cfg_scale is not None:
        print(f"CFG scale: {cfg_scale}")
    
    # Generate samples for each class
    total_generated = 0
    for class_idx in range(num_classes):
        class_dir = os.path.join(out_dir, f"class_{class_idx:03d}")
        os.makedirs(class_dir, exist_ok=True)
        
        num_batches = (samples_per_class + batch_size - 1) // batch_size
        samples_generated = 0
        
        print(f"\nGenerating class {class_idx}/{num_classes-1}...")
        
        for batch_idx in tqdm(range(num_batches), desc=f"Class {class_idx}"):
            # Determine actual batch size (last batch might be smaller)
            current_batch_size = min(batch_size, samples_per_class - samples_generated)
            
            # Create labels for this batch
            y = torch.full((current_batch_size,), class_idx, dtype=torch.long, device=device)
            
            # Get condition tensors
            with torch.no_grad():
                condition, uncondition = model.conditioner(y)
            
            # Create initial noise
            noise = torch.randn((current_batch_size, *latent_shape), device=device)
            
            # Sample
            with torch.no_grad():
                if cfg_scale is not None and hasattr(sampler, 'cfg_scale'):
                    # Some samplers support CFG scale directly
                    old_cfg = getattr(sampler, 'cfg_scale', None)
                    sampler.cfg_scale = cfg_scale
                    latents = sampler(net, noise, condition, uncondition)
                    if old_cfg is not None:
                        sampler.cfg_scale = old_cfg
                else:
                    latents = sampler(net, noise, condition, uncondition)
            
            # Decode latents to images
            with torch.no_grad():
                images = model.vae.decode(latents)
            
            # Save images
            for i in range(current_batch_size):
                img = to_image(images[i])
                img_path = os.path.join(class_dir, f"{samples_generated:05d}.png")
                img.save(img_path)
                samples_generated += 1
                total_generated += 1
            
            # Free GPU memory
            del noise, condition, uncondition, latents, images
            if device == "cuda":
                torch.cuda.empty_cache()
        
        print(f"Class {class_idx}: Generated {samples_generated} images")
    
    print(f"\nTotal images generated: {total_generated}")
    print(f"Output directory: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples for FID calculation')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--out', type=str, default='fid_samples', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file (optional, will auto-detect if not specified)')
    parser.add_argument('--num_classes', type=int, default=20, help='Number of classes')
    parser.add_argument('--samples_per_class', type=int, default=1000, help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for generation')
    parser.add_argument('--no-ema', action='store_true', help='Use raw denoiser instead of EMA')
    parser.add_argument('--cfg_scale', type=float, default=None, help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--num_steps', type=int, default=None, help='Override number of sampling steps (NFE)')
    
    args = parser.parse_args()
    
    generate_fid_samples(
        ckpt=args.ckpt,
        out_dir=args.out,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        use_ema=not args.no_ema,
        device=args.device,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        num_steps=args.num_steps,
        config_path=args.config
    )
