#!/usr/bin/env python3
"""
Complete Qualitative Analysis Suite for PixelGen.
Consolidates:
1. Attention/Activation Visualization (Structural Coherence)
2. Texture Fidelity (VAE Reconstruction)
3. Nearest Neighbor Analysis (Novelty/Overfitting)

Usage:
  python sample_diagnostic_sweep.py attention --ckpt <path> --config <path>
  # (Requires scikit-learn for nn)
  python sample_diagnostic_sweep.py texture --image_folder <path> --out_dir <path>
  python sample_diagnostic_sweep.py nn --gen_dir <path> --train_dir <path> --out_dir <path>
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
import torchvision.transforms as T
from torchvision.utils import save_image

# -----------------------------------------------------------------------------
# 1. Attention / Feature Visualization
# -----------------------------------------------------------------------------
def analyze_attention(args):
    print("Starting Attention/Feature Visualization...")
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Minimal import logic for model loading
    # We assume 'src' is in PYTHONPATH or CWD
    try:
        from src.lightning_model import LightningModel
        # Basic config loading helper if needed, or rely on LightningModel
    except ImportError:
        import sys
        sys.path.append(os.getcwd())
        from src.lightning_model import LightningModel

    # Load Model (Simplified for this snippet - assuming standard PL loading)
    # If the user has a specific loading function in generate_fid_samples, we reuse it partially
    print(f"Loading checkpoint: {args.ckpt}")
    # This might need adjustment depending on how 'LightningModel' is instantiated
    # We'll try to load from checkpoint directly
    try:
        model = LightningModel.load_from_checkpoint(args.ckpt, map_location=device)
    except Exception as e:
        print(f"Failed to load with LightningModel.load_from_checkpoint: {e}")
        # fallback: try manual load if config provided
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        model = LightningModel(cfg)
        sd = torch.load(args.ckpt, map_location=device)
        if "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd)
    
    model.to(device)
    model.eval()

    activations = {}
    def get_activation_hook(name):
        def hook(module, input, output):
            # Capture output tensor: (B, L, C)
            # Some implementations might return tuple
            if isinstance(output, tuple): 
                out = output[0]
            else:
                out = output
            # We want to detach and move to CPU immediately to save VRAM
            activations[name] = out.detach().cpu()
        return hook

    # Register hooks on DiT blocks
    # Structure of 'model.model' depends on implementation (e.g. denoiser or diffusion_model)
    denoiser = None
    if hasattr(model, 'model'): denoiser = model.model
    if hasattr(model, 'diffusion_model'): denoiser = model.diffusion_model
    
    if denoiser and hasattr(denoiser, 'blocks'):
        depth = len(denoiser.blocks)
        # We perform analysis at 4 depths
        indices = [depth // 4, depth // 2, 3 * depth // 4, depth - 1]
        for idx in indices:
            denoiser.blocks[idx].register_forward_hook(get_activation_hook(f"block_{idx:02d}"))
            print(f"Hook registered at block {idx}")
    else:
        print("Warning: Could not find 'blocks' in model to attach hooks. Skipping Hooks.")

    # Generate one sample
    # Resolution/Shape handling
    # Assuming standard DiT shape (1, C, H, W) -> patchified
    
    # We need to run p_sample_loop or just a forward pass with noise
    # Running a full generation might be slow, so let's just do a single forward pass 
    # at a meaningful timestep to see activations.
    
    print("Running single forward pass validation...")
    
    # Determine input size
    S = 32 # Default latent size (256px / 8)
    if hasattr(model, 'image_size'): 
        # model.image_size might be pixel size (256), latent is /8
        S = model.image_size // 8 
    C = 4 # Latent channels
    if hasattr(model, 'in_channels'): C = model.in_channels

    z = torch.randn(1, C, S, S, device=device)
    t = torch.tensor([500], device=device).long() # Mid-step
    y = torch.tensor([0], device=device).long()   # Class 0 if applicable
    
    # Check if model takes y
    with torch.no_grad():
        # Adjust forward signature based on model
        try:
            model.model(z, t, y=y)
        except:
             # Try without y
            try:
                 model.model(z, t)
            except:
                 # Try with cfg dictionary style if relevant
                 pass

    # Visualize
    if not activations:
        print("No activations captured. Check model structure.")
        return

    for name, act in activations.items():
        # act: (1, L, C) or (1, C, H, W) depending on architecture
        # If (1, C, H, W):
        if act.dim() == 4:
            # Spatial - calculate mean across channels
            # (1, C, H, W) -> (H, W)
            mean_act = act[0].mean(dim=0).numpy()
            
            plt.figure()
            plt.imshow(mean_act, cmap='viridis')
            plt.title(f"Mean Activation {name}")
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(args.out_dir, f"{name}_activation.png"))
            plt.close()
            
        # If (1, L, C):
        elif act.dim() == 3:
            L = act.shape[1]
            H = int(L**0.5)
            # (1, L, C) -> (C, H, W)
            feat = act[0].permute(1, 0).view(-1, H, H)
            
            # Mean Activation
            mean_act = feat.mean(dim=0).numpy()
            plt.figure()
            plt.imshow(mean_act, cmap='viridis')
            plt.title(f"Mean Activation {name}")
            plt.colorbar()
            plt.axis('off')
            plt.savefig(os.path.join(args.out_dir, f"{name}_activation.png"))
            plt.close()
            
            # Self-Similarity (Cosine sim of patch tokens)
            # Sample a subset of query patches (e.g. center patch)
            # Center index
            center_idx = (H * H) // 2
            
            # (L, C)
            flat = act[0] 
            # Normalize
            flat = torch.nn.functional.normalize(flat, p=2, dim=1)
            
            # Compute sim map for the center patch
            query = flat[center_idx].unsqueeze(0) # (1, C)
            sim = torch.mm(query, flat.t())       # (1, L)
            sim_map = sim.view(H, H).numpy()
            
            plt.figure(figsize=(6, 6))
            plt.imshow(sim_map, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title(f"Self-Sim (Center Patch) {name}")
            plt.axis('off')
            plt.savefig(os.path.join(args.out_dir, f"{name}_similarity.png"))
            plt.close()

    print(f"Attention/Feature maps saved to {args.out_dir}")


# -----------------------------------------------------------------------------
# 2. Texture Fidelity
# -----------------------------------------------------------------------------
def analyze_texture(args):
    print("Starting Texture Fidelity Analysis...")
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading VAE...")
    # Using standard SD VAE
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    except:
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    vae.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    valid = {".png", ".jpg", ".jpeg"}
    all_files = sorted([f for f in os.listdir(args.image_folder) if os.path.splitext(f)[1].lower() in valid])
    
    if not all_files:
        print(f"No images found in {args.image_folder}")
        return

    # Deterministic subset
    import random
    random.seed(42)
    files = random.sample(all_files, min(len(all_files), args.num_images))

    print(f"Processing {len(files)} images...")
    for f in tqdm(files):
        path = os.path.join(args.image_folder, f)
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Encode -> Decode
            z = vae.encode(x).latent_dist.sample()
            rec = vae.decode(z).sample
            
        # Clamp & Rescale for display
        x = torch.clamp(x, -1, 1)
        rec = torch.clamp(rec, -1, 1)
            
        # Make grid: [Original | Reconstructed]
        # (1, 3, H, W)
        comp = torch.cat([x, rec], dim=3) # Concatenate width-wise
        
        save_path = os.path.join(args.out_dir, f"compare_{f}")
        save_image(comp, save_path, normalize=True, value_range=(-1, 1))
        
    print(f"Texture comparisons saved to {args.out_dir}")


# -----------------------------------------------------------------------------
# 3. Nearest Neighbors
# -----------------------------------------------------------------------------
def analyze_nn(args):
    print("Starting Nearest Neighbor Analysis...")
    from sklearn.neighbors import NearestNeighbors
    os.makedirs(args.out_dir, exist_ok=True)
    
    def load_image_feature(path):
        img = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
        return np.array(img).flatten() / 255.0

    # Load Training Data (Reference)
    print(f"Loading reference images from {args.train_dir}...")
    valid = {".png", ".jpg", ".jpeg"}
    train_files = sorted([f for f in os.listdir(args.train_dir) if os.path.splitext(f)[1].lower() in valid])
    
    # Subsample if too large
    if len(train_files) > 5000:
        import random
        random.seed(42)
        train_files = random.sample(train_files, 5000)
    
    train_feats = []
    train_paths = []
    for f in tqdm(train_files, desc="Loading Train"):
        p = os.path.join(args.train_dir, f)
        try:
            train_feats.append(load_image_feature(p))
            train_paths.append(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if not train_feats:
        print("No training features loaded.")
        return
        
    train_feats = np.stack(train_feats)
    
    # Fit
    print("Fitting Nearest Neighbors Tree...")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train_feats)
    
    # Load Generated Data (Query)
    print(f"Loading generated images from {args.gen_dir}...")
    gen_files = sorted([f for f in os.listdir(args.gen_dir) if os.path.splitext(f)[1].lower() in valid])
    # Take top N
    gen_files = gen_files[:8]
    
    gen_feats = []
    gen_paths = []
    for f in tqdm(gen_files, desc="Loading Gen"):
        p = os.path.join(args.gen_dir, f)
        try:
            gen_feats.append(load_image_feature(p))
            gen_paths.append(p)
        except: pass
        
    if not gen_feats:
        print("No generated features loaded.")
        return

    gen_feats = np.stack(gen_feats)
    
    # Query
    distances, indices = nbrs.kneighbors(gen_feats)
    
    # Visualize
    # Create a long vertical strip of pairs
    # Pair: [Gen | Nearest Train]
    
    pairs = []
    for i in range(len(gen_paths)):
        gen_img = Image.open(gen_paths[i]).convert("RGB").resize((256, 256))
        
        train_idx = indices[i][0]
        train_img = Image.open(train_paths[train_idx]).convert("RGB").resize((256, 256))
        
        # Canvas (512, 256)
        canvas = Image.new("RGB", (512, 256))
        canvas.paste(gen_img, (0, 0))
        canvas.paste(train_img, (256, 0))
        pairs.append(canvas)
        
    # Stack vertically
    final_h = 256 * len(pairs)
    final_img = Image.new("RGB", (512, final_h))
    for i, p in enumerate(pairs):
        final_img.paste(p, (0, i * 256))
        
    save_p = os.path.join(args.out_dir, "nearest_neighbors_grid.jpg")
    final_img.save(save_p, quality=90)
    print(f"Nearest neighbor grid saved to {save_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative Analysis Suite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subparser: Attention
    p_attn = subparsers.add_parser("attention", help="Visualize internal activations/attention")
    p_attn.add_argument("--ckpt", required=True)
    p_attn.add_argument("--config", required=True)
    p_attn.add_argument("--out_dir", default="qualitative_analysis/attention")
    
    # Subparser: Texture
    p_tex = subparsers.add_parser("texture", help="Check VAE reconstruction quality")
    p_tex.add_argument("--image_folder", required=True, help="Folder of images to reconstruct")
    p_tex.add_argument("--out_dir", default="qualitative_analysis/texture")
    p_tex.add_argument("--num_images", type=int, default=8)
    
    # Subparser: NN
    p_nn = subparsers.add_parser("nn", help="Nearest Neighbor analysis")
    p_nn.add_argument("--gen_dir", required=True, help="Generated samples folder")
    p_nn.add_argument("--train_dir", required=True, help="Training data folder")
    p_nn.add_argument("--out_dir", default="qualitative_analysis/nn")
    
    args = parser.parse_args()
    
    if args.command == "attention":
        analyze_attention(args)
    elif args.command == "texture":
        analyze_texture(args)
    elif args.command == "nn":
        analyze_nn(args)
