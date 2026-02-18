#!/usr/bin/env python3
"""Sample a saved checkpoint and save the final decoded images (PNG).

Usage:
  python sample_final_image.py --ckpt /path/to/last.ckpt --out /path/to/outdir --num_samples 4
"""
import argparse
import os
import glob
import importlib
import inspect
import torch
import numpy as np
from PIL import Image

from src.lightning_model import LightningModel


def to_image(x):
    # x: tensor [C,H,W] or [H,W,C] in [-1,1]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, -1, 1)
    x = ((x + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(x)


def resolve_python_path(val):
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
        for k, v in list(init_args.items()):
            if isinstance(v, dict) and 'class_path' in v:
                init_args[k] = instantiate_node(v)
            elif isinstance(v, str) and '.' in v:
                init_args[k] = resolve_python_path(v)
        module_name, class_name = class_path.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls(**init_args)
    if isinstance(node, str) and '.' in node:
        return resolve_python_path(node)
    return node


def load_model(ckpt, device='cpu'):
    print(f"Loading checkpoint: {ckpt}")
    # Try direct Lightning load
    try:
        model: LightningModel = LightningModel.load_from_checkpoint(ckpt)
        print("Loaded LightningModel via load_from_checkpoint")
    except Exception as e:
        print(f"Direct load failed ({e}). Trying to instantiate model from config...")
        import yaml
        ckpt_dir = os.path.dirname(ckpt)
        yaml_candidates = glob.glob(os.path.join(ckpt_dir, "*.yaml")) + glob.glob(os.path.join(ckpt_dir, "*.yml"))
        config_path = None
        if yaml_candidates:
            for p in yaml_candidates:
                if 'config' in os.path.basename(p) or 'cfg' in os.path.basename(p):
                    config_path = p
                    break
            if config_path is None:
                config_path = yaml_candidates[0]
        if config_path is None:
            raise RuntimeError("Cannot find config next to checkpoint. Rerun script with --config /path/to/config.yaml")

        conf = yaml.safe_load(open(config_path))
        model_conf = conf.get('model', {})

        vae = instantiate_node(model_conf['vae'])
        denoiser = instantiate_node(model_conf['denoiser'])
        conditioner = instantiate_node(model_conf['conditioner'])
        # Skip diffusion_trainer for inference (it contains perceptual losses we don't need)
        # diffusion_trainer = instantiate_node(model_conf['diffusion_trainer'])
        diffusion_sampler = instantiate_node(model_conf['diffusion_sampler'])

        # Create a minimal model without trainer (set to None for inference)
        model = LightningModel(vae, conditioner, denoiser, None, diffusion_sampler)

        ckpt_data = torch.load(ckpt, map_location='cpu')
        sd = ckpt_data.get('state_dict', ckpt_data)
        model.load_state_dict(sd, strict=False)
        print("Instantiated model from config and loaded state_dict (non-strict)")

    model.to(device)
    model.eval()
    return model


def sample_and_save(ckpt, out_dir, num_samples=4, classes=None, use_ema=True, device=None, grid=True, ncols=None, seed=None):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(ckpt, device=device)

    net = model.ema_denoiser if use_ema else model.denoiser
    net.to(device)
    net.eval()

    sampler = model.diffusion_sampler

    C = model.denoiser.in_channels
    S = model.denoiser.input_size
    latent_shape = (C, S, S)

    if classes is None:
        num_classes = getattr(model.conditioner, 'num_classes', None)
        if num_classes is None:
            num_classes = getattr(model.conditioner, 'null_condition', None)
        if num_classes is None:
            num_classes = getattr(model.denoiser, 'num_classes', None)
        if num_classes is None:
            num_classes = 20
            print(f"[sample_final_image] Warning: couldn't find num_classes; defaulting to {num_classes}")
        y = torch.randint(0, int(num_classes), (num_samples,), device=device)
    else:
        y = torch.as_tensor(classes, device=device, dtype=torch.long)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if y.shape[0] < num_samples:
            y = y.repeat(int(np.ceil(num_samples / y.shape[0])))[:num_samples]

    with torch.no_grad():
        condition, uncondition = model.conditioner(y)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    noise = torch.randn((num_samples, *latent_shape), device=device)

    print("Sampling final images (this may take a while)...")
    with torch.no_grad():
        res = sampler(net, noise, condition, uncondition)
        # sampler often returns final sample as first output
        if isinstance(res, (tuple, list)):
            last = res[0]
        else:
            last = res

    # decode batch
    with torch.no_grad():
        decoded = model.vae.decode(last)

    imgs = [to_image(decoded[i]) for i in range(decoded.shape[0])]

    base = f"ckpt_{os.path.basename(ckpt).split('.')[0]}"

    # save per-sample images
    for i, im in enumerate(imgs):
        path = os.path.join(out_dir, f"{base}_sample_{i}.png")
        im.save(path)
        print(f"Saved image: {path}")

    # optionally save grid
    if grid:
        import math
        n = len(imgs)
        if ncols is None:
            ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        W, H = imgs[0].size
        grid_img = Image.new('RGB', (ncols * W, nrows * H))
        for idx, im in enumerate(imgs):
            r = idx // ncols
            c = idx % ncols
            grid_img.paste(im, (c * W, r * H))
        grid_path = os.path.join(out_dir, f"{base}_final_grid.png")
        grid_img.save(grid_path)
        print(f"Saved final grid PNG: {grid_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default='sampling_final')
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated class indices or "random"')
    parser.add_argument('--no-ema', action='store_true', help='Use raw denoiser instead of EMA')
    parser.add_argument('--no-grid', action='store_true', help='Disable saving grid PNG (only per-sample PNGs)')
    parser.add_argument('--ncols', type=int, default=None, help='Number of columns for grid (default=sqrt(num_samples))')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.classes is None or (isinstance(args.classes, str) and args.classes.lower() == 'random'):
        classes = None
    else:
        classes = [int(x) for x in args.classes.split(',')]

    sample_and_save(args.ckpt, args.out, num_samples=args.num_samples, classes=classes, use_ema=not args.no_ema, grid=not args.no_grid, ncols=args.ncols, seed=args.seed)
