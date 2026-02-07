#!/usr/bin/env python3
"""Sample a saved checkpoint and produce per-sample GIFs of the denoising trajectory.

Usage:
  python sample_traj_gif.py --ckpt /path/to/last.ckpt --out /path/to/outdir --num_samples 4 --fps 4 --steps_per_frame 1
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

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


def sample_traj_gif(ckpt, out_dir, num_samples=4, classes=None, fps=4, steps_per_frame=1, use_ema=True, device=None):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {ckpt}")
    model = None
    # Try direct Lightning load
    try:
        model: LightningModel = LightningModel.load_from_checkpoint(ckpt)
        print("Loaded LightningModel via load_from_checkpoint")
    except TypeError as e:
        print(f"Direct load failed ({e}). Trying to instantiate model from config...")
        # require config yaml path in CKPT directory or via env
        import yaml, importlib
        # Try to find config next to checkpoint (search for YAML files)
        config_path = None
        ckpt_dir = os.path.dirname(ckpt)
        import glob
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

        conf = yaml.safe_load(open(config_path))
        model_conf = conf.get('model', {})

        import inspect
        def resolve_python_path(val):
            """If val is a string like 'module.attr', import and return attr. If it's a class, try to instantiate it with no args; otherwise return attr."""
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
        diffusion_trainer = instantiate_node(model_conf['diffusion_trainer'])
        diffusion_sampler = instantiate_node(model_conf['diffusion_sampler'])

        model = LightningModel(vae, conditioner, denoiser, diffusion_trainer, diffusion_sampler)

        # load checkpoint state dict
        ckpt_data = torch.load(ckpt, map_location='cpu')
        sd = ckpt_data.get('state_dict', ckpt_data)
        model.load_state_dict(sd, strict=False)
        print("Instantiated model from config and loaded state_dict (non-strict)")

    model.to(device)
    model.eval()

    # choose net
    net = model.ema_denoiser if use_ema else model.denoiser
    net.to(device)
    net.eval()

    sampler = model.diffusion_sampler

    # noise / latent shape: infer from denoiser input_size and channels
    C = model.denoiser.in_channels
    S = model.denoiser.input_size
    latent_shape = (C, S, S)

    # prepare labels
    if classes is None:
        # discover num_classes from common locations
        num_classes = getattr(model.conditioner, 'num_classes', None)
        if num_classes is None:
            num_classes = getattr(model.conditioner, 'null_condition', None)
        if num_classes is None:
            num_classes = getattr(model.denoiser, 'num_classes', None)
        if num_classes is None:
            num_classes = 20
            print(f"[sample_traj_gif] Warning: couldn't find num_classes in conditioner/denoiser; defaulting to {num_classes}")
        # random classes
        y = torch.randint(0, int(num_classes), (num_samples,), device=device)
    else:
        y = torch.as_tensor(classes, device=device, dtype=torch.long)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if y.shape[0] < num_samples:
            # repeat or pad
            y = y.repeat(int(np.ceil(num_samples / y.shape[0])))[:num_samples]

    # get condition tensors
    with torch.no_grad():
        condition, uncondition = model.conditioner(y)

    # create initial noise
    noise = torch.randn((num_samples, *latent_shape), device=device)

    # sample trajectories (appending if duration requires it)
    desired_frames = getattr(sample_traj_gif, 'target_frames', None)
    
    # If target_frames is set (via --duration), temporarily set sampler.num_steps to match
    orig_steps = getattr(sampler, 'num_steps', None)
    if desired_frames is not None:
        # sampler produces num_steps+1 frames (initial + steps)
        desired_steps = desired_frames - 1
        max_steps = getattr(sample_traj_gif, 'max_steps', 1000)
        if desired_steps > max_steps:
             raise ValueError(f"Requested {desired_steps} steps exceeds safety cap of {max_steps}. Use --max-steps to increase.")
        print(f"[sample_traj_gif] Temporarily setting sampler.num_steps={desired_steps} to produce {desired_frames} frames (duration target).")
        sampler.num_steps = desired_steps

    print("Sampling trajectories (this may take a while)...")
    with torch.no_grad():
        last, x_trajs = sampler(net, noise, condition, uncondition, return_x_trajs=True)

    # restore sampler settings
    if desired_frames is not None and orig_steps is not None:
        sampler.num_steps = orig_steps

    # x_trajs: list of tensors [B,C,H,W]
    print(f"Collected {len(x_trajs)} frames")

    # If duration requested and current trajectory is too short, append extra trajectories until we reach target frames
    if desired_frames is not None and len(x_trajs) < desired_frames:
        max_frames = getattr(sample_traj_gif, 'max_frames', 5000)
        est_total = desired_frames
        if est_total > max_frames and not getattr(sample_traj_gif, 'force', False):
             raise RuntimeError(f"Requested total frames {est_total} exceeds safety cap {max_frames}. Use --force to override or lower duration.")
        print(f"[sample_traj_gif] Appending trajectories to reach {desired_frames} frames (current {len(x_trajs)}).")
        appended = 0
        round_idx = 0
        while len(x_trajs) < desired_frames:
            round_idx += 1
            # vary seed and optionally classes
            torch.manual_seed(12345 + round_idx)
            y_extra = torch.randint(0, getattr(model.conditioner, 'num_classes', 20), (num_samples,), device=device)
            with torch.no_grad():
                cond_e, uncond_e = model.conditioner(y_extra)
                _, extra_traj = sampler(net, torch.randn((num_samples, *latent_shape), device=device), cond_e, uncond_e, return_x_trajs=True)
            # append frames from extra_traj
            for f in extra_traj:
                x_trajs.append(f)
                appended += 1
                if len(x_trajs) >= desired_frames:
                    break
            print(f"[sample_traj_gif] Appended trajectory {round_idx}, total frames now {len(x_trajs)}")
        print(f"[sample_traj_gif] Finished appending {round_idx} extra trajectories, total frames {len(x_trajs)}")

    # If longer than desired, truncate
    if desired_frames is not None and len(x_trajs) > desired_frames:
        x_trajs = x_trajs[:desired_frames]
        print(f"[sample_traj_gif] Truncated to {desired_frames} frames")
    # subsample frames
    frames_idx = list(range(0, len(x_trajs), steps_per_frame))
    if frames_idx[-1] != len(x_trajs) - 1:
        frames_idx.append(len(x_trajs) - 1)

    # Option: build an animated grid GIF (default) or per-sample GIFs (if --no-grid)
    if not getattr(__import__('argparse'), 'Namespace') and False: # pragma: no cover
        pass

    if not hasattr(sample_traj_gif, '__grid_default__'):
        # default behavior: grid GIF
        pass

    if not getattr(sample_traj_gif, 'no_grid', False):
        import math
        n = num_samples
        ncols = None
        
        # Checking for imageio for memory-efficient writing
        try:
            import imageio
            use_imageio = True
        except ImportError:
            use_imageio = False
            print("Warning: imageio not found. Using PIL to save GIF, which may consume large amounts of RAM.")

        # Use MP4 if imageio is available, otherwise fallback to GIF
        ext = 'mp4' if use_imageio else 'gif'
        grid_video_path = os.path.join(out_dir, f"ckpt_{os.path.basename(ckpt).split('.')[0]}_grid.{ext}")
        
        # Prepare writer or list
        writer = None
        grid_frames = []
        if use_imageio:
             # Use macro_block_size=None only if we can confirm it's FFMPEG, but strictly speaking 
             # PyAV doesn't like it. Let's try without it. If it's FFMPEG and fails on divisibility, 
             # we might need to be smarter, but removing it fixes the PyAV crash.
             try:
                 writer = imageio.get_writer(grid_video_path, fps=fps, macro_block_size=None)
             except TypeError:
                 # Fallback for plugins like PyAV that don't accept macro_block_size
                 print("Basic writer init failed, retrying without macro_block_size...")
                 writer = imageio.get_writer(grid_video_path, fps=fps)
             
             print(f"Saving grid video incrementally to {grid_video_path} ...")
        else:
             print("Collecting frames in memory (imageio not found, falling back to GIF)...")

        print(f"Decoding {len(frames_idx)} frames...")
        
        for t_idx, t in enumerate(frames_idx):
            # Batch decode for speed
            with torch.no_grad():
                # Ensure input is on correct device
                batch_tensor = x_trajs[t].to(device)
                decoded_batch = model.vae.decode(batch_tensor)
            
            imgs = []
            for i in range(num_samples):
                pil = to_image(decoded_batch[i])
                imgs.append(pil)
            
            # Helper to clear GPU mem for this batch
            del decoded_batch

            # determine ncols/nrows (square by default)
            if sample_traj_gif.grid_ncols is None:
                ncols = int(math.ceil(math.sqrt(n)))
            else:
                ncols = sample_traj_gif.grid_ncols
            nrows = int(math.ceil(n / ncols))

            W, H = imgs[0].size
            grid_img = Image.new('RGB', (ncols * W, nrows * H))
            for idx, im in enumerate(imgs):
                r = idx // ncols
                c = idx % ncols
                grid_img.paste(im, (c * W, r * H))
            
            if use_imageio:
                writer.append_data(np.array(grid_img))
            else:
                grid_frames.append(grid_img)

            if t_idx % max(1, len(frames_idx)//10) == 0:
                print(f"Processed frame {t_idx}/{len(frames_idx)}")

        if use_imageio:
            writer.close()
            print(f"Finished saving grid video: {grid_video_path}")
            
            # Save final grid PNG separately
            final_grid_path = os.path.join(out_dir, f"ckpt_{os.path.basename(ckpt).split('.')[0]}_final_grid.png")
            # Reconstruct last frame purely for PNG save (or cache it?)
            # Since we didn't keep grid_frames, we grab the last grid_img from loop context
            if 'grid_img' in locals():
                grid_img.save(final_grid_path)
                print(f"Saved final grid PNG: {final_grid_path}")
        else:
            # save animated grid GIF via PIL
            if len(grid_frames) > 0:
                print("Saving accumulated frames to GIF (this may take a while)...")
                grid_frames[0].save(grid_video_path, save_all=True, append_images=grid_frames[1:], duration=1000//fps, loop=0)
                print(f"Saved grid GIF: {grid_video_path}")

                # also save final-grid PNG
                final_grid_path = os.path.join(out_dir, f"ckpt_{os.path.basename(ckpt).split('.')[0]}_final_grid.png")
                grid_frames[-1].save(final_grid_path)
                print(f"Saved final grid PNG: {final_grid_path}")
    else:
        # fallback: per-sample GIFs
        for i in range(num_samples):
            frames = []
            for t in frames_idx:
                x = x_trajs[t][i]
                img = model.vae.decode(x.unsqueeze(0))[0]
                pil = to_image(img)
                frames.append(pil)

            gif_path = os.path.join(out_dir, f"ckpt_{os.path.basename(ckpt).split('.')[0]}_sample_{i}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
            print(f"Saved GIF: {gif_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default='sampling_traj')
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--steps_per_frame', type=int, default=1, help='Take every k-th sampling step as a GIF frame')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated class indices or "random"')
    parser.add_argument('--no-ema', action='store_true', help='Use raw denoiser instead of EMA')
    parser.add_argument('--no-grid', action='store_true', help='Disable the default grid GIF output')
    parser.add_argument('--ncols', type=int, default=None, help='Number of columns for grid (default=sqrt(num_samples))')
    parser.add_argument('--duration', type=float, default=None, help='Total GIF duration in seconds (overrides steps_per_frame by matching num_steps)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum sampler steps allowed when using --duration (safety cap)')
    parser.add_argument('--max-frames', type=int, default=5000, help='Maximum total frames allowed for generated GIFs (safety cap)')
    parser.add_argument('--fast-steps', type=int, default=12, help='Fast temporary sampler steps for heavy "new_batch" sampling')
    parser.add_argument('--force', action='store_true', help='Force execution even if estimated work is large')
    args = parser.parse_args()

    # Attach grid preferences onto the function so the sampling logic can read them
    sample_traj_gif.no_grid = args.no_grid
    sample_traj_gif.grid_ncols = args.ncols
    sample_traj_gif.desired_duration = args.duration
    sample_traj_gif.max_steps = args.max_steps
    sample_traj_gif.fast_steps = args.fast_steps
    sample_traj_gif.force = args.force


    if args.classes is None or args.classes.lower() == 'random':
        classes = None
    else:
        classes = [int(x) for x in args.classes.split(',')]

    # If duration specified, compute target frames and set steps_per_frame to 1
    steps_per_frame = args.steps_per_frame
    if args.duration is not None:
        target_frames = int(round(args.fps * args.duration))
        if target_frames <= 1:
            raise ValueError("--duration too small for the given fps")
        steps_per_frame = 1
        sample_traj_gif.target_frames = target_frames
    else:
        sample_traj_gif.target_frames = None

    sample_traj_gif(args.ckpt, args.out, num_samples=args.num_samples, classes=classes, fps=args.fps, steps_per_frame=steps_per_frame, use_ema=not args.no_ema)
