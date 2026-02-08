import lightning.pytorch as pl
from lightning.pytorch import Callback
import torch
import os
from PIL import Image
import numpy as np

from concurrent.futures import ThreadPoolExecutor

class MidTrainingSampler(Callback):
    def __init__(self, sample_every_n_steps=1000, num_samples=4, save_dir="mid_training_samples", fast_steps=12, fast_guidance=1.0, async_mode=False):
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.fast_steps = fast_steps
        self.fast_guidance = fast_guidance
        self.async_mode = async_mode
        os.makedirs(save_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2) if self.async_mode else None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.sample_every_n_steps == 0 and trainer.global_step > 0:
            if self.async_mode and self.executor is not None:
                # submit async job to avoid blocking training
                self.executor.submit(self._async_sample_and_save, trainer.global_step, pl_module)
            else:
                # Blocking sampling (safer, deterministic)
                print(f"[MidTrainingSampler] Blocking sampling will run on step {trainer.global_step}")
                self._async_sample_and_save(trainer.global_step, pl_module)

    def _async_sample_and_save(self, step, pl_module):
        # Use EMA if available and ensure it's updated
        print(f"[MidTrainingSampler] Starting async sampling at step {step}")
        try:
            # Try to call EMA step to make sure EMA is up-to-date
            if hasattr(pl_module, 'ema_tracker') and hasattr(pl_module.ema_tracker, 'ema_step'):
                try:
                    pl_module.ema_tracker.ema_step()
                except Exception:
                    pass

            device = pl_module.device
            batch_size = self.num_samples

            # Use deterministic classes for debugging diversity
            num_classes = getattr(pl_module.conditioner, 'num_classes', 20)
            y = torch.arange(batch_size, device=device) % num_classes

            # Use a fixed but varying noise for visibility
            xT = torch.randn(batch_size, 3, 256, 256, device=device)

            # Log debug info
            print(f"[MidTrainingSampler] step={step} classes={y.tolist()} xT_mean={xT.mean().item():.4f} std={xT.std().item():.4f}")

            # Prepare conditions
            with torch.no_grad():
                condition, uncondition = pl_module.conditioner(y)

                # Use EMA model if available
                net = pl_module.ema_denoiser if not pl_module.eval_original_model else pl_module.denoiser

                # Temporarily adjust sampler for a quick check
                sampler = pl_module.diffusion_sampler
                orig_steps = getattr(sampler, 'num_steps', None)
                orig_guidance = getattr(sampler, 'guidance', None)
                try:
                    sampler.num_steps = min(self.fast_steps, orig_steps if orig_steps is not None else self.fast_steps)
                    sampler.guidance = self.fast_guidance

                    # Request trajectories so we can build GIFs
                    out = sampler(net, xT, condition, uncondition, return_x_trajs=True)
                    if isinstance(out, tuple) and len(out) >= 2:
                        _, x_trajs = out[0], out[1]
                    else:
                        # fallback: sampler returned last image only. Decode and wrap
                        last = out
                        x_trajs = [last]

                    # Convert and save per-sample GIFs
                    frames_idx = list(range(0, len(x_trajs)))
                    for i in range(batch_size):
                        frames = []
                        for t in frames_idx:
                            x = x_trajs[t][i]
                            img = pl_module.vae.decode(x.unsqueeze(0))[0]
                            img = img.permute(1, 2, 0).cpu().numpy()
                            img = np.clip(img, -1, 1)
                            img = ((img + 1) / 2 * 255).astype(np.uint8)
                            frames.append(Image.fromarray(img))

                        gif_path = os.path.join(self.save_dir, f"step_{step}_sample_{i}.gif")
                        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)

                    # Diversity check: compute pairwise distances between final frames
                    final = x_trajs[-1].detach().cpu()  # [B,C,H,W]
                    B = final.shape[0]
                    dists = []
                    for a in range(B):
                        for b in range(a+1, B):
                            d = torch.mean(torch.abs(final[a] - final[b])).item()
                            dists.append(d)
                    print(f"[MidTrainingSampler] step={step} pairwise_mean_abs_diff={np.mean(dists) if dists else 0:.6f}")

                finally:
                    # restore sampler
                    if orig_steps is not None:
                        sampler.num_steps = orig_steps
                    if orig_guidance is not None:
                        sampler.guidance = orig_guidance

        except Exception as e:
            print(f"[MidTrainingSampler] sampling failed at step {step}: {e}")

        print(f"[MidTrainingSampler] Finished sampling at step {step}")
