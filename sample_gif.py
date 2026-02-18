import torch
import os
import glob
from PIL import Image
import numpy as np
from lightning.pytorch import LightningModule, Trainer
from src.lightning_model import LightningModel
from src.lightning_data import DataModule
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

# Custom CLI to load model without full config
class SamplingCLI(LightningCLI):
    def __init__(self, model_class, datamodule_class, **kwargs):
        super().__init__(model_class, datamodule_class, **kwargs)

    def instantiate_classes(self):
        super().instantiate_classes()
        # Load checkpoint
        ckpt_path = self._get(self.config, "ckpt_path")
        if ckpt_path:
            self.model = self.model.__class__.load_from_checkpoint(ckpt_path, **self.model_kwargs)
            print(f"Loaded checkpoint: {ckpt_path}")

def sample_and_save_gif(checkpoints, output_dir, num_samples=4, gif_fps=2):
    os.makedirs(output_dir, exist_ok=True)

    # Fix seed for consistent sampling
    torch.manual_seed(42)
    np.random.seed(42)

    frames = []

    for ckpt in checkpoints:
        print(f"Sampling from {ckpt}")

        # Load model from checkpoint
        model = LightningModel.load_from_checkpoint(ckpt)
        model.eval()

        # Create datamodule for prediction
        datamodule = DataModule(
            train_dataset=None,
            eval_dataset=None,
            pred_dataset={
                "class_path": "src.data.dataset.randn.ClassLabelRandomNDataset",
                "init_args": {
                    "num_classes": 20,  # Adjust to your num_classes
                    "max_num_instances": num_samples,
                    "noise_scale": 1.0,
                    "latent_shape": [3, 256, 256]
                }
            },
            train_batch_size=1,
            pred_batch_size=num_samples
        )

        trainer = Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

        # Predict
        predictions = trainer.predict(model, datamodule)

        # Save images
        for i, batch in enumerate(predictions):
            samples = batch  # Shape: [B, C, H, W]
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            samples = np.clip(samples, -1, 1)
            samples = ((samples + 1) / 2 * 255).astype(np.uint8)

            for j in range(num_samples):
                img = Image.fromarray(samples[j])
                img.save(os.path.join(output_dir, f"ckpt_{os.path.basename(ckpt).split('.')[0]}_sample_{j}.png"))

        # Collect first sample for GIF
        first_sample = samples[0]
        frames.append(Image.fromarray(first_sample))

    # Create GIF
    gif_path = os.path.join(output_dir, "learning_progression.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000//gif_fps, loop=0)
    print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    # Example usage: python sample_gif.py /path/to/checkpoints/*.ckpt
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sample_gif.py <checkpoint_pattern> [output_dir]")
        sys.exit(1)

    checkpoint_pattern = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sampling_gif"

    checkpoints = sorted(glob.glob(checkpoint_pattern))
    if not checkpoints:
        print("No checkpoints found")
        sys.exit(1)

    sample_and_save_gif(checkpoints, output_dir)