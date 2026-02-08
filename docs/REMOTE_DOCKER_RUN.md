# Remote Docker commands (copy-paste ready) 🚀

This file contains minimal, environment-agnostic Docker commands you can copy-paste on a remote machine. Replace the placeholder paths and tokens with your values.

---

## Build image
```bash
# from repository root (contains docker/Dockerfile)
docker build -t pixelgen:cu130 -f docker/Dockerfile .
```

## Create host data directory (persistent)
```bash
# create a persistent host folder for datasets
mkdir -p /path/to/data
# ensure the current user owns it (optional if using sudo)
sudo chown $(id -u):$(id -g) /path/to/data
```

## Download dataset (one-shot, runs and exits)
```bash
# public dataset
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -v /path/to/data:/data \
  -w /workspace \
  pixelgen:cu130 \
  python3 scripts/get_data.py --out-dir /data

# if private/gated dataset (pass HF token)
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -v /path/to/data:/data \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/get_data.py --out-dir /data --repo your-user/dataset-name --repo-type dataset
```

## Run training (CLI override to use `/data` inside container)
```bash
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -v /path/to/data:/data \
  -w /workspace \
  pixelgen:cu130 \
  python3 main.py fit -c ./configs_c2i/PixelGen_Small.yaml \
    --data.train_dataset.init_args.root=/data/batik-256/images \
    --trainer.default_root_dir=/workspace/universal_pix_workdirs \
    --ckpt_path ./universal_pix_workdirs/exp_PixelGen_S/last.ckpt
```

## Interactive container (inspect, debug)
```bash
docker run --rm -it --gpus all \
  -v /path/to/repo:/workspace \
  -v /path/to/data:/data \
  -w /workspace \
  pixelgen:cu130 bash
# then inside the container:
# python3 scripts/get_data.py --out-dir /data
# or run other commands interactively
```

## Notes & tips ⚠️
- If Docker commands return "permission denied" for the socket, either run with `sudo` or add your user to the `docker` group:
  sudo usermod -aG docker $USER  &&  newgrp docker
- If `python3` is missing in the image, add it to `docker/Dockerfile` and rebuild (install `python3 python3-pip` and required pip packages).
- Keep dataset out of Docker image; mount host directory (`/path/to/data`) into `/data` inside container for persistence.
- Use `--force` on the downloader to re-extract if needed: `python3 scripts/get_data.py --out-dir /data --force`.

---

### Checkpoints: upload & download from inside container ✅

If you run from a container with your repo mounted as `/workspace`, use the `scripts/hf_upload_checkpoint.py` and `scripts/hf_download_checkpoint.py` helpers with workspace paths.

Upload example (inside container or one-shot):
```bash
# upload experiment folder under /workspace/universal_pix_workdirs
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/hf_upload_checkpoint.py \
    --exp-dir /workspace/universal_pix_workdirs/exp_PixelGen_S \
    --hf-repo your-user/PixelGen-Exp-S --create-repo
```

Download example (one-shot into workspace/ckpts):
```bash
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/hf_download_checkpoint.py \
    --hf-repo your-user/PixelGen-Exp-S --path exp_PixelGen_S/last.ckpt --out-dir /workspace/ckpts
```

Or run interactively and call the scripts from the shell inside the container.

---

If you want, I can add this snippet to `README.md` as a new "Docker / Remote quickstart" section or create a `configs_c2i/PixelGen_Small_docker.yaml` with `/data` paths. Which would you prefer? 🔧