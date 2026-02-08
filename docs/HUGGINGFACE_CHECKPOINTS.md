# Hugging Face integration — Checkpoints upload & download 🚀

This document explains how to upload and download experiment checkpoints (and related files) to Hugging Face using the helper scripts included in this repository.

- Upload script: `scripts/hf_upload_checkpoint.py`
- Download script: `scripts/hf_download_checkpoint.py`

---

## ✅ Goals
- Make experiments checkpointed under `universal_pix_workdirs/...` easy to publish to a Hugging Face repo.
- Keep one HF repo per experiment (recommended) or use a shared repo with per-experiment subfolders.
- Make it trivial to rehydrate checkpoints for remote training or inference.

---

## ⚙️ Requirements
- git and git-lfs installed and configured on the machine
  - Install: `sudo apt install git git-lfs` (or via your package manager)
  - Enable LFS: `git lfs install`
- Python packages: `huggingface_hub` (install with `pip install huggingface_hub`)
- Authentication: set `HUGGINGFACE_HUB_TOKEN` environment variable or pass `--token` to the scripts

---

## Upload basics (one-command examples)

- Create a new Hugging Face repo and upload `last.ckpt` from an experiment folder:

```bash
python scripts/hf_upload_checkpoint.py \
  --exp-dir universal_pix_workdirs/exp_PixelGen_S \
  --hf-repo your-user/PixelGen-Exp-S --create-repo
```

- Upload a specific checkpoint and include configs next to the checkpoint:

```bash
python scripts/hf_upload_checkpoint.py \
  --ckpt universal_pix_workdirs/exp_PixelGen_S/epoch=253-step=20000.ckpt \
  --hf-repo your-user/PixelGen-Exp-S --include-config
```

- Dry run (see what would be uploaded without pushing):

```bash
python scripts/hf_upload_checkpoint.py --exp-dir universal_pix_workdirs/exp_PixelGen_S --hf-repo your-user/PixelGen-Exp-S --dry-run
```

Notes:
- By default the script copies the selected files into the HF repo under a subdirectory named like the experiment folder (e.g., `exp_PixelGen_S/last.ckpt`).
- Files collected: checkpoint (.ckpt/.pt), optionally YAML configs and wandb summaries.

---

## Download basics

- List files in repo:

```bash
python scripts/hf_download_checkpoint.py --hf-repo your-user/PixelGen-Exp-S --list
```

- Download specific path:

```bash
python scripts/hf_download_checkpoint.py --hf-repo your-user/PixelGen-Exp-S \
  --path exp_PixelGen_S/last.ckpt --out-dir ./ckpts
```

This uses `hf_hub_download` to pull a single file and saves it to `--out-dir`.

---

## Recommended workflows & conventions 💡
- Per-experiment HF repo recommended: `your-user/PixelGen-<exp-name>` makes permissions and history simple.
- If you prefer a single HF repo, use `--exp-dir` to upload under a subfolder so experiments stay organized.
- Add an `upload_metadata.txt` (created automatically) in the experiment folder with origin info and commit message.
- For reproducibility upload the config YAMLs you used to run training (`--include-config`).

---

## Troubleshooting & tips ⚠️
- "Large files failing to push" → ensure `git-lfs` is installed and `git lfs install` was run. You can track manually: `git lfs track "*.ckpt"` then commit.
- If using `--create-repo` and creation fails, verify your token permissions: token needs `repo` (or `write` for org repos) scope.
- If a repo already exists, `create_repo(..., exist_ok=True)` is used to avoid hard failure.
- If you changed training code or argument semantics and only want to share model weights (not trainer state), prefer `--include-config` + documentation in the HF repo about how to rehydrate.

---

## CI / Automation ideas 🔁
- Add a GitHub Action or runner to upload `last.ckpt` on checkpoint/save (triggered by a push or workflow dispatch) using the same script with `--dry-run` for testing.
- Use a small push hook that uploads metadata and a small `manifest.json` with training hyperparameters for quick inspection.

---

## Security & storage notes 🔒
- Do not hardcode tokens in scripts. Use `HUGGINGFACE_HUB_TOKEN` or a secrets manager.
- Private repos are supported by `--create-repo --private`.

---

## Where to look in this repo
- Upload script: `scripts/hf_upload_checkpoint.py`
- Download/list script: `scripts/hf_download_checkpoint.py`
- Example experiment: `universal_pix_workdirs/exp_PixelGen_S`

---

## Docker-aware upload / download (workspace examples)
If you run in Docker and have your repository mounted at `/workspace` (recommended), use the same scripts but point to `/workspace` paths.

Upload from a container (one-shot):
```bash
# existing uploader (legacy)
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/hf_upload_checkpoint.py \
    --exp-dir /workspace/universal_pix_workdirs/exp_PixelGen_S \
    --hf-repo your-user/PixelGen-Exp-S --create-repo
```

Upload using unified sync (preferred, single-repo layout):
```bash
# upload all checkpoints matching 50k interval into single HF repo under exp folder
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/hf_sync.py upload \
    --exp-dir /workspace/universal_pix_workdirs/exp_PixelGen_S \
    --exp-name PixelGen_S \
    --hf-repo ruwwww/batik-pixelgen-exp \
    --interval 50000 --include-config
```

Download into the repo workspace (one-shot):
```bash
# legacy downloader
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/hf_download_checkpoint.py \
    --hf-repo your-user/PixelGen-Exp-S --path exp_PixelGen_S/last.ckpt --out-dir /workspace/ckpts
```

Download specific step from single-repo layout (preferred):
```bash
docker run --rm --gpus all \
  -v /path/to/repo:/workspace \
  -w /workspace \
  -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
  pixelgen:cu130 \
  python3 scripts/hf_sync.py download \
    --hf-repo ruwwww/batik-pixelgen-exp --exp-name PixelGen_S --step 50000 --out-dir /workspace/ckpts
```

Notes:
- These commands keep HF interactions inside the container and persist artifacts into the mounted `/workspace` tree (so they appear on the host as well).
- Ensure `git` and `git-lfs` are available in your image if using the upload script (the current `hf_upload_checkpoint.py` uses `huggingface_hub.Repository` which uses git + lfs under the hood).

---

If you want, I can:
- Add a GitHub Action example file to automate uploads, or
- Run a test upload for `universal_pix_workdirs/exp_PixelGen_S/last.ckpt` to a repo name you provide.

