# DiT_BEP — Diffusion Transformers for Magnetic Field Modeling in Medium Frequency Transformers

> **Official PyTorch Implementation** based on [Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)](https://arxiv.org/abs/2212.09748), adapted for modeling magnetic field distribution in Medium Frequency Transformers (MFTs) as an alternative to Finite Element Methods (FEM).

---

## Overview

This repository trains a **Diffusion Transformer (DiT)** to model the 2D magnetic field distribution in the In-Window (IW) section of a Medium Frequency Transformer, given an image of the transformer's winding structure as a condition. The model achieves an SSIM of **0.92** and generates images in **~8 seconds** — roughly **half the time** of traditional FEM simulation.

![MFT structure and magnetic field distribution](visuals/fig1_2.png) 
---

## Key Results

| Model   | SSIM | PSNR (dB) | LPIPS | Generation Time |
|---------|------|-----------|-------|-----------------|
| DiT-XS  | 0.92 | 22.52     | 0.050 | ~8s             |
| DiT-S   | 0.94 | 31.94     | 0.037 | ~25s            |
| DiT-S+  | 0.95 | 32.20     | 0.028 | ~25s            |
| DiT-B   | 0.95 | 32.37     | 0.028 | ~40s            |
| DiT-L   | 0.96 | 32.67     | 0.025 | ~105s           |

FEM baseline: ~16 seconds per image. DiT-XS generates in ~8 seconds (2× faster).

![MFT structure, Ground truth field, DiT reconstruction](visuals/fig7_8_9.png)
---

## How It Works

The model takes **pairs of images** as input:
- **Orange images** — the transformer winding structure (condition/input)
- **Blue images** — the corresponding magnetic field distribution (target)

The dataset consists of 1000 such pairs, split 80/10/10 into train/val/test. The DiT learns to generate the magnetic field image conditioned on the structure image, via a ResNet-18 image embedder that provides context to the diffusion process.

![Diffusion Process](visuals/fig3_diffusion.png)

Furthermore, a general overview of the software workflow of the Diffusion Model for Image Generation in this context is depicted below.

![Diffusion Process](visuals/fig4.png)



---

## Setup

```bash
git clone https://github.com/umervc/DiT_BEP.git
cd DiT_BEP
pip install torch lightning diffusers torchvision matplotlib
```

The VAE (`stabilityai/sd-vae-ft-ema`) will be downloaded automatically from HuggingFace on first run.

---

## Dataset

Your dataset must be organized as follows:

```
your_dataset/
├── Orange/    ← MFT winding structure images (condition)
└── Blue/      ← Magnetic field distribution images (target)
```

Filenames in `Blue/` must correspond to those in `Orange/` with `"o"` replaced by `"p"` (e.g., `structure_o_001.png` → `structure_p_001.png`).

The dataset is automatically split 80/10/10 (train/val/test) by the `ImgDataset` class. Update the dataset path inside `modules/dit_clipped.py` (or wherever the dataloader is initialized) to point to your local dataset directory.

Images are preprocessed with center-cropping, random horizontal flipping, tensor conversion, and normalization to `[-1, 1]`.

---

## Training

```bash
python train.py
```

With all defaults this trains `DiT_Clipped` (6 layers, hidden size 640, 4 heads, patch size 2) at 256×256 resolution for 500 epochs with batch size 8 in fp16.

### All Training Arguments

| Argument | Default | Options | Description |
|---|---|---|---|
| `--model` | `DiT_Clipped` | `DiT_Clipped` | Model architecture variant |
| `--image-size` | `256` | `128, 256, 512, 1024, 2048` | Input image resolution |
| `--epochs` | `500` | any int | Number of training epochs |
| `--global-batch-size` | `8` | any int | Batch size across all GPUs |
| `--global-seed` | `0` | any int | Random seed |
| `--vae` | `ema` | `ema`, `mse` | Stability AI VAE checkpoint (does not affect training) |
| `--num-workers` | `4` | any int | Dataloader worker threads |
| `--precision` | `fp16` | `fp16`, `fp32` | Training precision |

**Example — 512px training with smaller batch:**
```bash
python train.py --image-size 512 --global-batch-size 4 --epochs 150
```

Multi-GPU training is handled automatically via PyTorch Lightning DDP.

### Training Outputs

| File | Description |
|---|---|
| `ckpts/` | Checkpoints saved every 5 epochs; top 5 by `train_loss` + last |
| `losses.csv` | Per-epoch log: `[epoch, train_loss, val_loss]` |
| `loss_plot.png` | Train vs. validation loss curve, saved at end of training |

### Resuming from a Checkpoint

Uncomment and edit these lines in `train.py`:

```python
state_dict = find_model("ckpts/your_checkpoint.ckpt")
if 'pytorch-lightning_version' in state_dict.keys():
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict, strict=False)
```

---

## Sampling

```bash
python sample.py --ckpt /path/to/checkpoint.pt --image-size 256
```

### Sampling Arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | `None` | Path to trained checkpoint |
| `--image-size` | `256` | Image resolution |
| `--num-sampling-steps` | `250` | DDPM denoising steps (use `25` for fast sampling as in the paper) |
| `--cfg-scale` | `4.0` | Classifier-free guidance scale (≥ 1.0) |
| `--vae` | `mse` | VAE checkpoint |
| `--seed` | `0` | Random seed |

Output is saved as `sample.png`.

For large-scale parallel sampling (e.g. for FID evaluation):

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --num-fid-samples 50000
```

---

## Model Architecture

Only one model variant is currently defined — `DiT_Clipped`:

| Parameter | Value |
|---|---|
| Transformer layers | 6 |
| Hidden size | 640 |
| Attention heads | 4 |
| Patch size | 2 |
| Conditioning | ResNet-18 image embedder + adaLN-Zero |
| Diffusion steps | 1000 (linear schedule) |
| Loss function | Log-Cosh |
| Sigma | Learned |

To experiment with new architectures, add new builder functions to `modules/dit_builder.py` and register them in `DiT_models`.

---

## Diffusion Configuration

The diffusion process is configured in `modules/diffusion.py` with the following defaults (not exposed as CLI arguments — edit the file directly to change):

- **Timesteps:** 1000
- **Noise schedule:** Linear (β from 0.0001 to 0.02)
- **Loss:** Log-Cosh (outperformed MSE and MAE in experiments)
- **Variance:** Learned range
- **Prediction target:** Epsilon (noise)

---

## Hardware Requirements

| Image Size | Recommended GPU |
|---|---|
| 256×256 | NVIDIA A40 (48 GB VRAM) or equivalent |
| 512×512 | NVIDIA A100 (80 GB VRAM) or equivalent |

---

## Visuals Guide (for repo maintainer)

The following figures from the paper are good candidates to add to `assets/` and embed in this README:

| Figure | Suggested filename | Where to embed |
|---|---|---|
| Fig. 2 — MFT structure + field pair | `assets/fig2_data_example.png` | Overview section |
| Fig. 3 — Forward/reverse diffusion | `assets/fig3_diffusion.png` | How It Works section |
| Fig. 4 — Full software workflow | `assets/fig4_workflow.png` | How It Works section |
| Fig. 7/8/9 — Structure, ground truth, DiT output | `assets/fig7_structure.png` etc. | Results section |

---

## Citation

If you use this code in your work, please cite the original DiT paper and this study:

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748}
}
```

```bibtex
@article{Umer2024DiTBEP,
  title={Utilizing Diffusion Models to model Magnetic Field Distribution in Medium Frequency Transformers},
  author={M. Umer},
  year={2024},
  institution={Eindhoven University of Technology},
  note={ID 1686909}
}
```

---

## Acknowledgements

Built on top of the [facebookresearch/DiT](https://github.com/facebookresearch/DiT) codebase. Diffusion utilities adapted from OpenAI's GLIDE, ADM, and IDDPM repositories.
