Diffusion Transformers for Magnetic Field Modeling in Medium-Frequency Transformers (MFTs)

Official implementation for the paper:

“Utilizing Diffusion Models to Model Magnetic Field Distribution in Medium Frequency Transformers” 

paper

Overview

This repository implements a conditional Diffusion Transformer (DiT) model to approximate the magnetic field distribution in the In-Window (IW) section of Medium-Frequency Transformers (MFTs).

Instead of using computationally expensive Finite Element Methods (FEM), this work demonstrates that a properly trained diffusion model can:

Achieve SSIM ≈ 0.92

Generate magnetic field distributions in ~8 seconds

Operate at approximately half the time of FEM (~16s)

The model learns to map:

MFT Structure Image  →  Magnetic Field Distribution Image

using a conditional diffusion framework.

Problem Setup

The dataset consists of paired 2D images:

Structure Image

2D slice of IW section of MFT

Contains winding positions and geometry

Magnetic Field Image

Corresponding FEM-computed magnetic field distribution

Dataset characteristics:

1000 image pairs

80 / 10 / 10 train/val/test split

Images resized to 256×256 or 512×512

Magnetic field images generated using FEM (~16s per sample)

Architecture

The model consists of:

1️⃣ VAE Encoder (Pretrained)

stabilityai/sd-vae-ft-mse or ema

Converts images into latent space

2️⃣ Conditional Diffusion Transformer (DiT)

Transformer backbone instead of U-Net

Patchified latent tokens (patch size = 2)

Sinusoidal timestep embedding

Conditioning via:

ResNet-18 embedding of structure image

adaLN-Zero conditioning

Classifier-free guidance

3️⃣ Loss

Log-cosh loss (found to be most stable)

Linear beta scheduler

AdamW optimizer

DiT Architectures Used
Model	Layers	Hidden Size	Heads
DiT-S	12	384	6
DiT-S+	12	768	6
DiT-B	12	768	12
DiT-L	24	1024	16
DiT-XS (Final)	6	256	4
Final Model (DiT-XS)

SSIM: 0.92

PSNR: 22.52 dB

LPIPS: 0.050

Generation time: ~8 seconds

Hardware Used

Training performed on:

NVIDIA A40 (256×256 images)

NVIDIA A100 PCIe (512×512 images)

Inference speed varies with model size.

Installation

Clone the repository:

git clone <your_repo_url>
cd <repo_name>

Create environment (example):

conda create -n mft-diffusion python=3.10
conda activate mft-diffusion
pip install -r requirements.txt
Running on RunPod (Recommended Setup)

This repository was primarily used in a RunPod.io GPU Jupyter Notebook environment.

Workflow:

Launch GPU pod (A40 or A100 recommended)

Clone repository inside /workspace

Upload dataset folder into:

/workspace/BEP256

The dataset loader automatically reads from:

ImgDataset(r'/workspace/BEP256')
Dataset Format

Your dataset directory should look like:

/workspace/BEP256/
    sample1_structure.png
    sample1_field.png
    sample2_structure.png
    sample2_field.png
    ...

The ImgDataset class loads image pairs.

To test dataset loading:

if __name__ == '__main__':
    img1, img2 = next(iter(ImgDataset(r'/workspace/BEP256')))
    plt.imshow(img1.permute(1, 2, 0))
    plt.show()
    plt.imshow(img2.permute(1, 2, 0))
    plt.show()
Training

Training is performed using PyTorch Lightning.

Example:
python train.py \
    --model DiT_Clipped \
    --image-size 256 \
    --epochs 150 \
    --global-batch-size 32 \
    --precision fp16
Key Training Parameters

Loss: Log-cosh

Beta schedule: Linear

Learning rate: 9e-4

LR scheduler: Exponential (gamma=0.97)

Batch size: 32

Sampling steps: 25

What Happens During Training

Structure image embedded using ResNet-18

Field image encoded via VAE

Noise added (forward diffusion)

DiT predicts noise

Log-cosh loss computed

Backpropagation using AdamW

Validation metrics logged

Checkpoints saved every 5 epochs

Losses saved in:

losses.csv
loss_plot.png
Sampling / Inference

After training:

Start from Gaussian noise

Condition on structure image

Run 25 reverse diffusion steps

Decode via VAE

Compute metrics

Evaluation metrics:

SSIM

PSNR

LPIPS

Results Summary
Model	SSIM	PSNR (dB)	LPIPS	Generation Time
DiT-S	0.94	31.94	0.037	25s
DiT-L	0.96	32.67	0.025	105s
DiT-XS	0.92	22.52	0.050	8s

Tradeoff:

Larger models → higher accuracy

Smaller models → faster than FEM

Key Contribution

This work demonstrates that diffusion models:

Can approximate electromagnetic FEM simulations

Provide practical speed-accuracy tradeoffs

Open the door to AI-assisted engineering design

Future Work

LCM-LoRA acceleration

FP4 quantization

Richer training datasets

Extension to 3D MFT modeling

Application to other electromagnetic problems

Citation

If you use this repository, cite:

M. Umer,
"Utilizing Diffusion Models to Model Magnetic Field Distribution 
in Medium Frequency Transformers",
2024.
License

This repository is for research purposes.
