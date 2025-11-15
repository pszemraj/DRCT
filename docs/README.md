# DRCT Documentation

## What This Project Is

DRCT (Dense Residual Connected Transformer) is an image super-resolution model that addresses the **information bottleneck** problem in Swin Transformer-based SR methods. While CNN-based approaches like RDN and ESRGAN successfully use dense connections to preserve information flow, most Swin-based methods (HAT, CAT, DAT) focus on novel attention mechanisms but overlook information loss in deep networks.

DRCT re-introduces dense connections to the Swin Transformer architecture, achieving state-of-the-art performance while remaining more lightweight than competing methods (14M parameters vs HAT's 21M for comparable quality).

This implementation operates at typical SR scales (2x, 3x, 4x) on standard benchmark datasets (Set5, Set14, BSD100, Urban100, Manga109) and real-world images.

## What's Actually Here

**Core Implementation:**
- **DRCT architecture** (`drct/archs/DRCT_arch.py`): Swin Transformer blocks with dense connections via RDG (Residual Dense Group) modules
- **Channel Attention Block (CAB)**: Feature compression and recalibration
- **Window-based multi-head self-attention**: Shifted window mechanism from Swin Transformer
- **Pixel shuffle upsampler**: Efficient sub-pixel convolution for upscaling

**Training Infrastructure:**
- `drct/models/drct_model.py`: BasicSR-based training model with tile processing support
- `drct/models/realdrctgan_model.py`: GAN-based model for real-world image SR
- `drct/models/realdrctmse_model.py`: MSE-based model for real-world image SR
- `drct/data/imagenet_paired_dataset.py`: Dataset loader for paired LR/HR images
- `drct/train.py`: Training entry point using BasicSR's distributed training pipeline
- `drct/test.py`: Testing/validation entry point

**Inference Scripts:**
- `inference.py`: Basic inference with autocast support, tile mode, optional torch.compile
- `efficient_inference.py`: **Production-grade inference** with extensive optimizations (channels_last memory format, mixed precision, batched tiles, CUDA stream overlap, efficient padding)
- `onnx_inference.py`: ONNX export and inference for deployment
- `predict.py`: Cog predictor wrapper (legacy HAT reference, not functional)

**Supporting Modules:**
- `drct/archs/srvgg_arch.py`: SRResNet variant (SRVGG) architecture
- `drct/archs/discriminator_arch.py`: UNet-style discriminator for GAN training

## What's Notably NOT Here

- **No training/test configuration files**: The `options/` directory does not exist in this repository (referenced in README but missing)
- **No pre-trained model weights**: The `experiments/pretrained_models/` directory is empty
- **No evaluation harness**: No standalone evaluation scripts for computing PSNR/SSIM metrics
- **No deployment code**: No production serving infrastructure (FastAPI, TorchServe, etc.)
- **No data preprocessing scripts**: Assumes pre-prepared LR/HR image pairs
- **No x2 pretraining path**: Direct x4 training from scratch or ImageNet pretraining only
- **No MambaDRCT**: Mentioned in main README as abandoned due to slow training

## Why It Matters

DRCT demonstrates that **architectural simplicity + dense connections** can outperform complex attention mechanisms:

1. **Information preservation**: Dense connections prevent feature degradation in deep networks (12 RDG blocks)
2. **Efficiency**: Achieves HAT-level quality with 32% fewer parameters and 48% fewer FLOPs
3. **Practical performance**: `efficient_inference.py` provides production-ready optimizations (3-5x speedup with torch.compile, mixed precision, and batching)

**Divergence from standard patterns:**
- Unlike vanilla SwinIR, DRCT concatenates features from all previous blocks within each RDG
- Unlike HAT/CAT/DAT, DRCT uses simple channel attention instead of sophisticated spatial attention variants
- Unlike BasicSR's typical usage, the optimized inference path bypasses BasicSR entirely for speed

This implementation is research code adapted for practical use. The `efficient_inference.py` script contains significant engineering work beyond the original paper, including buffer management fixes for torch.compile compatibility and extensive performance tuning.
