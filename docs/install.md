# Installation & Environment

This document covers installation requirements, setup procedures, and environment configuration for DRCT.

## Quick Start (CPU-Only)

For CPU-only inference or testing without GPU:

```bash
# Install build tools
pip install ninja

# Install PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Clone repository
git clone https://github.com/pszemraj/DRCT.git
cd DRCT

# Install DRCT package
pip install -e .
```

**Note:** CPU inference is **extremely slow** (10-100× slower than GPU). Only recommended for testing or development without GPU access.

## Recommended Installation (GPU)

### Prerequisites

**System requirements:**
- **Operating System:** Linux (Ubuntu 20.04+, CentOS 7+), Windows 10/11, macOS (limited GPU support)
- **GPU:** NVIDIA GPU with CUDA support
  - Minimum: GTX 1060 (6GB VRAM) for basic inference
  - Recommended: RTX 3060+ (12GB VRAM) for optimal performance
  - Training: RTX 3090 / A100 (24GB+ VRAM)
- **CUDA:** Version 11.7 or later (12.1+ for torch.compile)
- **cuDNN:** Version 8.5 or later

**Software requirements:**
- Python 3.7 - 3.11 (3.10 recommended)
- pip 21.0+
- git

### Installation Steps

1. **Verify CUDA installation:**

```bash
nvcc --version  # Should show CUDA 11.7+
nvidia-smi      # Should show driver version and GPUs
```

If CUDA not installed, follow [NVIDIA CUDA Toolkit installation guide](https://developer.nvidia.com/cuda-downloads).

2. **Create virtual environment (recommended):**

```bash
# Using venv
python3 -m venv drct-env
source drct-env/bin/activate  # On Windows: drct-env\Scripts\activate

# OR using conda
conda create -n drct python=3.10
conda activate drct
```

3. **Install PyTorch with CUDA support:**

Check [PyTorch Get Started](https://pytorch.org/get-started/locally/) for latest version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**IMPORTANT:** Avoid PyTorch 1.8 and 1.12 (known performance issues mentioned in README).

Recommended versions:
- PyTorch 2.0+ (for torch.compile support)
- PyTorch 1.13.1 (stable, well-tested with BasicSR)

4. **Install BasicSR:**

```bash
pip install basicsr==1.3.4.9
```

**Critical:** Version must be exactly 1.3.4.9 for compatibility with DRCT training/testing infrastructure.

5. **Clone and install DRCT:**

```bash
git clone https://github.com/pszemraj/DRCT.git
cd DRCT
pip install -e .
```

**Alternative (without editable install):**
```bash
pip install -e ".[dev]"  # Includes development dependencies (cython, numpy)
```

6. **Verify installation:**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import drct; from drct.archs.DRCT_arch import DRCT; print('DRCT imported successfully')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA available: True
DRCT imported successfully
```

## Dependencies Explained

### Core Dependencies (Required)

Defined in `pyproject.toml`:

1. **torch >= 1.7:**
   - Deep learning framework
   - Provides tensor operations, autograd, CUDA support
   - Version constraints: Avoid 1.8 and 1.12 (performance regressions)

2. **basicsr == 1.3.4.9:**
   - Basic Super-Resolution framework
   - Provides training infrastructure, data loaders, metrics, model registry
   - **Version locked:** DRCT training depends on specific BasicSR APIs

3. **einops:**
   - Tensor manipulation library
   - **Note:** Listed as dependency but not actually used in code (may be vestigial)

### Inference-Only Dependencies

If only using inference scripts (not training), additional packages needed:

```bash
pip install opencv-python numpy tqdm
```

- **opencv-python (cv2):** Image I/O, preprocessing
- **numpy:** Numerical operations, array manipulation
- **tqdm:** Progress bars

### Optional Dependencies

For advanced features:

```bash
# ONNX export/inference
pip install onnx onnxruntime onnxruntime-gpu

# Development tools (included in [dev] extras)
pip install cython numpy
```

### Full Dependency List

From imports in codebase:

| Package | Purpose | Required For |
|---------|---------|--------------|
| torch | Core framework | All operations |
| torchvision | Image transforms (normalize) | Training, inference |
| basicsr | Training/testing infrastructure | Training, testing |
| opencv-python | Image I/O | Inference, data loading |
| numpy | Array operations | All operations |
| tqdm | Progress bars | Inference scripts |
| einops | Tensor manipulation | Listed but unused |
| onnx | Model export | ONNX inference |
| onnxruntime | ONNX execution | ONNX inference |
| cython | Build acceleration | Development (optional) |

## Hardware Configurations

### Minimum (Inference Only)

- **GPU:** GTX 1060 6GB or equivalent
- **VRAM:** 6GB
- **System RAM:** 8GB
- **Storage:** 5GB (model weights + code)

**Configuration:**
```bash
python inference.py input --tile 256 --tile_overlap 16
```

Expected performance: ~5-10 seconds per 1080p image

### Recommended (Optimized Inference)

- **GPU:** RTX 3060 12GB or better
- **VRAM:** 12GB
- **System RAM:** 16GB
- **Storage:** 10GB

**Configuration:**
```bash
python efficient_inference.py input \
    --tile 512 --tile_batch_size 2 --precision bf16 --compile reduce
```

Expected performance: ~2-3 seconds per 1080p image (after warmup)

### High-End (Training)

- **GPU:** RTX 3090 / A100 or multi-GPU setup
- **VRAM:** 24GB per GPU
- **System RAM:** 64GB+
- **Storage:** 500GB+ (datasets + checkpoints)

**Configuration:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
    drct/train.py -opt options/train/config.yml --launcher pytorch
```

### VRAM Requirements by Task

| Task | Tile Size | Precision | Batch | Est. VRAM |
|------|-----------|-----------|-------|-----------|
| Inference (min) | 256 | FP32 | 1 | 4GB |
| Inference (opt) | 512 | BF16 | 2 | 8GB |
| Inference (max) | 1024 | BF16 | 1 | 10GB |
| Training | 256 (crop) | FP32 | 4 | 16GB |
| Training (large) | 256 (crop) | FP32 | 8 | 24GB |

## Environment Variables

### PyTorch Configuration

```bash
# Enable TF32 on Ampere GPUs (automatically set by scripts)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 only
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

# Control number of CPU threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### DRCT-Specific Variables

```bash
# Skip torch.compile dry-run validation (efficient_inference.py)
export SKIP_DRY_RUN=1

# Enable verbose torch.compile logging (efficient_inference.py)
export TORCH_LOGS="+dynamo,graph_breaks,guards,recompiles"

# Control inductor debug output
export TORCHINDUCTOR_DEBUG=1
```

### BasicSR Configuration (Training)

```bash
# Disable wandb logging (if not using)
export WANDB_MODE=disabled

# Set BasicSR cache directory
export BASICSR_CACHE_DIR=/path/to/cache
```

## Docker Installation (Recommended for Deployment)

**Dockerfile example:**

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install DRCT
WORKDIR /app
RUN git clone https://github.com/pszemraj/DRCT.git . \
    && pip3 install -e .

# Download model weights (replace with actual URL)
RUN mkdir -p experiments/pretrained_models && \
    wget -O experiments/pretrained_models/DRCT-L_X4.pth \
    https://example.com/DRCT-L_X4.pth

ENTRYPOINT ["python3", "efficient_inference.py"]
```

**Build and run:**
```bash
docker build -t drct:latest .
docker run --gpus all -v /path/to/images:/input -v /path/to/output:/output \
    drct:latest /input --output /output
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

**Additional system dependencies for OpenCV:**
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

**For building from source (optional):**
```bash
sudo apt-get install build-essential python3-dev
```

### Windows

**Install Visual Studio Build Tools** (required for some dependencies):
- Download from [Visual Studio](https://visualstudio.microsoft.com/downloads/)
- Select "Desktop development with C++"

**Use Anaconda** (easier dependency management):
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge opencv
pip install basicsr==1.3.4.9
pip install -e .
```

### macOS

**Limited GPU support:**
- Apple Silicon (M1/M2): MPS backend support in PyTorch 2.0+
- Intel Macs: CPU-only (no NVIDIA CUDA)

**Installation:**
```bash
# For Apple Silicon
pip install torch torchvision

# Test MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Note:** Performance on Apple Silicon MPS is significantly slower than NVIDIA CUDA.

## Troubleshooting

### "CUDA out of memory" (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce tile size: `--tile 256` → `--tile 128`
2. Reduce batch size: `--tile_batch_size 4` → `--tile_batch_size 1`
3. Use lower precision: `--precision bf16` or `--precision fp16`
4. Disable compilation: `--compile off`
5. Close other GPU applications

### "ModuleNotFoundError: No module named 'basicsr'"

**Cause:** BasicSR not installed or wrong version

**Solution:**
```bash
pip install basicsr==1.3.4.9
```

### "Torch.compile failed" or "Graph breaks detected"

**Cause:** PyTorch version < 2.0 or compilation incompatibility

**Solutions:**
1. Upgrade PyTorch: `pip install --upgrade torch`
2. Disable compilation: `--compile off`
3. Set `export SKIP_DRY_RUN=1` to skip validation

### "shape mismatch" errors during inference

**Cause:** Tile size not multiple of window_size (16)

**Solution:** Use tile sizes: 256, 512, 768, 1024, etc.

### Slow performance on GPU

**Checks:**
1. Verify GPU is being used:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Check GPU utilization: `nvidia-smi` (should show ~80-95% GPU usage)
3. Enable optimizations:
   ```bash
   python efficient_inference.py input --precision bf16 --compile reduce
   ```

### "version mismatch" when loading checkpoint

**Cause:** Checkpoint saved with different PyTorch/CUDA version

**Solutions:**
- Load with `weights_only=True`: `torch.load(path, weights_only=True)`
- Use `map_location='cpu'` then move to GPU: `torch.load(path, map_location='cpu')`
- Try different PyTorch version

## Version Compatibility Matrix

| Component | Minimum | Recommended | Maximum Tested |
|-----------|---------|-------------|----------------|
| Python | 3.7 | 3.10 | 3.11 |
| PyTorch | 1.7 | 2.1 | 2.2 |
| CUDA | 11.1 | 12.1 | 12.3 |
| BasicSR | 1.3.4.9 | 1.3.4.9 | 1.3.4.9 |
| NVIDIA Driver | 470+ | 520+ | 545+ |

**Notes:**
- **PyTorch 1.8 and 1.12:** Avoid (performance issues)
- **PyTorch 2.0+:** Required for torch.compile
- **CUDA 12.1+:** Best performance with latest PyTorch
- **BasicSR:** Locked at 1.3.4.9 (do not upgrade)

## Verification Checklist

After installation, verify functionality:

```bash
# 1. Check Python environment
python --version  # Should be 3.7-3.11

# 2. Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Check DRCT installation
python -c "from drct.archs.DRCT_arch import DRCT; print('DRCT OK')"

# 4. Check dependencies
python -c "import cv2, numpy, tqdm, basicsr; print('Dependencies OK')"

# 5. Test inference (requires model weights)
python inference.py --help  # Should show CLI options
```

All checks should pass before proceeding to inference or training.
