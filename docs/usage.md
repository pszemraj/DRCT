# Usage

This document covers practical usage of DRCT for inference, training, and testing.

## Inference

### Minimal Viable Example

**Prerequisites:**
- Installed DRCT package (`pip install -e .`)
- Downloaded pretrained model weights (DRCT-L_X4.pth)
- Input images in a directory

**Simplest invocation:**
```bash
python inference.py /path/to/input/images --model_path experiments/pretrained_models/DRCT-L_X4.pth
```

**What happens:**
1. Loads all images from input directory (jpg, png, bmp, tiff, webp)
2. Upscales each image 4x using DRCT-L
3. Saves results to `{input_dir}/upscaled-DRCT-outputs/`
4. Uses tile mode (256x256 tiles with 16px overlap) by default
5. Applies mixed precision (autocast) if CUDA available

**Python API example:**
```python
import torch
import cv2
import numpy as np
from drct.archs.DRCT_arch import DRCT

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DRCT(
    upscale=4, in_chans=3, img_size=64, window_size=16,
    compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
    overlap_ratio=0.5, img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    gc=32, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv"
)
checkpoint = torch.load("path/to/DRCT-L_X4.pth", weights_only=True)
model.load_state_dict(checkpoint["params"], strict=True)
model.eval()
model = model.to(device)

# Process image
img = cv2.imread("input.jpg", cv2.IMREAD_COLOR).astype(np.float32) / 255.0
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
img = img.unsqueeze(0).to(device)

with torch.inference_mode(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
    output = model(img)

# Save result
output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round().astype(np.uint8)
cv2.imwrite("output.jpg", output)
```

## Typical Workflows

### 1. Batch Inference (Production)

**For maximum speed with modern GPU:**
```bash
python efficient_inference.py /path/to/images \
    --model_path weights/DRCT-L_X4.pth \
    --output /path/to/output \
    --tile 512 \
    --tile_overlap 32 \
    --tile_batch_size 4 \
    --streams 2 \
    --precision bf16 \
    --compile reduce \
    --jpeg_quality 95
```

**Explanation:**
- `--tile 512`: Process in 512x512 tiles (balances speed and quality)
- `--tile_batch_size 4`: Process 4 tiles simultaneously (requires ~8GB VRAM)
- `--streams 2`: Overlap H2D transfer with computation
- `--precision bf16`: Use bfloat16 for 2x memory savings (Ampere+ GPUs)
- `--compile reduce`: Use torch.compile with reduce-overhead mode
- `--jpeg_quality 95`: High-quality output

**Memory-constrained systems (4GB VRAM):**
```bash
python efficient_inference.py /path/to/images \
    --model_path weights/DRCT-L_X4.pth \
    --tile 256 \
    --tile_batch_size 1 \
    --precision bf16
```

### 2. Whole-Image Inference (No Tiling)

**For small images or abundant VRAM:**
```bash
python inference.py /path/to/images \
    --model_path weights/DRCT-L_X4.pth \
    --tile -1
```

**Constraints:**
- Image dimensions must be ≤ 512x512 (approx) for 8GB VRAM
- No tile seams, slightly better quality
- Faster for small images (no tile stitching overhead)

### 3. ONNX Inference

**Convert PyTorch model to ONNX (one-time):**
```bash
python onnx_inference.py /path/to/images \
    --model_path weights/DRCT-L_X4.pth \
    --onnx_path weights/DRCT-L_X4.onnx
```

The script automatically exports to ONNX if the ONNX file doesn't exist or is older than the PyTorch checkpoint.

**Run inference with ONNX Runtime:**
```bash
python onnx_inference.py /path/to/images \
    --onnx_path weights/DRCT-L_X4.onnx
```

**Benefits:**
- Faster cold-start time (no PyTorch JIT overhead)
- Deployment on ONNX Runtime (CPP, mobile, edge devices)
- Potential TensorRT optimization

**Limitations:**
- Fixed tile size (must match export dummy input)
- Less flexible than PyTorch for dynamic shapes

### 4. Training (Requires External Config)

**Basic training from scratch:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=4321 \
    drct/train.py \
    -opt options/train/train_DRCT_SRx4_from_scratch.yml \
    --launcher pytorch
```

**Prerequisites:**
- YAML config file in `options/train/` (not included in repo)
- Training dataset (DF2K, ImageNet, DIV2K, etc.)
- Validation dataset (Set5, Set14, etc.)

**Config file must specify:**
```yaml
name: DRCT_SRx4
model_type: DRCTModel
scale: 4
num_gpu: 8

datasets:
  train:
    name: DF2K
    type: ImageNetPairedDataset
    dataroot_gt: /path/to/DF2K/HR
    io_backend:
      type: disk
    gt_size: 256
    use_hflip: true
    use_rot: true

network_g:
  type: DRCT
  upscale: 4
  in_chans: 3
  img_size: 64
  window_size: 16
  depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  # ... more params

train:
  optim_g:
    type: Adam
    lr: !!float 2e-4
  scheduler:
    type: MultiStepLR
    milestones: [250000, 400000, 450000, 475000]
  total_iter: 500000
```

**Resume training:**
```bash
python -m torch.distributed.launch ... \
    drct/train.py -opt options/train/config.yml \
    --auto_resume
```

### 5. Testing/Evaluation

**Evaluate on benchmark dataset:**
```bash
python drct/test.py -opt options/test/DRCT_SRx4_Set5.yml
```

**Config file must specify:**
```yaml
name: DRCT_SRx4_Set5
model_type: DRCTModel
scale: 4
num_gpu: 1

datasets:
  test_1:
    name: Set5
    type: PairedImageDataset
    dataroot_gt: datasets/Set5/HR
    dataroot_lq: datasets/Set5/LR_bicubic/X4
    io_backend:
      type: disk

network_g:
  type: DRCT
  # ... architecture params

path:
  pretrain_network_g: experiments/pretrained_models/DRCT_SRx4.pth
  visualization: results/DRCT_SRx4_Set5/visualization

val:
  save_img: true
  suffix: ~
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: true
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: true
```

**Output:**
- Metrics printed to console and log files
- Visualizations saved to `results/{name}/visualization/{dataset_name}/`

## Configuration and Customization

### Key Model Parameters

**Controlled in inference scripts (hardcoded):**

| Parameter | DRCT-L Value | Effect |
|-----------|--------------|--------|
| `upscale` | 4 | Super-resolution scale factor (2, 3, 4) |
| `embed_dim` | 180 | Feature dimension (96=DRCT, 180=DRCT-L, 210=DRCT-XL) |
| `depths` | [6]*12 | Number of blocks per stage (12 RDG blocks) |
| `num_heads` | [6]*12 | Attention heads per block |
| `window_size` | 16 | Local attention window size (must divide tile_size) |
| `gc` | 32 | Growth channel for dense connections |
| `compress_ratio` | 3 | CAB channel compression ratio |
| `squeeze_factor` | 30 | Channel attention squeeze ratio |
| `mlp_ratio` | 2 | MLP expansion ratio (2 for DRCT-L, 4 for DRCT) |

**To use DRCT (base) instead of DRCT-L:**

Modify `inference.py` or `efficient_inference.py`:
```python
model = DRCT(
    upscale=4,
    embed_dim=96,  # Changed from 180
    depths=[6, 6, 6, 6, 6, 6],  # Changed from [6]*12
    num_heads=[6, 6, 6, 6, 6, 6],  # Changed from [6]*12
    mlp_ratio=4,  # Changed from 2
    # ... other params unchanged
)
```

**To use 2x or 3x upscaling:**
- Change `upscale=2` or `upscale=3`
- Use corresponding pretrained weights (DRCT_SRx2.pth, DRCT_SRx3.pth)
- Update `--scale` CLI argument to match

### Tile Processing Settings

**`--tile N`**: Tile size in pixels (default: 256)
- Must be multiple of `window_size` (16)
- Larger = better quality (no seams) but more VRAM
- Recommended: 256 (safe), 512 (optimal), 1024 (high-end GPUs)
- Set to -1 to disable tiling (process whole image)

**`--tile_overlap N`**: Overlap between tiles in pixels (default: 16)
- Reduces visible seams at tile boundaries
- Higher = better blending but slower
- Recommended: 16-32

**`--tile_batch_size N`**: Process N tiles in parallel (default: 1)
- Only in `efficient_inference.py`
- Increases throughput but requires N× more VRAM
- Recommended: 1 (safe), 2-4 (if VRAM available)

### Precision Settings

**`--precision {fp32, fp16, bf16}`**: Numerical precision (efficient_inference.py only)
- Auto-detected if not specified
- `bf16`: Best for Ampere+ GPUs (A100, RTX 30/40 series), 2x memory savings, negligible quality loss
- `fp16`: Best for older GPUs (V100, RTX 20 series), 2x memory savings, rare artifacts
- `fp32`: Full precision, highest quality, 2x memory usage

**Trade-offs:**
- Quality: fp32 > bf16 ≈ fp16 (differences imperceptible in most cases)
- Speed: bf16 ≈ fp16 > fp32 (1.5-2x faster on tensor cores)
- Memory: bf16 = fp16 < fp32 (50% savings)

### Compilation Settings

**`--compile {off, default, reduce, max}`**: torch.compile mode (efficient_inference.py only)
- `off`: No compilation (default for compatibility)
- `default`: Standard torch.compile optimization
- `reduce`: Reduce-overhead mode (recommended, 2-3x speedup)
- `max`: Max-autotune mode (aggressive optimization, 3-5x speedup, longer compile time)

**Trade-offs:**
- First image: Compilation overhead (~30-120 seconds warmup)
- Subsequent images: 2-5x faster inference
- Use if processing ≥10 images with same dimensions

## Limits and Edge Cases

### Known Constraints

1. **Input size constraints:**
   - Minimum: No hard limit, but <64px may produce artifacts
   - Maximum: Limited by VRAM (16K×16K with tile mode, ~2K×2K without)
   - Dimensions must be multiples of `window_size` (16) after padding (handled automatically)

2. **Tile mode limitations:**
   - Visible seams possible with `--tile_overlap 0`
   - Tile boundaries may have slight brightness/color shifts
   - Smaller tiles = more seams but less VRAM

3. **Model checkpoint compatibility:**
   - Checkpoints contain `"params"` dict key (BasicSR format)
   - Loading with `strict=False` due to buffer handling differences
   - Older checkpoints may be missing `mean` buffer (handled gracefully)

4. **Channel attention limitations:**
   - Fixed squeeze factor (30) may not be optimal for all feature dimensions
   - No learned adjustment of squeeze ratio

5. **Window size restrictions:**
   - Fixed at 16 (hardcoded)
   - Tile size must be multiple of 16
   - Changing window size requires retraining

### Sharp Edges

1. **`--skip_completed` flag:**
   - Only checks if output file exists, not if it's valid
   - Will skip corrupted outputs
   - Filename format: `{imgname}_DRCT-L_X{scale}.jpg` (inference.py) or `{imgname}.jpg` (efficient_inference.py)

2. **`--compile` modes:**
   - May fail silently and fall back to eager mode
   - Incompatible with dynamic shapes (different image sizes)
   - Requires CUDA 11.7+ and PyTorch 2.0+
   - Warmup penalty on first image (~30-120s)

3. **Memory management:**
   - No automatic VRAM detection
   - OOM errors crash the script (no graceful degradation)
   - Use `--tile_batch_size 1` if encountering OOM

4. **Color space:**
   - Input assumed RGB (via cv2.imread BGR→RGB conversion)
   - No support for grayscale or RGBA
   - Transparency channel discarded if present

5. **Output format:**
   - Always JPEG (lossy compression)
   - Quality controlled by `--jpeg_quality` (default: 90)
   - No PNG or lossless output option in CLI scripts

### Partially Implemented Features

1. **Flash Attention:**
   - Mentioned in PERFORMANCE.md
   - Not explicitly used in DRCT_arch.py (uses standard attention)
   - Would require switching to `F.scaled_dot_product_attention`

2. **Hann window blending:**
   - Mentioned in PERFORMANCE.md as future work
   - Current implementation uses simple averaging (`E / W`)
   - No windowing function applied to tiles

3. **CUDA graphs:**
   - Mentioned in PERFORMANCE.md as future optimization
   - Not implemented in current code

4. **Automatic tile sizing:**
   - Mentioned in PERFORMANCE.md as future work
   - Currently requires manual `--tile` specification

5. **Config flags that do nothing:**
   - `predict.py` is non-functional (references HAT paths)
   - Some BasicSR options may not affect DRCT (e.g., `use_checkpoint` not used in forward pass)

### Testing Without Ground Truth

**Inference-only mode (no metrics):**
```bash
python drct/test.py -opt options/test/DRCT_SRx4_ImageNet-LR.yml
```

Config should omit `dataroot_gt` and `metrics`:
```yaml
datasets:
  test_1:
    name: custom
    type: SingleImageDataset  # Not PairedImageDataset
    dataroot_lq: /path/to/input/images
    io_backend:
      type: disk

val:
  save_img: true
  suffix: ~
  # No metrics section
```

This mode only produces visualizations, no PSNR/SSIM computation.
