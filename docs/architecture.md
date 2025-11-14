# Architecture

This document describes DRCT's system architecture from the code's perspective, following execution paths from entry points through the model.

## Entry Points and Execution Flow

### Inference Path (Production)

**Entry:** `efficient_inference.py` → `main()`

1. **Argument parsing** (lines 1019-1066): Parses CLI arguments for input/output paths, model checkpoint, tile settings, precision, compilation mode
2. **GPU setup** (line 1072): `check_ampere_gpu()` enables TF32 on Ampere+ GPUs
3. **Precision selection** (lines 1075-1083): Auto-detects optimal precision (bf16 for Ampere+, fp16 otherwise, fp32 for CPU)
4. **Model instantiation** (lines 1086-1103): Constructs `DRCT` with hardcoded DRCT-L configuration
5. **Weight loading** (lines 1104-1106): Loads checkpoint dict with `strict=False` to handle missing buffers
6. **Memory format conversion** (line 1109): Converts model to `channels_last` for GPU efficiency
7. **Optional compilation** (lines 1111-1146): If `--compile` is set, runs `torch.compile` with mode (default/reduce-overhead/max-autotune), including dry-run validation and warmup
8. **Image processing loop** (lines 1154-1183):
   - Read image with cv2, convert BGR→RGB, normalize to [0, 1]
   - Convert to tensor, apply `channels_last` memory format
   - Pad to window_size multiple (16) using reflect padding
   - Call `test()` function for tile-based or whole-image inference
   - Crop output to original scaled dimensions
   - Save as JPEG with specified quality

**Data flow:**
```
cv2.imread → np.array (HWC, BGR, [0,1])
  → torch.Tensor (CHW, RGB, [0,1], channels_last)
  → F.pad (reflect mode)
  → DRCT.forward (with autocast for mixed precision)
  → crop → clamp → cpu → numpy
  → transpose RGB→BGR
  → cv2.imwrite
```

### Inference Path (Basic)

**Entry:** `inference.py` → `main()`

Similar to efficient path but simpler:
- No channels_last optimization
- No stream overlap
- No batched tile processing (sequential tiles only)
- Uses flip-concat padding instead of F.pad
- Single precision mode via autocast

### Training Path

**Entry:** `drct/train.py` → `train_pipeline(root_path)`

1. **Module imports**: Imports `drct.archs`, `drct.data`, `drct.models` to register components with BasicSR registries
2. **Delegates to BasicSR**: Calls `basicsr.train.train_pipeline(root_path)`
3. **BasicSR workflow** (not shown in this repo):
   - Loads YAML config from `options/train/*.yml`
   - Instantiates dataset via `DATASET_REGISTRY`
   - Instantiates model via `MODEL_REGISTRY`
   - Runs distributed training loop with validation callbacks

**Configuration dependency:** Requires external YAML files (not present in repo) specifying:
- Network architecture parameters
- Dataset paths and augmentation settings
- Optimizer, scheduler, loss functions
- Training iterations, validation intervals

### Testing/Validation Path

**Entry:** `drct/test.py` → BasicSR's test pipeline

1. Loads test configuration from `options/test/*.yml`
2. Instantiates `DRCTModel` which wraps the DRCT architecture
3. Calls `DRCTModel.nondist_validation()`:
   - Pads input to window_size multiple (`pre_process`)
   - Processes via `process()` or `tile_process()` based on config
   - Crops output (`post_process`)
   - Computes metrics (PSNR, SSIM) if ground truth available
   - Saves results to `results/{model_name}/visualization/{dataset_name}/`

## Module Responsibilities

### `drct/archs/DRCT_arch.py`

**Purpose:** Core network architecture.

**Key classes:**

- **`DRCT`** (lines 691-869): Main model class
  - Constructor builds: shallow feature extraction → deep feature extraction (RDG layers) → reconstruction → upsampling
  - `forward()`: Normalizes input → extracts features → applies residual connection → upsamples → denormalizes
  - Registered with `@ARCH_REGISTRY.register()`

- **`RDG`** (Residual Dense Group, lines 491-617): Dense connection block
  - Contains 5 `SwinTransformerBlock`s with progressively growing channel dimensions
  - Each block's output is concatenated to inputs of all subsequent blocks
  - Implements: `x1 → x2(x,x1) → x3(x,x1,x2) → x4(x,x1,x2,x3) → x5(x,x1,x2,x3,x4)`
  - Uses `adjust` 1x1 convs to compress concatenated features to growth channel count `gc`
  - Final output: `x5 * 0.2 + x` (residual learning with scaling)

- **`SwinTransformerBlock`** (lines 341-488): Window-based transformer block
  - Shifted window multi-head self-attention (W-MSA / SW-MSA)
  - MLP with GELU activation
  - Layer normalization and residual connections
  - Dynamic attention mask calculation for shifted windows

- **`WindowAttention`** (lines 252-338): Multi-head self-attention with relative position bias
  - Computes QKV projections
  - Applies learnable relative position bias table
  - Supports optional attention masks for shifted windows

- **`CAB`** (Channel Attention Block, lines 154-174): Feature recalibration
  - Compresses channels by `compress_ratio` (3), applies GELU, expands back
  - Applies channel attention via global average pooling + squeeze-excite

- **Helper modules:**
  - `PatchEmbed` / `PatchUnEmbed`: Converts between spatial (B,C,H,W) and sequence (B,H*W,C) representations
  - `Upsample`: Pixel shuffle upsampler (2^n or 3x scales)
  - `window_partition` / `window_reverse`: Spatial→window and window→spatial transformations

### `drct/models/drct_model.py`

**Purpose:** Training and testing wrapper for DRCT architecture.

**Key class: `DRCTModel(SRModel)`** (inherits from BasicSR's base SR model)

- **`pre_process()`** (lines 16-26): Pads input to window_size multiple using reflect mode
- **`process()`** (lines 28-38): Runs model inference in `eval()` mode with `torch.no_grad()`
- **`tile_process()`** (lines 40-128): Tile-based inference for memory-limited scenarios
  - Divides image into overlapping tiles
  - Processes each tile independently
  - Stitches tiles back into full output
  - Uses `tile_size` and `tile_pad` from config
- **`post_process()`** (lines 130-137): Removes padding from output
- **`nondist_validation()`** (lines 139-223): Validation loop for testing
  - Iterates through dataloader
  - Applies pre/process/post pipeline
  - Computes metrics if ground truth available
  - Saves visualizations to disk

### `drct/data/imagenet_paired_dataset.py`

**Purpose:** Dataset loader for paired LR/HR training data.

**Key class: `ImageNetPairedDataset(data.Dataset)`**

- **`__init__()`**: Sets up file paths from config, supports LMDB or directory backends
- **`__getitem__()`** (lines 39-95):
  1. Loads GT image from path
  2. Crops to scale multiple (modcrop)
  3. Resizes to at least `gt_size` using cv2.resize
  4. Generates LR image via `imresize(img_gt, 1/scale)` (bicubic downsampling)
  5. Applies training augmentations: random crop, horizontal flip, rotation
  6. Optional color space transform (RGB→Y channel only)
  7. Converts to tensor, normalizes if specified
  8. Returns dict: `{"lq": img_lq, "gt": img_gt, "gt_path": gt_path}`

**Note:** Assumes high-quality images only (no paired LR/HR); generates LR via bicubic downsampling.

### `drct/models/realdrctgan_model.py` and `realdrctmse_model.py`

**Purpose:** Real-world SR models (GAN-based and MSE-based).

These are thin wrappers around BasicSR's `RealESRGANModel` and base model classes, registered with `@MODEL_REGISTRY.register(name='RealDRCTGANModel')` and `@MODEL_REGISTRY.register(name='RealDRCTMSEModel')`. Implementation details delegate to BasicSR's training infrastructure.

## Key Patterns and Abstractions

### Registry Pattern

All components use BasicSR's registry system:
- `ARCH_REGISTRY.register()`: Architectures (DRCT, SRVGG, discriminators)
- `MODEL_REGISTRY.register()`: Training models (DRCTModel, RealDRCT variants)
- `DATASET_REGISTRY.register()`: Datasets (ImageNetPairedDataset)

Modules are auto-imported via `drct/{archs,models,data}/__init__.py` which scans for `*_arch.py`, `*_model.py`, `*_dataset.py` files.

### Window-Based Processing

DRCT uses shifted window attention (from Swin Transformer):

1. Input tensor (B, H, W, C) is partitioned into non-overlapping windows of size `window_size × window_size`
2. Self-attention computed within each window independently
3. For every other layer, windows are shifted by `window_size // 2` to enable cross-window connections
4. Attention masks prevent information leakage across shifted boundaries

**Code path:**
```
x (B,H,W,C) → window_partition → (num_windows*B, ws, ws, C)
  → flatten → (num_windows*B, ws²,C)
  → WindowAttention.forward (with optional mask)
  → reshape → (num_windows*B, ws, ws, C)
  → window_reverse → (B,H,W,C)
```

### Dense Connections in RDG

Each RDG block implements dense connectivity:

```python
# drct/archs/DRCT_arch.py, RDG.forward (lines 611-617)
x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, xsize), xsize))))
x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize))))
x5 = self.pe(self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)))
return x5 * 0.2 + x
```

**Pattern:** Each Swin block receives concatenation of all previous outputs, then a 1x1 conv (`adjust`) reduces channels to `gc` (growth channel, default 32). This mirrors DenseNet's pattern but applied to transformer blocks.

### Tile-Based Inference

For large images, both `inference.py` and `efficient_inference.py` support tile mode:

**`inference.py` approach (sequential):**
```python
# Lines 226-262
for h_idx in h_idx_list:
    for w_idx in w_idx_list:
        in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
        out_patch = model(in_patch)
        E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
        W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(1)
output = E.div_(W)
```

**`efficient_inference.py` approach (batched):**
```python
# Lines 977-1010
coords = [(y, x) for y in h_idx_list for x in w_idx_list]
for i in range(0, len(coords), args.tile_batch_size):
    ysxs = coords[i:i+args.tile_batch_size]
    batch = torch.cat([img_lq[..., y:y+tile, x:x+tile] for (y,x) in ysxs], dim=0)
    outs = model(batch)  # Process multiple tiles in one forward pass
    for j, (y, x) in enumerate(ysxs):
        E[..., y*sf:(y+tile)*sf, x*sf:(x+tile)*sf] += outs[j]
        W[..., y*sf:(y+tile)*sf, x*sf:(x+tile)*sf] += 1.0
output = E / W
```

Batched approach reduces Python overhead and improves GPU utilization.

## Configuration and Wiring

### Hardcoded vs Configurable

**Hardcoded in inference scripts:**
- All DRCT-L architecture parameters (embed_dim=180, depths=[6]*12, num_heads=[6]*12, gc=32, etc.)
- Window size (16)
- Image range normalization (1.0)
- RGB mean for normalization (0.4488, 0.4371, 0.4040)

**Configurable via CLI:**
- Input/output paths
- Tile size and overlap
- Precision (fp32/fp16/bf16)
- JPEG quality
- Compilation mode
- Skip completed images flag

**Configurable via YAML (training/testing only):**
- Network architecture type (DRCT, SRVGG, etc.)
- Dataset paths and augmentation
- Training hyperparameters
- Validation metrics

### Component Instantiation

**Inference (direct instantiation):**
```python
model = DRCT(upscale=4, in_chans=3, img_size=64, window_size=16, ...)
model.load_state_dict(torch.load(path)["params"], strict=True)
```

**Training (registry-based):**
```yaml
# In YAML config (not present in repo)
network_g:
  type: DRCT
  upscale: 4
  in_chans: 3
  # ... other params
```
BasicSR calls `ARCH_REGISTRY.get('DRCT')(**network_g)` to instantiate.

## Dependencies and Constraints

### Framework Dependencies

- **PyTorch >= 1.7** (avoid 1.8 and 1.12 per README warnings)
- **BasicSR == 1.3.4.9**: Provides training infrastructure, data utilities, metrics
- **einops**: Tensor manipulation (imported but not used in main codebase)
- **opencv-python (cv2)**: Image I/O and preprocessing
- **numpy**: Numerical operations
- **tqdm**: Progress bars

### Hardware Assumptions

- **GPU optional but recommended**: All scripts support CPU fallback but are extremely slow
- **Ampere+ GPUs (compute capability 8.0+)**: Enable TF32 acceleration automatically
- **CUDA 11+**: Required for torch.compile and modern mixed precision features
- **VRAM requirements:**
  - Minimum (tile=256, fp32): ~4GB
  - Recommended (tile=512, bf16): ~8GB
  - Optimal (tile=1024, bf16, batched): ~12-16GB

### Environment Variables

- **`TORCH_LOGS`**: Set to enable torch._logging output (used in efficient_inference.py header)
- **`SKIP_DRY_RUN`**: Set to "1" to skip torch.compile dry-run validation

### File Format Constraints

- **Input images**: JPEG, PNG, BMP, TIFF, WebP (via cv2.imread)
- **Output images**: JPEG with configurable quality
- **Model checkpoints**: PyTorch state dict files with `"params"` key
- **ONNX export**: Requires fixed input size (64x64 dummy input used)

### Incomplete / Experimental

- **predict.py** (lines 28-34): References `hat/test.py` and HAT model paths, appears to be legacy code from HAT project, non-functional
- **MambaDRCT**: Mentioned in README as experimental and abandoned, no code present
- **Flash Attention**: Code mentions it but uses standard attention (no explicit SDPA call in DRCT_arch.py)
- **Hann window blending**: Mentioned in PERFORMANCE.md as future work, not implemented (uses simple averaging)
