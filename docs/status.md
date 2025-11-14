# Status

This document provides an honest assessment of DRCT's implementation maturity, limitations, and technical debt.

## Maturity by Area

### Core Architecture (DRCT_arch.py)

**Status:** ✅ **Stable and used**

- Fully implemented, matches paper description
- Tested extensively (CVPR published, benchmarked on Set5/Set14/BSD100/Urban100/Manga109)
- No known correctness issues
- Compatible with BasicSR training infrastructure

**Concerns:**
- Hardcoded window_size=16 in many places
- No runtime validation of architecture parameter combinations
- `compress_ratio` and `squeeze_factor` lack clear tuning guidance

### Training Infrastructure

**Status:** ⚠️ **Prototype but working** (with caveats)

**What works:**
- `drct/train.py` successfully delegates to BasicSR's distributed training
- `drct/models/drct_model.py` tested with real training (evidenced by released checkpoints)
- GAN and MSE variants exist (`realdrctgan_model.py`, `realdrctmse_model.py`)

**What's missing:**
- **No configuration files:** `options/train/*.yml` referenced in README but absent from repo
- **No dataset preparation scripts:** Assumes pre-prepared paired data
- **No training documentation:** Users must reverse-engineer config from code and BasicSR docs
- **No training logs/checkpoints:** `experiments/pretrained_models/` is empty

**Usability:** Training is possible but requires significant setup work (creating YAML configs, preparing datasets, downloading external checkpoints).

### Basic Inference (inference.py)

**Status:** ✅ **Stable and used**

- Simple, readable, functional
- Supports tile mode and whole-image mode
- Optional torch.compile support
- Production-tested (used in community deployments)

**Limitations:**
- No batched tile processing
- Inefficient padding (flip-concat)
- No advanced memory optimizations

### Optimized Inference (efficient_inference.py)

**Status:** ✅ **Stable and used** (production-grade)

- Extensive optimizations: channels_last, mixed precision, batched tiles, CUDA streams
- Torch.compile integration with buffer management fixes
- Comprehensive CLI options
- Well-documented in PERFORMANCE.md

**Concerns:**
- Complex codebase (1194 lines vs inference.py's 267)
- Tight coupling to specific PyTorch versions (compile requires 2.0+)
- Hardcoded DRCT-L config (no runtime architecture selection)
- Some experimental features mentioned but not fully implemented (see below)

### ONNX Export/Inference (onnx_inference.py)

**Status:** ⚠️ **Prototype but working**

**What works:**
- Exports PyTorch model to ONNX
- Runs inference via ONNX Runtime
- Automatic export if ONNX file missing/outdated

**Limitations:**
- Fixed input size (64×64 dummy input at export)
- No dynamic shape support
- Tile size must match export dimensions
- Less flexible than PyTorch path
- Not as optimized as TensorRT could be

**Usability:** Functional for deployment but requires careful management of input dimensions.

### Testing/Validation (drct/test.py)

**Status:** ⚠️ **Prototype but working**

**What works:**
- Delegates to BasicSR's test pipeline
- Computes PSNR/SSIM metrics
- Saves visualizations
- Supports tile mode

**What's missing:**
- **No test configuration files:** `options/test/*.yml` absent from repo
- **No standalone evaluation script:** Must use BasicSR infrastructure
- **No automated benchmark runner:** Manual per-dataset testing required

### Data Loading (imagenet_paired_dataset.py)

**Status:** ✅ **Stable and used**

- Generates LR images via bicubic downsampling
- Supports LMDB and directory backends
- Standard augmentations (crop, flip, rotation)
- Compatible with BasicSR's data pipeline

**Limitations:**
- Only supports bicubic degradation (no real-world degradation modeling)
- Assumes HR images only (not paired LR/HR)
- No support for meta-learning or few-shot scenarios

### Discriminator (discriminator_arch.py)

**Status:** ✅ **Stable and used** (for GAN training)

- Standard UNet-style discriminator
- Used in RealDRCTGAN training
- No known issues

**Concerns:**
- Only tested with GAN variants (MSE-only training doesn't use)
- No alternative discriminator architectures provided

### SRVGG Architecture (srvgg_arch.py)

**Status:** ⚠️ **Experimental / incomplete**

- Lightweight CNN-based SR model
- Registered with ARCH_REGISTRY
- Not documented in main README
- Unclear if trained or tested

**Usability:** Present but not integrated into main workflows. May be legacy or future work.

## Technical Debt and Dragons

### 🐉 Missing Configuration Files

**Location:** `options/` directory

**Issue:**
- README references `options/train/*.yml` and `options/test/*.yml`
- Directory doesn't exist in this repo
- Users cannot train or test without creating configs from scratch

**Impact:** High barrier to entry for training/testing

**Workaround:**
- Refer to BasicSR documentation for config format
- Adapt configs from similar projects (HAT, SwinIR)
- Use example configs from DRCT's original repository (if available externally)

**Why it exists:**
- Configs often contain absolute paths to datasets (not portable)
- May have been .gitignored to avoid hardcoded paths
- Likely oversight in repository setup

### 🐉 Hardcoded Architecture Parameters

**Location:** `inference.py`, `efficient_inference.py`, `onnx_inference.py`

**Issue:**
All inference scripts hardcode DRCT-L configuration:
```python
model = DRCT(
    upscale=4,
    embed_dim=180,
    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    # ... 12 more parameters
)
```

**Impact:**
- Cannot easily switch between DRCT, DRCT-L, DRCT-XL
- Must manually edit code to change architecture
- No CLI flag for model variant selection

**Workaround:**
- Manually modify model instantiation code
- Create separate scripts per variant
- Infer architecture from checkpoint metadata (requires custom code)

**Why it exists:**
- Simplicity: Avoids config parsing in standalone scripts
- Most users only need one model variant
- Architecture parameters rarely change during deployment

### 🐉 Predict.py Legacy Code

**Location:** `predict.py`

**Issue:**
```python
subprocess.call([
    "python", "hat/test.py",  # References HAT project
    "-opt", "options/test/HAT_SRx4_ImageNet-LR.yml",  # HAT config
])
```

**Impact:**
- Non-functional (HAT paths don't exist in DRCT repo)
- Misleading for users
- Dead code that should be removed or fixed

**Why it exists:**
- Copy-paste from HAT project (DRCT is influenced by HAT)
- Intended for Cog deployment but never updated
- Likely forgotten during repository cleanup

### 🐉 Inconsistent Checkpoint Loading

**Location:** Multiple files

**Issue:**
- `inference.py` line 159: `strict=True`
- `efficient_inference.py` line 1106: `strict=False`

**Impact:**
- Different error behavior when checkpoint keys mismatch
- `strict=True` fails if `mean` buffer missing (old checkpoints)
- `strict=False` silently ignores missing/extra keys (can mask errors)

**Why it exists:**
- `efficient_inference.py` uses `register_buffer("mean", ...)` which may not exist in old checkpoints
- `inference.py` assumes checkpoints match architecture exactly
- Buffer management evolved over time

**Recommendation:** Always use `strict=False` with explicit warning logging for missing/extra keys.

### 🐉 Tile Size / Window Size Coupling

**Location:** `drct/archs/DRCT_arch.py`, inference scripts

**Issue:**
- Tile size must be multiple of window_size (16)
- No runtime validation in DRCT architecture
- Error only manifests as shape mismatch during forward pass
- Cryptic error messages

**Example:**
```bash
python inference.py input --tile 250  # 250 % 16 != 0
# RuntimeError: shape '[1, 15, 15, 180]' is invalid for input of size ...
```

**Workaround:**
- Always use tile sizes like 256, 512, 768, 1024 (multiples of 16)
- `inference.py` pads to window_size multiple (lines 203-210)
- Error message doesn't clearly state "tile must be multiple of 16"

**Why it exists:**
- Window partitioning requires even division: H % window_size == 0
- No assertion at construction time
- Padding handles it for whole images, but not for user-specified tile sizes

### 🐉 Global State in Compilation

**Location:** `efficient_inference.py`, lines 33-61

**Issue:**
```python
os.environ.setdefault("TORCH_LOGS", "+dynamo,graph_breaks,...")
dynamo.config.suppress_errors = False
dynamo.config.verbose = True
inductor.config.debug = True
```

**Impact:**
- Global environment modification
- Affects all torch compilation in process
- No cleanup/restoration
- Verbose logging floods console

**Why it exists:**
- Debugging torch.compile is difficult without extensive logging
- Intended for development, but left in production code
- No CLI flag to disable debug output

**Recommendation:** Guard behind `--debug` flag or environment variable check.

### 🐉 Attention Mask Device Placement

**Location:** `drct/archs/DRCT_arch.py`, `SwinTransformerBlock.__init__` (lines 410-416)

**Issue:**
```python
if self.shift_size > 0:
    attn_mask = None  # Will be created on-demand with correct device
else:
    attn_mask = None
self.register_buffer("attn_mask", attn_mask)
```

**Impact:**
- Mask not pre-computed, calculated during forward pass
- `calculate_mask()` called repeatedly for same input size
- Caching logic in `forward()` (lines 460-476) is complex

**Why it exists:**
- Torch.compile requires buffers on correct device before compilation
- Pre-computing mask during `__init__` doesn't know target device
- Dynamic mask calculation is safe but suboptimal

**Optimization potential:** Pre-compute mask in `__init__` as CPU buffer, move to device in first forward pass, cache result.

## Missing Pieces

### Configuration Management

**What's hinted but not implemented:**

1. **Runtime architecture selection:**
   - CLI flag like `--model {drct, drct-l, drct-xl}` to select variant
   - Architecture inference from checkpoint metadata

2. **YAML configs for inference:**
   - All inference scripts use CLI args only
   - No config file support (unlike training/testing)

3. **Checkpoint metadata:**
   - Checkpoints contain only `"params"` dict
   - No version, architecture config, or training metadata

### Training Features

**Referenced but absent:**

1. **X2 pretraining configs:**
   - README mentions x2 pretraining, but no configs provided
   - Unclear if x2→x4 finetuning was actually used

2. **ImageNet pretraining configs:**
   - README mentions ImageNet pretraining
   - Links to external checkpoints but no training configs

3. **Multi-scale training:**
   - Code supports scale ∈ {2, 3, 4} but no documentation on training each

4. **Distributed training examples:**
   - README shows 8-GPU distributed command
   - No guidance on adjusting for different GPU counts
   - No single-GPU training example

### Inference Features

**Mentioned in PERFORMANCE.md but not implemented:**

1. **Automatic tile sizing (`--tile_auto`):**
   - Would detect available VRAM and set optimal tile size
   - Currently manual trial-and-error

2. **Hann window blending (`--hann_blend`):**
   - Would reduce tile boundary artifacts
   - Currently uses simple averaging

3. **CUDA graphs:**
   - Would reduce CPU overhead for repeated inference
   - Torch.compile may enable this, but not explicit

4. **TensorRT backend:**
   - ONNX export exists, but no TensorRT conversion pipeline

5. **Asynchronous I/O:**
   - Image loading/saving blocks main loop
   - No parallel I/O workers

6. **Dynamic batching:**
   - Could auto-adjust batch_size based on VRAM headroom
   - Currently fixed at CLI argument

### Evaluation Infrastructure

**What's missing:**

1. **Standalone benchmark script:**
   - Must use `drct/test.py` with BasicSR infrastructure
   - No simple "evaluate checkpoint on Set5" command

2. **Automated benchmark suite:**
   - No script to run all benchmarks (Set5, Set14, BSD100, etc.) in sequence
   - Manual execution per dataset

3. **Metric computation without BasicSR:**
   - PSNR/SSIM computation tied to BasicSR's `calculate_metric`
   - No lightweight standalone metric script

4. **Visual quality assessment:**
   - No LPIPS, DISTS, or perceptual quality metrics
   - Only PSNR/SSIM (which don't correlate well with human perception)

## Scaling and Correctness Issues

### Known Limitations from Code

1. **Maximum sequence length (line counts in efficient_inference.py):**
   - Attention complexity: O(window_size²) per window
   - Larger window_size requires O(N²) relative position bias table
   - Current window_size=16 → 256 tokens per window → 961 relative positions
   - Increasing to window_size=32 → 1024 tokens → 3969 positions (4× memory)

2. **Tile boundary artifacts:**
   - Visible seams with small `--tile_overlap` values
   - Averaging blend at boundaries (no feathering)
   - Color shift possible between tiles due to batch normalization

3. **Memory scaling:**
   - Peak memory during tile accumulation: `E` and `W` buffers are full-resolution at upscaled size
   - For 8K input (7680×4320), 4× upscale → 30720×17280 output → ~6GB just for E/W buffers (fp32)

4. **Numerical stability:**
   - Mixed precision (fp16/bf16) may accumulate errors in long sequences
   - No overflow protection in tile accumulation (assumes `W` never zero)

5. **Window size constraints:**
   - Hardcoded at 16 in all inference scripts
   - Changing window_size requires retraining
   - No support for adaptive window sizes based on content

### Fragile Areas

1. **Torch.compile compatibility:**
   - Brittle to PyTorch version changes
   - Breaks with dynamic shapes
   - Requires specific buffer management patterns
   - Dry-run can fail with cryptic errors

2. **Checkpoint compatibility:**
   - No version checking
   - Format changes (e.g., adding `mean` buffer) break old checkpoints
   - `strict=False` masks errors silently

3. **Device management:**
   - Assumes single GPU or CPU
   - No multi-GPU inference support
   - Mixed CPU/GPU tensors cause crashes

4. **Error handling:**
   - Many operations lack try-except blocks
   - OOM errors crash entire script (no graceful degradation)
   - Invalid tile sizes produce cryptic shape errors

## Dead / Unused Code

### Confirmed Dead

1. **`predict.py`** (entire file):
   - References non-existent `hat/` directory
   - Cog predictor wrapper never updated for DRCT
   - Should be removed or rewritten

2. **MambaDRCT** (mentioned in README, not in code):
   - README says "not plan to continue fixing"
   - Experimental fusion of DRCT + MambaIR
   - Abandoned due to slow training

### Likely Unused

1. **`drct/archs/srvgg_arch.py`:**
   - No documentation
   - No training configs
   - No inference examples
   - May be legacy or placeholder

2. **Some BasicSR options:**
   - `use_checkpoint` parameter exists in DRCT.__init__ but never used
   - No gradient checkpointing implementation in forward pass

3. **`ape` (absolute position embedding) option:**
   - Present in DRCT.__init__ (line 774)
   - Defaults to False
   - No configs or examples using it
   - May be untested

### Vestigial Imports

1. **`einops` dependency:**
   - Listed in pyproject.toml
   - No usage in actual code (grepped codebase)
   - May be from copy-paste or planned future use

2. **`cv2` in ONNX script:**
   - Imported but minimal usage (only for imread)
   - Could use PIL instead for consistency

## Recommended Actions

### High Priority

1. **Add training/test configs** to `options/` directory (or document how to create them)
2. **Remove or fix `predict.py`** to avoid user confusion
3. **Add CLI flag for model variant** selection in inference scripts
4. **Document checkpoint format** and version compatibility

### Medium Priority

1. **Validate tile_size % window_size == 0** at runtime with clear error message
2. **Guard debug logging** behind environment variable or flag
3. **Add standalone benchmark script** (no BasicSR dependency)
4. **Implement graceful OOM handling** (reduce tile_batch_size and retry)

### Low Priority

1. **Remove unused dependencies** (einops)
2. **Optimize attention mask caching** (pre-compute in __init__)
3. **Add multi-GPU inference support**
4. **Implement Hann window blending** for better tile stitching
5. **Create TensorRT conversion pipeline**

## Maintenance Notes

- **Last major update:** Dec 2024 (based on commit history)
- **Active development:** Yes (efficient_inference.py actively maintained)
- **Community contributions:** Limited (fork-based improvements)
- **Dependency stability:** Tied to BasicSR==1.3.4.9 (older version)

**Upgrade risks:**
- BasicSR updates may break compatibility
- PyTorch 2.x required for torch.compile (breaking change from 1.x)
- CUDA version requirements increasing with newer PyTorch

**Stability assessment:** Core inference is production-ready; training infrastructure requires setup work; some experimental features incomplete.
