# Internals

This document explains how DRCT's interesting technical components work at the implementation level.

## Dense Connections in RDG (Residual Dense Group)

### Core Idea

Traditional Swin Transformer blocks pass features sequentially: `x → block1 → block2 → ... → blockN`. Information can degrade through deep networks (information bottleneck). RDG addresses this by concatenating outputs from all previous blocks as inputs to each subsequent block.

### Implementation

**Code:** `drct/archs/DRCT_arch.py`, `RDG.forward()` (lines 611-617)

```python
def forward(self, x: torch.Tensor, xsize: Tuple[int, int]) -> torch.Tensor:
    x1 = self.pe(self.lrelu(self.adjust1(self.pue(self.swin1(x, xsize), xsize))))
    x2 = self.pe(self.lrelu(self.adjust2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize))))
    x3 = self.pe(self.lrelu(self.adjust3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize))))
    x4 = self.pe(self.lrelu(self.adjust4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize))))
    x5 = self.pe(self.adjust5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize)))
    return x5 * 0.2 + x
```

**Flow breakdown:**

1. **Block 1:** Input `x` (dim=180) → Swin → `x1` (dim=180)
   - `adjust1` (1×1 conv) compresses to `gc=32` channels

2. **Block 2:** Concat `[x, x1]` (dim=180+32=212) → Swin → `x2` (dim=212)
   - Input channels grow because Swin accepts variable dims
   - `num_heads` adjusted: `num_heads - (dim % num_heads)` to ensure divisibility
   - `adjust2` compresses to `gc=32` channels

3. **Block 3:** Concat `[x, x1, x2]` (dim=180+32+32=244) → Swin → `x3` (dim=244)
   - `adjust3` → `gc=32` channels

4. **Block 4:** Concat `[x, x1, x2, x3]` (dim=180+96=276) → Swin → `x4` (dim=276)
   - `mlp_ratio=1` (reduced MLP for memory efficiency)
   - `adjust4` → `gc=32` channels

5. **Block 5:** Concat `[x, x1, x2, x3, x4]` (dim=180+128=308) → Swin → `x5` (dim=308)
   - `mlp_ratio=1`
   - `adjust5` compresses back to original `dim=180`

6. **Residual connection:** `x5 * 0.2 + x`
   - Scaling factor 0.2 stabilizes training (prevents residual dominance)

**Why this works:**

- **Information preservation:** Each block receives direct access to all previous features, preventing gradient vanishing
- **Feature reuse:** Later blocks can selectively use features from any earlier block
- **Hierarchical features:** `x1` has 1 Swin block's processing, `x2` has 2, etc.—concatenating provides multi-scale features
- **Compact representation:** Growth channels (`gc=32`) keep memory manageable despite concatenation

**Memory cost:**

- Without dense connections: 5 blocks × 180 channels = 900 channel-blocks
- With dense connections: 180 + 212 + 244 + 276 + 308 = 1220 channel-blocks (~36% overhead)
- But information flow is significantly improved

**Comparison to DenseNet:**

DenseNet (CNNs):
```python
x1 = conv(x0)
x2 = conv(cat[x0, x1])
x3 = conv(cat[x0, x1, x2])
```

DRCT RDG (Transformers):
```python
x1 = swin(x0)
x2 = swin(cat[x0, x1])  # concat in channel dim, then adjust before next block
x3 = swin(cat[x0, x1, x2])
```

Key difference: DRCT adds `adjust` layers (1×1 convs) to compress after each Swin block, controlling growth.

## Window-Based Attention with Relative Position Bias

### Core Idea

Global self-attention has O(N²) complexity where N = H×W (all pixels). For SR, images are large (e.g., 256×256 = 65K pixels), making global attention prohibitive. Swin Transformer solves this by:

1. **Local windows:** Divide image into non-overlapping M×M windows (M=16), compute attention only within each window
2. **Shifted windows:** Every other layer, shift windows by M/2 to enable cross-window communication
3. **Relative position bias:** Learnable bias based on relative positions of tokens

### Implementation

**Code:** `drct/archs/DRCT_arch.py`, `WindowAttention.forward()` (lines 307-338)

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B_, N, C = x.shape  # N = window_size²
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * self.scale  # scale = head_dim^(-0.5)
    attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

    # Add relative position bias
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(window_size * window_size, window_size * window_size, -1).permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

    attn = self.softmax(attn)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

**Relative position bias details:**

Precomputed during `__init__` (lines 287-297):
```python
# Create relative position index
coords_h = torch.arange(window_size[0])  # [0, 1, ..., 15]
coords_w = torch.arange(window_size[1])  # [0, 1, ..., 15]
coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, 16, 16)
coords_flatten = torch.flatten(coords, 1)  # (2, 256)

# Compute pairwise relative positions
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, 256, 256)
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (256, 256, 2)

# Shift to start from 0 and convert to 1D index
relative_coords[:, :, 0] += window_size[0] - 1  # range [0, 30]
relative_coords[:, :, 1] += window_size[1] - 1
relative_coords[:, :, 0] *= 2 * window_size[1] - 1
relative_position_index = relative_coords.sum(-1)  # (256, 256), values in [0, 961)
```

**Bias table:** `self.relative_position_bias_table` is shape `(961, num_heads)` — one bias vector per relative position per head.

**Why this works:**

- Captures spatial relationship: Pixels closer together should have different attention than pixels far apart
- Learnable: Network learns which relative positions are important for SR
- Shared across all windows: Same bias table used for all spatial locations (parameter efficiency)
- Per-head: Different attention heads can focus on different spatial patterns

**Shifted window mechanism:**

`SwinTransformerBlock.forward()` (lines 445-487):
```python
if self.shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
else:
    shifted_x = x

x_windows = window_partition(shifted_x, self.window_size)
x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

if self.shift_size > 0:
    attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size, device=x.device))
else:
    attn_windows = self.attn(x_windows, mask=None)

attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
shifted_x = window_reverse(attn_windows, self.window_size, H, W)

if self.shift_size > 0:
    x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
```

**Mask calculation** (lines 418-443):
- For shifted windows, some window positions contain tokens from different regions
- Mask prevents attention between tokens from non-adjacent regions
- Computed as: `mask = -100.0` where tokens shouldn't attend, `0.0` otherwise
- Added to attention logits before softmax (−100 → ~0 after softmax)

**Complexity:**

- Standard attention: O(H×W)² = O(N²)
- Window attention: O(M²) per window × (H/M × W/M) windows = O(H×W × M²) = O(N×M²)
- For M=16, H=W=256: Standard = 4.3B ops, Window = 268M ops (~16× reduction)

## Channel Attention Block (CAB)

### Core Idea

After transformer processing, features may have redundant channels. CAB recalibrates channel-wise features using squeeze-excite style attention.

### Implementation

**Code:** `drct/archs/DRCT_arch.py`, `CAB` (lines 154-174) and `ChannelAttention` (lines 130-151)

```python
class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),  # Compress
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),  # Expand
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) → (B, C, 1, 1)
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),  # Squeeze
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),  # Excite
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)  # (B, C, 1, 1), values in [0, 1]
        return x * y  # Broadcast multiply: recalibrate each channel
```

**Flow:**

1. **Compress:** 180 channels → 60 channels (3x compression) via 3×3 conv + GELU
2. **Expand:** 60 channels → 180 channels via 3×3 conv
3. **Global pooling:** (B, 180, H, W) → (B, 180, 1, 1) by averaging over spatial dimensions
4. **Squeeze:** 180 → 6 channels (30x squeeze) via 1×1 conv
5. **Excite:** 6 → 180 channels via 1×1 conv
6. **Sigmoid:** Convert to [0, 1] attention weights
7. **Recalibrate:** Multiply original features by attention weights

**Why this works:**

- **Compress-expand:** Reduces redundancy and adds non-linearity (similar to MLP)
- **Global context:** AdaptiveAvgPool2d gives each channel a global statistic
- **Channel recalibration:** Sigmoid weights suppress less important channels, amplify important ones
- **Squeeze-excite bottleneck:** Forces network to learn a compact representation of channel importance

**Comparison to RCAN:**

RCAN uses simpler squeeze-excite (no compress-expand):
```python
pool → squeeze → relu → excite → sigmoid
```

DRCT CAB adds:
```python
compress → gelu → expand → (pool → squeeze → relu → excite → sigmoid)
```

This provides more modeling capacity at the cost of slightly more computation.

## Efficient Padding Strategy

### Problem

Original `inference.py` uses flip-concat padding:
```python
# Lines 205-210
img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
```

This creates intermediate tensors (cat, flip) and requires slicing, which is memory-inefficient.

### Solution

`efficient_inference.py` uses `F.pad` with reflect mode:
```python
# Line 1172
img = F.pad(img, (0, w_pad, 0, h_pad), mode="reflect")
```

**Why this is better:**

- **Single operation:** No intermediate tensors
- **Reflect mode:** Mirrors boundary pixels, similar to flip but more efficient
- **In-place capable:** Modern PyTorch can optimize this better
- **Memory:** O(output_size) instead of O(2×input_size) for concat

**Padding format:** `F.pad(tensor, (left, right, top, bottom))`
- `(0, w_pad, 0, h_pad)` means: 0 padding left, w_pad right, 0 top, h_pad bottom

**Reflect vs other modes:**

- **Reflect:** `[1, 2, 3, 4] → [1, 2, 3, 4, 3, 2]` (mirrors excluding edge)
- **Replicate:** `[1, 2, 3, 4] → [1, 2, 3, 4, 4, 4]` (repeats edge)
- **Zero:** `[1, 2, 3, 4] → [1, 2, 3, 4, 0, 0]`

Reflect is best for images: avoids discontinuities at boundaries.

## Batched Tile Processing

### Problem

Sequential tile processing (inference.py):
```python
for h_idx in h_idx_list:
    for w_idx in w_idx_list:
        in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
        out_patch = model(in_patch)  # Process one tile at a time
        E[...] += out_patch
        W[...] += 1
```

**Issues:**
- GPU underutilized (only one tile processed at a time)
- Python loop overhead between tiles
- No batching means low arithmetic intensity

### Solution

`efficient_inference.py` batches tiles:
```python
coords = [(y, x) for y in h_idx_list for x in w_idx_list]
for i in range(0, len(coords), args.tile_batch_size):
    ysxs = coords[i:i+args.tile_batch_size]
    batch = torch.cat([img_lq[..., y:y+tile, x:x+tile] for (y,x) in ysxs], dim=0)
    outs = model(batch)  # Process multiple tiles in one forward pass
    for j, (y, x) in enumerate(ysxs):
        E[..., y*sf:(y+tile)*sf, x*sf:(x+tile)*sf] += outs[j]
```

**Code path:** Lines 977-1010

**Why this works:**

- **Batching:** Model sees batch_size tiles simultaneously (e.g., 4 tiles)
- **GPU utilization:** Parallelizes computation across tiles
- **Reduced overhead:** Fewer Python→CUDA kernel launches
- **Memory trade-off:** Uses batch_size × tile memory, but 2-3× faster

**Example:**
- Image: 2048×2048, Tile: 512, Overlap: 32
- Tiles needed: 4×4 = 16 tiles
- Sequential: 16 forward passes
- Batched (batch_size=4): 4 forward passes (4× reduction in calls)

**CUDA stream overlap** (lines 970-1009):

Additional optimization: Use multiple CUDA streams to overlap H2D transfer with compute:
```python
streams = [torch.cuda.Stream() for _ in range(args.streams)]
for i, coords_batch in enumerate(batched_coords):
    current_stream = streams[i % len(streams)]
    with torch.cuda.stream(current_stream):
        batch = prepare_batch(coords_batch)
        outs = model(batch)
        accumulate_results(outs)
```

Overlaps:
- Stream 0: Transfer batch 0 → Compute batch 0
- Stream 1: Transfer batch 1 (while stream 0 computes) → Compute batch 1
- Result: Hides H2D transfer latency

## Torch.Compile Integration

### Implementation

**Code:** `efficient_inference.py`, lines 1111-1146

```python
if args.compile != "off":
    # Audit for CPU buffers (torch.compile requires all tensors on same device)
    audit_buffers(model)

    # Dry run with fullgraph=True to detect graph breaks
    if os.environ.get("SKIP_DRY_RUN", "0") != "1":
        try:
            _dry_run_compile(model, device, tile, amp_dtype=torch.bfloat16, mode=mode)
        except Exception as e:
            print(f"Dry run failed: {e}")

    # Compile for execution
    model = torch.compile(model, mode=mode, dynamic=False)

    # Warmup to trigger compilation
    dummy = torch.zeros(1, 3, tile, tile, device=device).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _ = model(dummy)
```

**Key aspects:**

1. **Buffer audit** (line 1120):
   - Checks all model buffers are on CUDA
   - CPU buffers cause compilation failure
   - `relative_position_index` buffer is on CUDA (registered in WindowAttention.__init__)

2. **Dry run** (lines 1127-1133):
   - Uses `torch._dynamo.explain(model)(x)` to count graph breaks
   - Attempts `fullgraph=True` compilation to surface any breaks as errors
   - Helps debug before real execution

3. **Static shapes** (`dynamic=False`):
   - Forces all tensor shapes to be fixed at compile time
   - Enables aggressive optimizations (operator fusion, CUDA graphs)
   - **Trade-off:** Recompiles if input shape changes

4. **Warmup**:
   - First forward pass triggers JIT compilation
   - Can take 30-120 seconds depending on mode
   - Subsequent passes use compiled kernels (2-5× faster)

**Why this works:**

- **Operator fusion:** Combines multiple ops into single CUDA kernels (reduces memory I/O)
- **Layout optimization:** Better memory access patterns
- **CUDA graphs:** Records kernel launches for reuse (reduces CPU overhead)
- **Inductor backend:** Generates optimized Triton kernels

**Graph breaks:**

Common causes in DRCT:
- **Dynamic control flow:** `if` statements based on tensor values (avoided in DRCT)
- **CPU fallbacks:** Operations not supported by inductor (e.g., some scatter ops)
- **Data-dependent shapes:** Shapes depending on tensor contents (RDG uses fixed concatenations)

DRCT is mostly graph-break-free due to:
- Fixed architecture (no dynamic layers)
- Static concatenations (dims known at compile time)
- No data-dependent branching

## Buffer Management for Torch.Compile

### Problem

Original DRCT_arch.py registered `mean` as a buffer during forward:
```python
# In forward(), problematic:
self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1).to(device)
```

This causes issues:
- Buffers must be registered in `__init__`, not `forward`
- Torch.compile requires all buffers on same device before compilation
- Dynamic buffer creation breaks graph tracing

### Solution

`efficient_inference.py` DRCT class (lines 736-739):
```python
if in_chans == 3:
    self.register_buffer("mean", torch.Tensor(rgb_mean).view(1, 3, 1, 1))
else:
    self.register_buffer("mean", torch.zeros(1, 1, 1, 1))
```

**Why this works:**

- `register_buffer()` makes `mean` part of module state
- Automatically moved to device with `model.to(device)`
- Persistent across `state_dict` save/load
- Compile-time constant (no runtime device movement)

**Usage in forward:**
```python
x = (x - self.mean) * self.img_range  # self.mean already on correct device
```

**Comparison:**

❌ **Dynamic (breaks compile):**
```python
mean = torch.tensor([0.44, 0.43, 0.40]).view(1, 3, 1, 1).to(x.device)
x = x - mean
```

✅ **Static (compile-safe):**
```python
# In __init__:
self.register_buffer("mean", torch.tensor([0.44, 0.43, 0.40]).view(1, 3, 1, 1))

# In forward:
x = x - self.mean  # Compile sees this as constant buffer access
```

This pattern is crucial for torch.compile compatibility and is applied consistently in the efficient_inference.py DRCT implementation.
