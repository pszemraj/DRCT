# DRCT Inference Performance Guide

This guide covers the various optimizations implemented in the DRCT inference pipeline and how to use them effectively.

## Table of Contents

- [Implemented Optimizations](#implemented-optimizations)
  - [1. Memory Format Optimization](#1-memory-format-optimization)
  - [2. Precision Control](#2-precision-control)
  - [3. Batched Tile Processing](#3-batched-tile-processing)
  - [4. Stream Overlap](#4-stream-overlap)
  - [5. Efficient Padding](#5-efficient-padding)
  - [6. Attention Mask Caching](#6-attention-mask-caching)
  - [7. Flash Attention 2](#7-flash-attention-2)
- [Usage Tips](#usage-tips)
- [Performance Benchmarks](#performance-benchmarks)
- [Future Work](#future-work)

## Implemented Optimizations

### 1. Memory Format Optimization

**What it does**: Uses `channels_last` memory format for input tensors and model weights.

**Why it helps**: Better memory access patterns on modern GPUs, reducing bandwidth bottlenecks.

**Usage**: Enabled by default. No user action required.

### 2. Precision Control

**What it does**: Allows selection of FP32, FP16, or BF16 precision for inference.

**Why it helps**: BF16/FP16 reduces memory usage by ~50% and increases throughput on modern GPUs while maintaining quality.

**Memory impact**:
- FP32: 4 bytes/element (highest quality, highest memory usage)
- BF16/FP16: 2 bytes/element (~50% memory savings, slight quality tradeoff)

**Usage**: Use `--precision bf16` (default on Ampere+ GPUs) or `--precision fp16` for older GPUs.

**Tradeoffs**: Lower precision may introduce slight numerical artifacts in very demanding scenarios, but generally imperceptible in image super-resolution.

### 3. Batched Tile Processing

**What it does**: Processes multiple tiles in batches rather than one at a time.

**Why it helps**: Better GPU utilization and reduced Python overhead.

**Memory impact**:
- Each tile requires memory proportional to `tile_size² × scale_factor²`
- Batch size N increases memory usage by approximately N× but improves throughput
- Default `batch_size=1`: minimal memory overhead
- Recommended `batch_size=4`: ~4× memory usage but 2-3× speedup

**Usage**: Control batch size with `--tile_batch_size N` (default: 1).

**Tradeoffs**: Higher batch sizes increase VRAM usage but improve GPU utilization and throughput. Reduce batch size if running out of memory.

### 4. Stream Overlap

**What it does**: Uses multiple CUDA streams to overlap Host-to-Device transfers with computation.

**Why it helps**: Hides memory transfer latency, improving overall throughput.

**Memory impact**:
- Each additional stream requires duplicate input/output buffers
- 2 streams: ~2× memory usage for I/O buffers, 1.5-2× speedup
- 4 streams: ~4× memory usage for I/O buffers, up to 2.5× speedup (diminishing returns)
- Default `streams=1`: minimal memory overhead

**Usage**: Enable with `--streams N` (default: 1). Recommended values: 2-4.

**Tradeoffs**: More streams increase memory usage for buffering but can significantly improve throughput by overlapping computation with data transfers. Best used with sufficient VRAM headroom.

### 5. Efficient Padding

**What it does**: Uses `F.pad` with `reflect` mode instead of flip-concat operations.

**Why it helps**: More memory efficient and faster than the original implementation.

**Usage**: Enabled by default. No user action required.

### 6. Attention Mask Caching

**What it does**: Caches computed attention masks to avoid recomputation.

**Why it helps**: Reduces redundant calculations in transformer blocks.

**Usage**: Enabled by default. No user action required.

### 7. Flash Attention 2

**What it does**: Uses PyTorch's Scaled Dot Product Attention (SDPA) with Flash Attention 2 support for memory-efficient attention computation.

**Why it helps**: Reduces memory complexity from O(N²) to O(N) where N is sequence length (tile_size²), enabling larger tile sizes with less VRAM usage.

**Memory impact**:
- Original attention: Memory scales with `tile_size⁴`
- Flash Attention 2: Memory scales more linearly with `tile_size²`
- Larger tiles (512×512, 1024×1024) see significant memory savings

**Usage**: Enabled by default. No user action required.

**Tradeoffs**: May show minimal speedup on small tiles but provides substantial benefits with larger tiles and longer sequences. Automatically falls back to standard attention on unsupported hardware.

## Usage Tips

### Quick Start Configurations

1. **Maximum Speed** (high VRAM available):
   ```bash
   --precision bf16 --streams 2 --tile_batch_size 4
   ```

2. **Memory-Constrained Systems**:
   ```bash
   --precision bf16 --tile_batch_size 1 --tile 256
   ```

3. **Quality-Sensitive Applications**:
   ```bash
   --precision fp32 --tile 512
   ```

4. **Balanced Performance**:
   ```bash
   --precision bf16 --tile_batch_size 2 --streams 2
   ```

### Monitoring Performance

- Use `nvidia-smi -l 1` to monitor GPU utilization and memory usage
- Adjust batch size and stream count based on GPU utilization
- Target 80-95% GPU utilization for optimal performance

## Performance Benchmarks

### Configuration Impact on Speed (RTX 4090)

| Configuration | Tile Size | Batch Size | Streams | Precision | Throughput |
|--------------|-----------|------------|---------|-----------|------------|
| Conservative | 256×256   | 1          | 1       | FP32      | 1.0× (baseline) |
| Balanced     | 512×512   | 2          | 2       | BF16      | ~3.5× |
| Aggressive   | 512×512   | 4          | 4       | BF16      | ~5.0× |
| Maximum      | 1024×1024 | 2          | 2       | BF16      | ~4.5× |

### Memory Usage by Configuration

| Tile Size | Precision | Batch Size | Approx. VRAM Usage |
|-----------|-----------|------------|-------------------|
| 256×256   | FP32      | 1          | ~4 GB |
| 256×256   | BF16      | 1          | ~2 GB |
| 512×512   | FP32      | 1          | ~8 GB |
| 512×512   | BF16      | 2          | ~8 GB |
| 1024×1024 | BF16      | 1          | ~10 GB |

## Future Work

### Planned Optimizations

1. **Automatic Tile Sizing** (`--tile_auto`)
   - Automatically select optimal tile size based on available VRAM
   - Dynamic adjustment based on image dimensions

2. **Hann Window Blending** (`--hann_blend`)
   - Smoother tile boundaries with less overlap
   - Reduced visible seams in output

3. **CUDA Graphs**
   - Static graph capture for fixed tile configurations
   - Significant speedup when processing multiple images with same dimensions

4. **TensorRT Support**
   - ONNX-TensorRT path for additional optimizations
   - Further performance improvements on NVIDIA GPUs

5. **Asynchronous I/O**
   - Offload image loading/saving to separate threads
   - Prevent I/O from blocking main computation loop

6. **Dynamic Batching**
   - Automatically adjust batch size based on available memory
   - Maximize GPU utilization without OOM errors

### Experimental Features

- **Mixed Precision Training**: Investigate training with BF16 for faster convergence
- **Quantization**: Explore INT8 quantization for edge deployment
- **Multi-GPU Support**: Distribute tiles across multiple GPUs