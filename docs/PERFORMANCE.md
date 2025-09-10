# DRCT Inference Performance Guide

## Implemented Optimizations

### 1. Memory Format Optimization
**What it does**: Uses `channels_last` memory format for input tensors and model weights.
**Why it helps**: Better memory access patterns on modern GPUs, reducing bandwidth bottlenecks.
**Usage**: Enabled by default. No user action required.

### 2. Precision Control
**What it does**: Allows selection of FP32, FP16, or BF16 precision for inference.
**Why it helps**: BF16/FP16 reduces memory usage and increases throughput on modern GPUs while maintaining quality.
**Usage**: Use `--precision bf16` (default on Ampere+ GPUs) or `--precision fp16` for older GPUs.

### 3. Batched Tile Processing
**What it does**: Processes multiple tiles in batches rather than one at a time.
**Why it helps**: Better GPU utilization and reduced Python overhead.
**Usage**: Control batch size with `--tile_batch_size N` (default: 1).

### 4. Stream Overlap
**What it does**: Uses multiple CUDA streams to overlap Host-to-Device transfers with computation.
**Why it helps**: Hides memory transfer latency, improving overall throughput.
**Usage**: Enable with `--streams N` (default: 1). Recommended values: 2-4.

### 5. Efficient Padding
**What it does**: Uses `F.pad` with `reflect` mode instead of flip-concat operations.
**Why it helps**: More memory efficient and faster than the original implementation.
**Usage**: Enabled by default. No user action required.

### 6. Attention Mask Caching
**What it does**: Caches computed attention masks to avoid recomputation.
**Why it helps**: Reduces redundant calculations in transformer blocks.
**Usage**: Enabled by default. No user action required.

### 7. SDPA Attention
**What it does**: Uses PyTorch's Scaled Dot Product Attention (SDPA) with Flash Attention support.
**Why it helps**: Leverages optimized kernels for faster attention computation.
**Usage**: Enabled by default. No user action required.

## Usage Tips

1. **For maximum speed**: Use `--precision bf16 --streams 2 --tile_batch_size 4`
2. **For memory-constrained systems**: Reduce `--tile_batch_size` and `--tile` size
3. **For quality-sensitive applications**: Use `--precision fp32` (slower but highest precision)
4. **Monitor GPU utilization**: Use `nvidia-smi -l 1` to find optimal batch/stream settings

## Future Work

1. **Automatic tile sizing**: Implement `--tile_auto` to automatically select optimal tile size based on available VRAM
2. **Hann window blending**: Add `--hann_blend` for smoother tile boundaries with less overlap
3. **CUDA Graphs**: Implement static graph capture for fixed tile configurations (significant speedup when shapes are constant)
4. **TensorRT support**: Add ONNX-TensorRT path for additional optimizations
5. **Async I/O**: Offload image loading/saving to separate threads to avoid blocking main loop