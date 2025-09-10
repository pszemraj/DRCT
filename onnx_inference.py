import argparse
import cv2
import glob
import json
import numpy as np
import onnxruntime as ort
import os

# For PyTorch model loading and export
import torch
from pathlib import Path
from tqdm.auto import tqdm

from drct.archs.DRCT_arch import DRCT

# Define image file extensions
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


def get_sorted_files_by_size(input_path, image_extensions):
    """
    Get a list of files sorted by file size (ascending).
    """
    case_insensitive_patterns = [
        f
        for ext in image_extensions
        for f in (
            os.path.join(input_path, ext.lower()),
            os.path.join(input_path, ext.upper()),
        )
    ]

    input_files = []
    for pattern in case_insensitive_patterns:
        input_files.extend(glob.glob(pattern))

    input_files = list(dict.fromkeys(input_files))
    input_files = sorted(input_files, key=lambda x: os.path.getsize(x))

    return input_files


def check_gpu_provider():
    """
    Check available ONNX Runtime providers and select the best GPU provider.
    """
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        print("CUDA GPU acceleration is available")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        print("No GPU provider found, falling back to CPU")
        return ["CPUExecutionProvider"]


def export_to_onnx(pytorch_model_path, onnx_path, device):
    """
    Export PyTorch model to ONNX format if needed.
    Returns True if export was performed, False if skipped.
    """
    # Check if ONNX model already exists and is newer than PyTorch model
    if os.path.exists(onnx_path):
        pytorch_mtime = os.path.getmtime(pytorch_model_path)
        onnx_mtime = os.path.getmtime(onnx_path)
        if onnx_mtime > pytorch_mtime:
            print(f"ONNX model {onnx_path} is up to date")
            return False

    print(f"Exporting PyTorch model to ONNX format: {onnx_path}")

    # Load PyTorch model
    model = DRCT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        gc=32,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )

    model.load_state_dict(
        torch.load(pytorch_model_path, weights_only=True)["params"], strict=True
    )
    model.eval()
    model = model.to(device)

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=17,  # Using a recent opset version for better compatibility
    )
    print("ONNX export completed")
    return True


def test(img_lq, session, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = session.run(None, {"input": img_lq})[0]
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.shape
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = np.zeros((b, c, h * sf, w * sf), dtype=img_lq.dtype)
        W = np.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                out_patch = session.run(None, {"input": in_patch})[0]
                out_patch_mask = np.ones_like(out_patch)

                E[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ] += out_patch
                W[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ] += out_patch_mask

        output = E / W

    return output


def main():
    parser = argparse.ArgumentParser(
        description="DRCT ONNX Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="input test image folder",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="experiments/pretrained_models/DRCT-L_X4.pth",
    )
    parser.add_argument("--output", type=str, default=None, help="output folder")
    parser.add_argument("--scale", type=int, default=4, help="scale factor: 1, 2, 3, 4")
    parser.add_argument(
        "--tile",
        type=int,
        default=256,
        help="Tile size, -1 for no tile during inference",
    )
    parser.add_argument(
        "--tile_overlap", type=int, default=16, help="Overlapping of different tiles"
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=90, help="JPEG quality (0-100)"
    )
    parser.add_argument(
        "-skip", "--skip_completed", action="store_true", help="skip completed images"
    )
    parser.add_argument(
        "--force_export",
        action="store_true",
        help="Force ONNX model export even if it exists",
    )

    args = parser.parse_args()
    print(f"Running inference with args:\n{json.dumps(args.__dict__, indent=4)}")

    if args.tile < 1:
        args.tile = None

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate ONNX model path from PyTorch model path
    pytorch_model_path = args.model_path
    onnx_model_path = str(Path(pytorch_model_path).with_suffix(".onnx"))

    # Export model if needed
    if args.force_export or not os.path.exists(onnx_model_path):
        export_to_onnx(pytorch_model_path, onnx_model_path, device)

    # Initialize ONNX Runtime session with GPU provider
    providers = check_gpu_provider()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_model_path, providers=providers, sess_options=session_options
    )

    window_size = 16
    out_dir = (
        Path(args.output)
        if args.output is not None
        else Path(args.input) / "upscaled-DRCT-outputs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = get_sorted_files_by_size(args.input, image_extensions)
    for path in tqdm(input_files, desc="inference"):
        imgname = os.path.splitext(os.path.basename(path))[0]
        out_path = out_dir / f"{imgname}_DRCT-L_X{args.scale}.jpg"

        if args.skip_completed and out_path.exists():
            print(f"Skipping completed image: {out_path.name}")
            continue

        try:
            # Read and preprocess image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # Handle padding
            _, _, h_old, w_old = img.shape
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old

            # Pad image using numpy operations
            img_flipped_h = np.flip(img, axis=2)
            img = np.concatenate([img, img_flipped_h], axis=2)[:, :, : h_old + h_pad, :]

            img_flipped_w = np.flip(img, axis=3)
            img = np.concatenate([img, img_flipped_w], axis=3)[:, :, :, : w_old + w_pad]

            # Run inference
            output = test(img, session, args, window_size)
            output = output[..., : h_old * args.scale, : w_old * args.scale]
            output = output.squeeze()

        except Exception as error:
            print(f"Error processing {imgname}:", error)
        else:
            # Save image
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(
                str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )


if __name__ == "__main__":
    main()
