import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from drct.archs.DRCT_arch import DRCT

# Define image file extensions
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


def get_sorted_files_by_size(input_path, image_extensions):
    """
    Get a list of files sorted by file size (ascending).

    Args:
        input_path (str): Path to the directory containing images
        image_extensions (list): List of image extensions to look for (e.g., ['*.jpg', '*.png'])

    Returns:
        list: List of file paths sorted by size (smallest to largest)
    """
    # Create case-insensitive patterns by using both lowercase and uppercase versions
    case_insensitive_patterns = [
        f
        for ext in image_extensions
        for f in (
            os.path.join(input_path, ext.lower()),
            os.path.join(input_path, ext.upper()),
        )
    ]

    # Collect all files matching any of the patterns
    input_files = []
    for pattern in case_insensitive_patterns:
        input_files.extend(glob.glob(pattern))

    # Remove any duplicates that might have been found
    input_files = list(dict.fromkeys(input_files))

    # Sort files by their size
    input_files = sorted(input_files, key=lambda x: os.path.getsize(x))

    return input_files


def check_ampere_gpu():
    """
    Check if the GPU supports NVIDIA Ampere or later and enable FP32 in PyTorch if it does.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("No GPU detected, running on CPU.")
        return

    try:
        # Get the compute capability of the GPU
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        # Check if the GPU is Ampere or newer (compute capability >= 8.0)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) supports NVIDIA Ampere or later, enabled TF32 in PyTorch."
            )
        else:
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) does not support NVIDIA Ampere or later."
            )

        torch.backends.cudnn.benchmark = True

    except Exception as e:
        print(f"Error occurred while checking GPU: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DRCT Inference",
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
        # noqa: E251
        default="experiments/pretrained_models/DRCT-L_X4.pth",  # noqa: E501
    )

    parser.add_argument("--output", type=str, default=None, help="output folder")
    parser.add_argument("--scale", type=int, default=4, help="scale factor: 1, 2, 3, 4")
    # parser.add_argument('--window_size', type=int, default=16, help='16')

    parser.add_argument(
        "--tile",
        type=int,
        default=256,
        help="Tile size, -1 for no tile during inference (inference on whole img)",
    )
    parser.add_argument(
        "--tile_overlap", type=int, default=16, help="Overlapping of different tiles"
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=90, help="JPEG quality (0-100)"
    )

    parser.add_argument("--compile", action="store_true", help="use torch.compile")
    parser.add_argument(
        "-skip", "--skip_completed", action="store_true", help="skip completed images"
    )

    args = parser.parse_args()
    print(f"Running inference with args:\n{json.dumps(args.__dict__, indent=4)}")

    if args.tile < 1:
        args.tile = None

    check_ampere_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set up model (DRCT-L)
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
        torch.load(args.model_path, weights_only=True)["params"], strict=True
    )
    model.eval()
    model = model.to(device)

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")

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
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            img = (
                torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1)))
                .float()
                .pin_memory()
            )

            img = img.unsqueeze(0).to(device, non_blocking=True)

            # inference
            with (
                torch.inference_mode(),
                torch.autocast("cuda", enabled=torch.cuda.is_available()),
            ):
                # output = model(img)
                _, _, h_old, w_old = img.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img = torch.cat([img, torch.flip(img, [2])], 2)[
                    :, :, : h_old + h_pad, :
                ]
                img = torch.cat([img, torch.flip(img, [3])], 3)[
                    :, :, :, : w_old + w_pad
                ]
                output = test(img, model, args, window_size)
                output = output[..., : h_old * args.scale, : w_old * args.scale]
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        except Exception as error:
            print("Error", error, imgname)
        else:
            # save image
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(
                str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ].add_(out_patch)
                W[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == "__main__":
    main()
