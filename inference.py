import argparse
import cv2
import glob
import json
import numpy as np
import os
import torch
from tqdm.auto import tqdm

from drct.archs.DRCT_arch import *

# Define image file extensions
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff", "*.webp")


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
        "--tile_overlap", type=int, default=32, help="Overlapping of different tiles"
    )
    parser.add_argument(("--compile"), action="store_true", help="use torch.compile")

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

    model = torch.compile(model)

    window_size = 16

    out_dir = (
        args.output
        if args.output is not None
        else os.path.join(args.input, "upscaled-DRCT-outputs")
    )
    os.makedirs(out_dir, exist_ok=True)
    input_files = sorted(
        [
            f
            for ext in image_extensions
            for f in glob.glob(os.path.join(args.input, ext))
        ]
    )
    for idx, path in enumerate(tqdm(input_files, desc="inference")):
        imgname = os.path.splitext(os.path.basename(path))[0]
        # print("Testing", idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

        # img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # print(img.shape)
        # inference
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
                # output = model(img)
                _, _, h_old, w_old = img.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img = torch.cat([img, torch.flip(img, [2])], 2)[
                    :, :, : h_old + h_pad, :
                ]
                img = torch.cat([img, torch.flip(img, [3])], 3)[
            with torch.inference_mode(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
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
                os.path.join(out_dir, f"{imgname}_DRCT-L_X{args.scale}.jpg"), output
            )

        if "cuda" in str(device):
            torch.cuda.empty_cache()


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
            cv2.imwrite(
                os.path.join(out_dir, f"{imgname}_DRCT-L_X{args.scale}.jpg"), output
            )

        if "cuda" in str(device):
            torch.cuda.empty_cache()
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
