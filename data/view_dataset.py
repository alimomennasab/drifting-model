import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.vae_util import VAEWrapper


# python view_dataset.py --batch_file /data/ali/imf_latents/train_overfit30_10classes.pt

def main():
    parser = argparse.ArgumentParser(
        description="Decode and plot a saved latent batch with class labels."
    )
    parser.add_argument("--batch_file", help="Path to the saved .pt batch dataset")
    parser.add_argument(
        "--output",
        help="Output image path (default: <batch_file>_preview.png)",
    )
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.columns < 1:
        raise ValueError("--columns must be positive")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be positive")

    batch = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    x_batch = batch["x_batch"]
    y_batch = batch["y_batch"]

    if len(x_batch) != len(y_batch):
        raise ValueError(
            f"Mismatched batch sizes: {len(x_batch)} images and "
            f"{len(y_batch)} labels"
        )
    if len(x_batch) == 0:
        raise ValueError("The dataset contains no samples")

    vae = VAEWrapper(decode_batch_size=args.decode_batch_size)
    device = next(vae.vae.parameters()).device
    decoded_chunks = []

    with torch.no_grad():
        for start in range(0, len(x_batch), args.decode_batch_size):
            latents = x_batch[start : start + args.decode_batch_size].to(device)
            decoded_chunks.append(vae.decode(latents).float().cpu())

    images = torch.cat(decoded_chunks)
    images = ((images + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    columns = min(args.columns, len(images))
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2 * columns, 2.3 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index])
            axis.set_title(f"Class {y_batch[index].item()}", fontsize=9)

    figure.tight_layout()
    output_path = args.output or (
        f"{os.path.splitext(args.batch_file)[0]}_preview.png"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {len(images)} decoded samples to {output_path}")


if __name__ == "__main__":
    main()
