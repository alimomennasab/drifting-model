"""Build a held-out positive bank by skipping the first K samples per class.

create_batch.py takes the first N samples of each class for overfit training.
This script skips those early samples and keeps the rest.

    CUDA_VISIBLE_DEVICES=1 python create_positive_bank.py \
        --train-batch /data/ali/imf_latents/train_overfit30_10classes.pt \
        --skip-first 5 \
        --out-name positive_bank_30samples_10classes.pt
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from latent_dataset import ShardedLatentDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.vae_util import VAEWrapper


def plot_bank(bank, output_path, decode_batch_size=8):
    """Decode bank latents and save one row per class as a PNG."""
    vae = VAEWrapper(decode_batch_size=decode_batch_size)
    device = next(vae.vae.parameters()).device
    keys = sorted(bank.keys())
    decoded = {}

    with torch.no_grad():
        for key in keys:
            latents = bank[key]
            chunks = []
            for start in range(0, len(latents), decode_batch_size):
                batch = latents[start : start + decode_batch_size].to(device)
                chunks.append(vae.decode(batch).float().cpu())
            images = torch.cat(chunks)
            images = ((images + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()
            decoded[key] = images

    max_n = max(len(decoded[k]) for k in keys)
    fig, axes = plt.subplots(
        len(keys),
        max_n,
        figsize=(1.6 * max_n, 1.8 * len(keys)),
        squeeze=False,
    )

    for row, key in enumerate(keys):
        images = decoded[key]
        for col in range(max_n):
            axes[row, col].axis("off")
            if col < len(images):
                axes[row, col].imshow(images[col])
        axes[row, 0].set_ylabel(key, fontsize=8, rotation=0, ha="right", va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/ali/imf_latents/train_imagenet")
    p.add_argument("--train-batch", required=True)
    p.add_argument("--skip-first", type=int, default=5)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="/data/ali/imf_latents/")
    p.add_argument("--out-name", required=True)
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    train = torch.load(args.train_batch, map_location="cpu", weights_only=False)
    class_ids = sorted(int(c) for c in train["y_batch"].unique().tolist())
    wanted = set(class_ids)
    print(f"Classes: {class_ids}")

    torch.manual_seed(args.seed)
    ds = ShardedLatentDataset(args.data_root, use_flip=False)

    seen = defaultdict(int)
    buckets = {c: [] for c in class_ids}

    for i in range(len(ds)):
        x, y = ds[i]
        if y not in wanted:
            continue
        seen[y] += 1
        if seen[y] <= args.skip_first:
            continue
        if args.max_per_class is not None and len(buckets[y]) >= args.max_per_class:
            continue
        buckets[y].append(x)
        if args.max_per_class is not None and all(
            len(buckets[c]) >= args.max_per_class for c in class_ids
        ):
            break

    bank = {}
    for c in class_ids:
        if not buckets[c]:
            raise RuntimeError(
                f"No held-out samples for class {c} after skip_first={args.skip_first}"
            )
        bank[f"class_{c:04d}"] = torch.stack(buckets[c])

    out_path = os.path.join(args.out_dir, args.out_name)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(bank, out_path)

    print(f"Saved {out_path}")
    for key, tensor in bank.items():
        print(f"  {key}: {tuple(tensor.shape)}")

    if not args.no_plot:
        plot_path = os.path.splitext(out_path)[0] + ".png"
        plot_bank(bank, plot_path, decode_batch_size=args.decode_batch_size)


if __name__ == "__main__":
    main()
