"""Save the image drift bank as a png."""

import os

import matplotlib.pyplot as plt
import torch
from utils.drift_util import create_positive_bank
from utils.vae_util import VAEWrapper


DATA_ROOT = "/data/ali/imf_latents/val"
TRAIN_BATCH = "/data/ali/imf_latents/train_overfit1000_10classes.pt"
K_POS = 50
OUT_DIR = "/data/ali/gmd_gens/pos_bank_50_per_class"
DECODE_BATCH = 8


def main():
    ds = torch.load(TRAIN_BATCH, map_location="cpu")
    class_ids = ds["y_batch"].unique(sorted=True).tolist()
    bank = create_positive_bank(DATA_ROOT, class_ids)
    vae = VAEWrapper(decode_batch_size=DECODE_BATCH)
    device = next(vae.vae.parameters()).device
    os.makedirs(OUT_DIR, exist_ok=True)

    for class_id in class_ids:
        key = f"class_{class_id:04d}"
        latents = bank[key][:K_POS]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(latents), DECODE_BATCH):
                batch = latents[start : start + DECODE_BATCH].to(device)
                chunks.append(vae.decode(batch).float().cpu())
        images = torch.cat(chunks, dim=0)
        images = ((images + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()

        n = len(images)
        cols = 10
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows), squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < n:
                ax.imshow(images[i])
        fig.suptitle(f"{key}  |  {n} bank images", fontsize=12)
        path = os.path.join(OUT_DIR, f"{key}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path} ({n} images)")


if __name__ == "__main__":
    main()
