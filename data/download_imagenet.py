"""Download the first N ImageNet-1k classes from Hugging Face as JPEG folders.

Output layout: <out_root>/<split>/class_{label:04d}/{idx:07d}.jpg

Safe to interrupt and rerun: the stream always restarts from the beginning,
so each class skips as many images as it already has on disk before saving.
This relies on the (unshuffled) stream order being the same on every run.

    python download_imagenet.py
"""

import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

KEEP_LABELS = set(range(200))
MAX_PER_CLASS = 1100
OUT_ROOT = "/data/ali/imagenet"
SPLITS = ["train"]


def existing_counts(split_dir):
    """Number of .jpg files already saved for each kept class."""
    counts = {}
    for y in KEEP_LABELS:
        cls_dir = os.path.join(split_dir, f"class_{y:04d}")
        counts[y] = (
            sum(name.endswith(".jpg") for name in os.listdir(cls_dir))
            if os.path.isdir(cls_dir)
            else 0
        )
    return counts


def download_split(split):
    split_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(split_dir, exist_ok=True)

    counts = existing_counts(split_dir)
    target_total = len(KEEP_LABELS) * MAX_PER_CLASS
    total_saved = sum(counts.values())
    remaining = {y for y in KEEP_LABELS if counts[y] < MAX_PER_CLASS}

    print(
        f"{split}: resume with {total_saved}/{target_total} saved; "
        f"{len(remaining)} classes still need images"
    )
    if not remaining:
        print(f"{split}: already complete")
        return

    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True).decode(False) # faster download with no decoding
    seen = {y: 0 for y in KEEP_LABELS}
    pbar = tqdm(ds, desc=f"{split} stream", unit="img")

    for ex in pbar:
        if not remaining:
            break

        y = ex["label"]
        if y not in remaining:
            continue

        # skip images this class already saved in a previous run
        seen[y] += 1
        if seen[y] <= counts[y]:
            continue

        cls_dir = os.path.join(split_dir, f"class_{y:04d}")
        os.makedirs(cls_dir, exist_ok=True)
        with open(os.path.join(cls_dir, f"{counts[y]:07d}.jpg"), "wb") as f:
            f.write(ex["image"]["bytes"])
        counts[y] += 1
        total_saved += 1

        if counts[y] >= MAX_PER_CLASS:
            remaining.discard(y)

        pbar.set_postfix(
            saved=total_saved,
            need=target_total - total_saved,
            open_cls=len(remaining),
            refresh=False,
        )

    print(
        f"{split}: saved {total_saved}/{target_total} images "
        f"across {len(KEEP_LABELS)} classes (max {MAX_PER_CLASS} each)"
    )


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Add HF_TOKEN to drifting-model/data/.env")
    login(token=token)

    for split in SPLITS:
        download_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
