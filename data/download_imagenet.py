import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent / ".env")
token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("Add HF_TOKEN to drifting-model/data/.env")
login(token=token)

# First N ImageNet classes (labels 0 .. n-1).
keep_labels = set(range(200))

out_root = "/data/ali/imagenet"
splits = ["train"]
max_per_class = 1000


def existing_counts(split_dir: str) -> dict[int, int]:
    counts = {k: 0 for k in keep_labels}
    if not os.path.isdir(split_dir):
        return counts
    for y in keep_labels:
        cls_dir = os.path.join(split_dir, f"class_{y:04d}")
        if not os.path.isdir(cls_dir):
            continue
        counts[y] = sum(
            1 for name in os.listdir(cls_dir) if name.endswith(".jpg")
        )
    return counts


for split in splits:
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)

    counts = existing_counts(split_dir)
    total_saved = sum(counts.values())
    target_total = len(keep_labels) * max_per_class
    remaining = {
        y for y in keep_labels if counts[y] < max_per_class
    }

    print(
        f"{split}: resume with {total_saved}/{target_total} saved; "
        f"{len(remaining)} classes still need images"
    )
    if not remaining:
        print(f"{split}: already complete")
        continue

    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    pbar = tqdm(ds, desc=f"{split} stream", unit="img")

    for ex in pbar:
        if not remaining:
            break

        y = ex["label"]
        if y not in remaining:
            continue

        cls_dir = os.path.join(split_dir, f"class_{y:04d}")
        os.makedirs(cls_dir, exist_ok=True)

        idx = counts[y]
        ex["image"].save(os.path.join(cls_dir, f"{idx:07d}.jpg"), quality=95)
        counts[y] += 1
        total_saved += 1

        if counts[y] >= max_per_class:
            remaining.discard(y)

        pbar.set_postfix(
            saved=total_saved,
            need=target_total - total_saved,
            open_cls=len(remaining),
            refresh=False,
        )

    print(
        f"{split}: saved {total_saved}/{target_total} images "
        f"across {len(keep_labels)} classes (max {max_per_class} each)"
    )

print("Done.")
