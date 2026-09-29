"""Faster alternative to download_imagenet.py: fetch the parquet files
directly (several in parallel) instead of streaming one record at a time.

The full train split is 146 GB, so each parquet file is deleted as soon as the
JPEG bytes of the kept classes are written out. Afterwards, the first
MAX_PER_CLASS images of each class in stream order (parquet file, then row)
are numbered 0000000.jpg, 0000001.jpg, ... exactly like download_imagenet.py.
The validation split has only 50 images per class, so all of them are kept.

Safe to interrupt and rerun: finished parquet files are listed in
<STAGING_ROOT>/<split>/done.txt and skipped.

    python download_imagenet_parquet.py --split train
    python download_imagenet_parquet.py --split validation
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

KEEP_LABELS = list(range(200))
MAX_PER_CLASS = 1100
REPO = "ILSVRC/imagenet-1k"
STAGING_ROOT = "/data/ali/imagenet_staging"
OUT_ROOT = "/data/ali/imagenet_parquet"
NUM_WORKERS = 8


def process_file(file_idx, filename, token, staging):
    """Download one parquet file, write its kept-class images, delete it."""
    path = hf_hub_download(
        REPO,
        filename,
        repo_type="dataset",
        token=token,
        local_dir=os.path.join(staging, "parquet"),
    )
    table = pq.read_table(path, columns=["image", "label"])
    table = table.append_column("row", pa.array(range(table.num_rows)))
    table = table.filter(pc.is_in(table["label"], value_set=pa.array(KEEP_LABELS)))

    for ex in table.to_pylist():
        cls_dir = os.path.join(staging, "images", f"class_{ex['label']:04d}")
        os.makedirs(cls_dir, exist_ok=True)
        # file index + row keeps the stream order when sorted by name
        name = f"{file_idx:03d}_{ex['row']:05d}.jpg"
        with open(os.path.join(cls_dir, name), "wb") as f:
            f.write(ex["image"]["bytes"])

    os.remove(path)
    return filename


def finalize(staging, out_dir):
    """Move the first MAX_PER_CLASS images of each class into out_dir."""
    images_dir = os.path.join(staging, "images")
    for cls in sorted(os.listdir(images_dir)):
        files = sorted(os.listdir(os.path.join(images_dir, cls)))[:MAX_PER_CLASS]
        out_cls = os.path.join(out_dir, cls)
        os.makedirs(out_cls, exist_ok=True)
        for i, name in enumerate(files):
            os.rename(
                os.path.join(images_dir, cls, name),
                os.path.join(out_cls, f"{i:07d}.jpg"),
            )
        print(f"{cls}: {len(files)} images")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "validation"], default="train")
    args = p.parse_args()
    staging = os.path.join(STAGING_ROOT, args.split)
    out_dir = os.path.join(OUT_ROOT, args.split)

    load_dotenv(Path(__file__).resolve().parent / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Add HF_TOKEN to drifting-model/data/.env")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise RuntimeError(f"{out_dir} is not empty; move the old download aside first")

    files = sorted(
        f
        for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
        if f.startswith(f"data/{args.split}-") and f.endswith(".parquet")
    )
    os.makedirs(staging, exist_ok=True)
    done_path = os.path.join(staging, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    todo = [(i, f) for i, f in enumerate(files) if f not in done]
    print(f"{len(files)} {args.split} parquet files, {len(done)} done, {len(todo)} to go")

    with ThreadPoolExecutor(NUM_WORKERS) as pool, open(done_path, "a") as done_file:
        futures = [pool.submit(process_file, i, f, token, staging) for i, f in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="file"):
            done_file.write(fut.result() + "\n")
            done_file.flush()

    finalize(staging, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
