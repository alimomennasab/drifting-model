import argparse
import os
import torch
from latent_dataset import ShardedLatentDataset

# Create a latent dataset of n samples and labels
"""
CUDA_VISIBLE_DEVICES=6 python create_batch.py \
    --data-root "/data/ali/imf_latents/train_imagenet" \
    --n-samples 30 \
    --n-classes 15 \
    --class-select random
"""


def select_labels(available, n_classes, mode, generator):
    """Pick class IDs from sorted `available`.

    ImageNet folders are WordNet-ordered, so consecutive IDs are often similar
    (e.g. many bird species). Prefer ``spaced`` to spread picks across the list.
    """
    if mode == "sequential":
        return available[:n_classes]
    if mode == "random":
        perm = torch.randperm(len(available), generator=generator)[:n_classes]
        return sorted(available[i] for i in perm.tolist())
    if mode == "spaced":
        if n_classes == 1:
            return [available[len(available) // 2]]
        # Evenly spaced indices over the sorted class list.
        indices = [
            round(i * (len(available) - 1) / (n_classes - 1))
            for i in range(n_classes)
        ]

        chosen = []
        used = set()
        for idx in indices:
            if idx not in used:
                chosen.append(available[idx])
                used.add(idx)

        # Fill any remaining from unused slots farthest from chosen.
        if len(chosen) < n_classes:
            remaining = [c for i, c in enumerate(available) if i not in used]
            chosen.extend(remaining[: n_classes - len(chosen)])
        return chosen

    raise ValueError(f"Unknown --class-select mode: {mode!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="/data/ali/imf_latents/train")
    parser.add_argument(
        "--out-dir",
        default="/data/ali/imf_latents/",
    )
    parser.add_argument(
        "--class-select",
        choices=["spaced", "random", "sequential"],
        default="spaced",
        help=(
            "How to choose which classes to keep. "
            "'spaced' spreads picks across available IDs"
            "'random' samples uniformly. "
            "'sequential' takes the first N sorted IDs."
        ),
    )
    args = parser.parse_args()

    if args.n_samples < 1:
        raise ValueError("--n-samples must be positive")
    if args.n_classes < 1:
        raise ValueError("--n-classes must be positive")
    if args.n_classes > args.n_samples:
        raise ValueError("--n-classes cannot exceed --n-samples")

    out_path = os.path.join(
        args.out_dir,
        f"train_overfit{args.n_samples}_{args.n_classes}classes.pt",
    )

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    ds = ShardedLatentDataset(args.data_root, use_flip=False)

    # See what labels are present in the shards
    available = set()
    for path in ds.shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        available.update(int(y) for y in shard["labels"].tolist())
    available = sorted(available)
    if args.n_classes > len(available):
        raise ValueError(
            f"--n-classes={args.n_classes} but dataset only has "
            f"{len(available)} classes: {available}"
        )

    labels = select_labels(available, args.n_classes, args.class_select, generator)
    print(
        f"Available classes: {len(available)}; "
        f"selected ({args.class_select}): {labels}"
    )

    samples_per_label, remainder = divmod(args.n_samples, args.n_classes)
    target_counts = {
        label: samples_per_label + (index < remainder)
        for index, label in enumerate(labels)
    }
    collected_counts = {label: 0 for label in labels}

    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        if y not in target_counts or collected_counts[y] >= target_counts[y]:
            continue
        xs.append(x)
        ys.append(y)
        collected_counts[y] += 1
        if len(xs) == args.n_samples:
            break

    if len(xs) != args.n_samples:
        raise RuntimeError(
            f"Found only {len(xs)} of {args.n_samples} requested samples. "
            f"Collected per class: {collected_counts}"
        )

    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.tensor(ys, dtype=torch.long)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_file = f"{out_path}.tmp"
    torch.save({"x_batch": x_batch, "y_batch": y_batch}, tmp_file)
    os.replace(tmp_file, out_path)
    print(
        f"Saved {len(x_batch)} samples across "
        f"{y_batch.unique().numel()} classes to {out_path}"
    )
    print(f"Samples per class: {collected_counts}")
    print(f"x_batch shape: {tuple(x_batch.shape)}")
    print(f"y_batch shape: {tuple(y_batch.shape)}")


if __name__ == "__main__":
    main()