import argparse
import os
import torch
from latent_dataset import ShardedLatentDataset

# Create a latent dataset of n samples and labels
# note: we only have a dataset containing 10 classes (0-9) for now

# python create_batch.py --n-samples 50 --n-classes 10
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
    args = parser.parse_args()

    if args.n_samples < 1:
        raise ValueError("--n-samples must be positive")
    if not 1 <= args.n_classes <= min(args.n_samples, 10):
        raise ValueError(
            "--n-classes must be between 1 and min(n-samples, 10)"
        )

    out_path = os.path.join(
        args.out_dir,
        f"train_overfit{args.n_samples}_{args.n_classes}classes.pt",
    )

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    labels = torch.randperm(10, generator=generator)[:args.n_classes].tolist()
    print("labels: ", labels)

    samples_per_label, remainder = divmod(args.n_samples, args.n_classes)
    target_counts = {
        label: samples_per_label + (index < remainder)
        for index, label in enumerate(labels)
    }
    collected_counts = {label: 0 for label in labels}

    ds = ShardedLatentDataset(args.data_root, use_flip=False)
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