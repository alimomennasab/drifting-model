"""Pre-compute SD-VAE latents for ImageNet train split, saved as shards.

Output layout under <output_dir>/train/:
  shard_{rank:02d}_{idx:05d}.pt  -> dict(images=(N,8,32,32) fp16, labels=(N,))

Run:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 --master-port=29500 \
  prepare_latents.py \
  --imagenet-root /data/ali/imagenette-train \
  --output-dir /data/ali/imf_latents2

  OR

  CUDA_VISIBLE_DEVICES=6 python prepare_latents.py \
  --imagenet-root /data/ali/imagenet \
  --output-dir /data/ali/imf_latents_imagenet
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from PIL import Image
from diffusers.models import AutoencoderKL
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy : cy + image_size, cx : cx + image_size])


def init_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def load_existing_shards(out_dir, rank, shard_size):
    """Validate this rank's shards and return resume state.

    A short final shard is retained as the in-memory prefix of the next shard so
    resuming never leaves a partial shard in the middle of the dataset.
    """
    pattern = os.path.join(out_dir, f"shard_r{rank:02d}_*.pt")
    paths = sorted(glob.glob(pattern))
    total_samples = 0
    partial_images = []
    partial_labels = []

    for idx, path in enumerate(paths):
        expected = os.path.join(out_dir, f"shard_r{rank:02d}_{idx:05d}.pt")
        if path != expected:
            raise RuntimeError(
                f"Non-contiguous shards for rank {rank}: expected {expected}, found {path}"
            )

        try:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            images = shard["images"]
            labels = shard["labels"]
        except Exception as exc:
            raise RuntimeError(f"Cannot resume from invalid shard {path}: {exc}") from exc

        if images.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Mismatched images/labels in {path}: {images.shape[0]} vs {labels.shape[0]}"
            )
        count = int(labels.shape[0])
        if count <= 0 or count > shard_size:
            raise RuntimeError(f"Invalid sample count in {path}: {count}")
        if count < shard_size and idx != len(paths) - 1:
            raise RuntimeError(f"Partial shard is not last for rank {rank}: {path}")

        total_samples += count
        if count < shard_size:
            partial_images = [images]
            partial_labels = [labels]

    next_shard_idx = len(paths)
    if partial_images:
        # Fill and atomically replace the existing partial shard.
        next_shard_idx -= 1

    return total_samples, next_shard_idx, partial_images, partial_labels


def save_shard(path, images, labels):
    """Atomically save a shard so interrupted writes are not treated as valid."""
    tmp_path = f"{path}.tmp"
    torch.save({"images": images, "labels": labels}, tmp_path)
    os.replace(tmp_path, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--vae-type", default="mse")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=2048,
                   help="Number of samples per output shard.")
    args = p.parse_args()

    rank, world, local_rank = init_dist()
    device = torch.device("cuda", local_rank)

    out_train = os.path.join(args.output_dir, "train")
    if rank == 0:
        os.makedirs(out_train, exist_ok=True)
    if world > 1:
        dist.barrier()

    transform = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.ImageFolder(
        os.path.join(args.imagenet_root, "train"),
        transform=transform,
        loader=pil_loader,
    )
    if rank == 0:
        print(f"Dataset size: {len(ds)}")
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    )
    rank_indices = list(iter(sampler))
    resumed_samples, shard_idx, shard_imgs, shard_labels = load_existing_shards(
        out_train, rank, args.shard_size
    )
    if resumed_samples > len(rank_indices):
        raise RuntimeError(
            f"Rank {rank} has {resumed_samples} saved samples but only "
            f"{len(rank_indices)} assigned dataset samples"
        )

    remaining_indices = rank_indices[resumed_samples:]
    remaining_ds = torch.utils.data.Subset(ds, remaining_indices)
    print(
        f"Rank {rank}: found {resumed_samples} existing samples in "
        f"{shard_idx + (1 if shard_imgs else 0)} shards; "
        f"{len(remaining_indices)} samples remain."
    )
    loader = torch.utils.data.DataLoader(
        remaining_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
    vae.eval()
    for p_ in vae.parameters():
        p_.requires_grad = False
    if hasattr(vae, "decoder"):
        del vae.decoder

    shard_imgs = []
    shard_labels = []
    shard_idx = 0
    total_written = 0
    newly_processed = 0
    pbar = tqdm(loader, disable=rank != 0)
    with torch.no_grad():
        for imgs, labels in pbar:
            newly_processed += int(labels.shape[0])
            imgs = imgs.to(device, non_blocking=True)
            posterior = vae.encode(imgs).latent_dist
            mean = posterior.mean.float()
            std = posterior.std.float()
            cat = torch.cat([mean, std], dim=1).cpu().to(torch.float16)  # (B, 8, 32, 32)
            shard_imgs.append(cat)
            shard_labels.append(labels.to(torch.int32))
            total = sum(t.shape[0] for t in shard_imgs)
            if total >= args.shard_size:
                imgs_cat = torch.cat(shard_imgs, dim=0)[: args.shard_size]
                labels_cat = torch.cat(shard_labels, dim=0)[: args.shard_size]
                fp = os.path.join(out_train, f"shard_r{rank:02d}_{shard_idx:05d}.pt")
                save_shard(fp, imgs_cat, labels_cat)
                shard_idx += 1
                # remainder
                rest_imgs = torch.cat(shard_imgs, dim=0)[args.shard_size:]
                rest_labels = torch.cat(shard_labels, dim=0)[args.shard_size:]
                shard_imgs = [rest_imgs] if rest_imgs.shape[0] > 0 else []
                shard_labels = [rest_labels] if rest_labels.shape[0] > 0 else []
                if rank == 0:
                    pbar.set_postfix({"shards": shard_idx})

    if shard_imgs:
        imgs_cat = torch.cat(shard_imgs, dim=0)
        labels_cat = torch.cat(shard_labels, dim=0)
        fp = os.path.join(out_train, f"shard_r{rank:02d}_{shard_idx:05d}.pt")
        save_shard(fp, imgs_cat, labels_cat)
        shard_idx += 1

    if world > 1:
        dist.barrier()
    print(
        f"Rank {rank}: complete with {shard_idx} shards; "
        f"resumed={resumed_samples}, newly_processed={newly_processed}."
    )


if __name__ == "__main__":
    main()
