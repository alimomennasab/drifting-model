import glob
import os

import matplotlib.pyplot as plt
import torch

from data.latent_dataset import VAE_MEAN, VAE_STD
from utils.vae_util import VAEWrapper


VAL_LATENT_ROOT = "/data/ali/imf_latents/val"
ZHUYU_LATENT_SCALE = 0.18125 # zhuyu MeanFlow scales latents by 0.18125 


def _shard_paths(data_root):
    paths = sorted(glob.glob(os.path.join(data_root, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(
            f"No shard_*.pt under {data_root}. Cache val latents there "
            f"(same dict as train: images [N,8,32,32] mean||std, labels [N])."
        )
    return paths


def class_ids_from_val_root(data_root=VAL_LATENT_ROOT):
    """Sorted class ids present in cached val shards."""
    ids = set()
    for path in _shard_paths(data_root):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        ids.update(int(c) for c in shard["labels"].unique().tolist())
    if not ids:
        raise RuntimeError(f"No labels in shards under {data_root}")
    return sorted(ids)


def _moments_to_model_latents(mean, latent_norm):
    """[N,4,H,W] VAE mean -> model-space latents."""
    if latent_norm == "zhuyu":
        return mean * ZHUYU_LATENT_SCALE
    mean_ = VAE_MEAN.view(1, -1, 1, 1)
    std_ = VAE_STD.view(1, -1, 1, 1)
    return (mean - mean_) / std_


def create_positive_bank(
    data_root=VAL_LATENT_ROOT,
    class_ids=None,
    max_per_class=None,
    latent_norm="imf",
):
    """Load a positive bank from cached val latent shards.

    Expects ``data_root/shard_*.pt`` like ``imf_latents/train``:
    ``images`` (N, 8, 32, 32) mean||std and ``labels`` (N,).
    Default root is ``/data/ali/imf_latents/val``. No VAE encode.

    ``latent_norm="imf"``: (mean - μ) / σ (iMF / VAEWrapper).
    ``latent_norm="zhuyu"``: mean * 0.18125 (official MeanFlow).

    Returns {"class_XXXX": [N, 4, H, W]}.
    """
    if class_ids is None:
        class_ids = class_ids_from_val_root(data_root)
    class_ids = sorted(int(c) for c in class_ids)
    buckets = {c: [] for c in class_ids}
    kept = {c: 0 for c in class_ids}

    for path in _shard_paths(data_root):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        labels = shard["labels"]
        images = shard["images"].float()
        for c in class_ids:
            if max_per_class is not None and kept[c] >= max_per_class:
                continue
            idx = (labels == c).nonzero().squeeze(1)
            if len(idx) == 0:
                continue
            if max_per_class is not None:
                idx = idx[: max_per_class - kept[c]]
            mean = images[idx]
            if mean.shape[1] == 8:
                mean, _ = mean.chunk(2, dim=1)
            elif mean.shape[1] != 4:
                raise RuntimeError(
                    f"{path}: expected 4 or 8 image channels, got {tuple(mean.shape)}"
                )
            buckets[c].append(_moments_to_model_latents(mean, latent_norm))
            kept[c] += len(idx)
        if max_per_class is not None and all(kept[c] >= max_per_class for c in class_ids):
            break

    bank = {}
    for c in class_ids:
        if not buckets[c]:
            raise RuntimeError(f"No val latents for class {c} under {data_root}")
        bank[f"class_{c:04d}"] = torch.cat(buckets[c], dim=0)
        print(f"val bank class_{c:04d}: {bank[f'class_{c:04d}'].shape[0]} images")
    return bank


def plot_pos_bank(bank, output_path, decode_batch_size=8):
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


# --------------------- DRIFT COMPUTATION HELPERS ---------------------
def compute_attraction(y, real_data, tau):
    # y: one-step sample
    # real_data: real data batch
    # tau: temperature
    squared_distances = torch.cdist(y, real_data).square()
    weights = torch.softmax(-squared_distances / (2 * tau ** 2), dim=-1)
    attraction = weights @ real_data

    return attraction


def compute_repulsion(y, sigma_r=1.5):
    # y: one-step sample

    squared_distances = torch.cdist(y, y).square()
    weights = torch.exp(-squared_distances / (2 * sigma_r ** 2))
    self_mask = torch.eye(
        y.shape[0], dtype=torch.bool, device=y.device
    )
    weights = weights.masked_fill(self_mask, 0)

    weight_sum = weights.sum(dim=-1, keepdim=True)
    weighted_neighbors = weights @ y

    repulsion = (weight_sum * y - weighted_neighbors) / weight_sum.clamp_min(1e-8)

    return repulsion


def compute_sharpener_drift(
    y,
    real_data,
    tau,
    lambda_rep=0.2,
    sigma_r=1.5,
):
    # y: one-step sample
    # real_data: real (positive) data sample
    # tau: temperature

    attraction = compute_attraction(y, real_data, tau)
    repulsion = compute_repulsion(y, sigma_r)
    v = (attraction - y) - lambda_rep * repulsion
    return v


def computeV(x, y_pos, y_neg, T):
    # x: [HW, B, C] 
    # y_pos: [HW, N_pos, C]
    # y_neg: [HW, N_neg, C]
    # T: temperature
    num_x = x.shape[-2]
    num_pos = y_pos.shape[-2]
    num_neg = y_neg.shape[-2]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # [HW, B, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [HW, B, N_neg]

    # ignore self in distance computation (if y_neg is x)
    if y_neg is x:
        if num_x != num_neg:
            raise ValueError("Self-masking requires equal x and y_neg batch sizes.")

        # - we create a boolean identity matrix of size [B, B]
        # [[True,  False, False],
        # [False, True,  False],
        # [False, False, True ]]
        # - we add a dim -> [1, B, B], so we can broadcast to every distance in dist_neg
        # - we make all true entries infinity. these true entries are each sample's distance with itself
        # - later, logit_neg = -dist_neg / T makes all inf -> -inf
        # - then, when softmaxed, these -inf's become 0
        # - so, each sample's distance with itself is not utilized in the V calculation 

        self_mask = torch.eye(
            num_x, dtype=torch.bool, device=x.device
        ).unsqueeze(0)
        dist_neg = dist_neg.masked_fill(self_mask, float("inf"))

    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T

    # concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=-1)

    # normalize along both dimensions
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = torch.sqrt(A_row * A_col)

    # back to [HW, B, N_pos] and [HW, B, N_neg]
    A_pos, A_neg = torch.split(A, [num_pos, num_neg], dim=-1)

    # compute the weights
    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)

    drift_pos = W_pos @ y_pos  # [HW, B, C]
    drift_neg = W_neg @ y_neg  # [HW, B, C]
    
    V = drift_pos - drift_neg

    return V

