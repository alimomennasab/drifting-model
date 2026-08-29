"""On-disk sharded latent dataset for ImageNet (PyTorch)."""

import glob
import os
import random
from collections import OrderedDict

import torch
import torch.utils.data as data


# SD-VAE-MSE channel statistics, matching the iMF repos.
VAE_MEAN = torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409])
VAE_STD = torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003])


class _LRUShardCache:
    """Simple LRU cache for loaded shards (process-local)."""

    def __init__(self, max_items: int = 4):
        self.max_items = max_items
        self.d = OrderedDict()

    def get(self, key, loader):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        shard = loader()
        self.d[key] = shard
        if len(self.d) > self.max_items:
            self.d.popitem(last=False)
        return shard


class ShardedLatentDataset(data.Dataset):
    """Loads .pt shards storing (N, 8, 32, 32) latent (mean,std) and (N,) labels.

    Each ``__getitem__`` samples a latent ~ mean + std * eps, normalizes by the
    VAE channel stats, and optionally flips horizontally.
    """

    def __init__(
        self,
        root: str,
        use_flip: bool = True,
        shard_glob: str = "shard_*.pt",
        cache_size: int = 4,
        items_per_shard: int = 2048,
    ):
        self.root = root
        self.shard_paths = sorted(glob.glob(os.path.join(root, shard_glob)))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards under {root!r}")
        self.use_flip = use_flip
        # Per-shard length is fixed except possibly the last; we lazily fix
        # it on first access.
        self._items_per_shard = items_per_shard
        self._actual_sizes = [None] * len(self.shard_paths)
        # Heuristic length: assume all shards full except inspect the last.
        last = torch.load(self.shard_paths[-1], map_location="cpu", weights_only=False)
        self._actual_sizes[-1] = int(last["labels"].shape[0])
        for i in range(len(self.shard_paths) - 1):
            self._actual_sizes[i] = items_per_shard
        # Precompute cumulative starts for O(log N) lookup.
        self._starts = [0]
        for s in self._actual_sizes:
            self._starts.append(self._starts[-1] + s)
        self._total = self._starts[-1]
        self._cache = _LRUShardCache(max_items=cache_size)
        self.mean = VAE_MEAN.view(-1, 1, 1)
        self.std = VAE_STD.view(-1, 1, 1)

    def __len__(self):
        return self._total

    def __repr__(self):
        return (
            f"ShardedLatentDataset(root={self.root}, shards={len(self.shard_paths)},"
            f" n={len(self)}, flip={self.use_flip})"
        )

    def _shard_for(self, idx):
        # Binary search on cumulative starts.
        lo, hi = 0, len(self._starts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._starts[mid] <= idx:
                lo = mid
            else:
                hi = mid
        return lo, idx - self._starts[lo]

    def _load_shard(self, s_idx):
        path = self.shard_paths[s_idx]
        return self._cache.get(
            path, lambda: torch.load(path, map_location="cpu", weights_only=False)
        )

    def __getitem__(self, idx):
        s_idx, k = self._shard_for(idx)
        shard = self._load_shard(s_idx)
        # Clamp k for the (possibly short) last shard.
        k = min(k, shard["images"].shape[0] - 1)
        img = shard["images"][k].float()  # (8, H, W)
        mean, std = torch.chunk(img, 2, dim=0)
        latent = mean + std * torch.randn_like(std)
        latent = (latent - self.mean) / self.std
        if self.use_flip and random.random() < 0.5:
            latent = torch.flip(latent, dims=[-1])
        label = int(shard["labels"][k].item())
        return latent.float(), label
