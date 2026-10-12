"""
Pytorch port of https://github.com/lambertae/drifting/blob/main/models/mae_model.py

Differences from the JAX version:
- Tensors are NCHW (the JAX code is NHWC). Spatial token order in get_activations
  matches the JAX output.
- Module names mirror the flax param tree, so released checkpoints
  (https://huggingface.co/Goodeat/drifting) load with load_mae_from_jax().
- HF download / ConvNeXt / distributed-init helpers are not ported.
"""

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import msgpack
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# The released latent MAEs were trained on sd-vae-ft-mse samples * 0.18215 (dataset/vae.py)
MAE_LATENT_SCALE = 0.18215

# flax.linen.GroupNorm default epsilon
GN_EPS = 1e-6


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(_choose_gn_groups(channels, max_groups), channels, eps=GN_EPS)


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs with GroupNorm, ReLU, and residual skip."""

    def __init__(self, in_channels: int, filters: int, stride: int = 1, gn_max_groups: int = 32, dropout_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(filters, gn_max_groups)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(filters, gn_max_groups)
        self.drop = nn.Dropout(dropout_prob)
        # JAX only uses (and only has params for) the projection when shapes differ
        if stride != 1 or in_channels != filters:
            self.proj_conv = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.proj_gn = _gn(filters, gn_max_groups)
        else:
            self.proj_conv = None
            self.proj_gn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.drop(y)
        y = self.gn2(self.conv2(y))
        if self.proj_conv is not None:
            residual = self.proj_gn(self.proj_conv(residual))
        return F.relu(residual + y)


class ResNetEncoder(nn.Module):
    """4-stage ResNet encoder producing multi-scale feature maps.

    Returns a dict {conv1, layer1, ..., layer4}, each (B, C_i, H_i, W_i).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
        gn_max_groups: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.gn1 = _gn(base_channels, gn_max_groups)

        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        ch_in = base_channels
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = base_channels * (2 ** stage_idx)
            blocks = [BasicBlock(ch_in, out_ch, stride=stride, gn_max_groups=gn_max_groups, dropout_prob=dropout_prob)]
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(out_ch, out_ch, stride=1, gn_max_groups=gn_max_groups, dropout_prob=dropout_prob))
            self.stages.append(nn.ModuleList(blocks))
            self.stage_norms.append(_gn(out_ch, gn_max_groups))
            ch_in = out_ch

    def forward(self, x: torch.Tensor, return_block_outputs: bool = False):
        feats: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, List[torch.Tensor]] = {}
        x = F.relu(self.gn1(self.conv1(x)))
        feats["conv1"] = x

        for i, (blocks, norm) in enumerate(zip(self.stages, self.stage_norms)):
            layer_name = f"layer{i + 1}"
            outs: List[torch.Tensor] = []
            for block in blocks:
                x = block(x)
                outs.append(x)
            block_outputs[layer_name] = outs
            x = norm(x)
            feats[layer_name] = x
        if return_block_outputs:
            return feats, block_outputs
        return feats


class ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel, padding=kernel // 2, bias=False)
        self.gn = _gn(channels, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        cat_ch = in_channels + skip_channels
        self.concat_norm_fn = nn.GroupNorm(32, cat_ch, eps=GN_EPS)
        self.proj = ConvGNReLU(cat_ch, out_channels, kernel=3)
        self.refine = ConvGNReLU(out_channels, out_channels, kernel=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # jax.image.resize(bilinear) on upsampling == half-pixel bilinear without antialiasing
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm_fn(x)
        x = self.proj(x)
        return self.refine(x)


class UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8
        self.bridge = ConvGNReLU(c5, c5)
        self.up43 = UpBlock(c5, c4, c4)
        self.up32 = UpBlock(c4, c3, c3)
        self.up21 = UpBlock(c3, c2, c2)
        self.up10 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


def patch_input(x: torch.Tensor, input_patch_size: int) -> torch.Tensor:
    """NCHW equivalent of 'b (h1 h2) (w1 w2) c -> b h1 w1 (h2 w2 c)'."""
    if input_patch_size == 1:
        return x
    p = input_patch_size
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 3, 5, 1, 2, 4)  # b h2 w2 c h1 w1
    return x.reshape(b, p * p * c, h // p, w // p)


def make_patch_mask(
    x: torch.Tensor,
    mask_ratio: torch.Tensor,
    patch_size: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Returns a (B, 1, H, W) mask where 1 = masked."""
    b, _, h, w = x.shape
    nh, nw = h // patch_size, w // patch_size
    noise = torch.rand((b, nh, nw), device=x.device, dtype=x.dtype, generator=generator)
    mask = (noise < mask_ratio[:, None, None]).to(x.dtype)
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return mask[:, None]


def safe_std(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var.clamp_min(0.0) + eps)


def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H*W, C), row-major like 'b h w c -> b (h w) c'."""
    return feat.flatten(2).transpose(1, 2)


def _patch_groups(feat: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/s)*(W/s), s*s, C), like 'b (h s1) (w s2) c -> b (h w) (s1 s2) c'."""
    b, c, h, w = feat.shape
    x = feat.reshape(b, c, h // size, size, w // size, size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # b h w s1 s2 c
    return x.reshape(b, (h // size) * (w // size), size * size, c)


class MAEResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Sequence[int] = (2, 2, 2, 2),
        input_patch_size: int = 1,
        with_decoder: bool = True,
        use_bf16: bool = False,  # kept for metadata compatibility; use torch.autocast instead
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.input_patch_size = input_patch_size
        enc_in = in_channels * input_patch_size * input_patch_size
        self.encoder = ResNetEncoder(enc_in, base_channels=base_channels, layers=layers, dropout_prob=dropout_prob)
        self.decoder = UNetDecoder(base_channels, enc_in) if with_decoder else None
        self.fc = nn.Linear(base_channels * 2 ** (len(layers) - 1), num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MAE training loss. x: (B, C, H, W)."""
        if self.decoder is None:
            raise RuntimeError("MAE loss needs the decoder; build with with_decoder=True.")
        x = patch_input(x, self.input_patch_size)
        b = x.shape[0]
        mask_ratio = torch.rand((b,), device=x.device, dtype=x.dtype, generator=generator)
        mask_ratio = mask_ratio * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, mask_ratio, self.patch_size, generator=generator)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in)
        pooled = feats["layer4"].mean(dim=(2, 3))
        logits = self.fc(pooled)
        recon = self.decoder(feats)

        cls_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        mse = (recon.float() - x.float()) ** 2
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (logits.argmax(dim=-1) == labels).float(),
            "mask_ratio": mask.mean(dim=(1, 2, 3)),
        }
        return loss, metrics

    def get_activations(
        self,
        x: torch.Tensor,
        patch_mean_size: Optional[List[int]] = (2, 4),
        patch_std_size: Optional[List[int]] = (2, 4),
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-scale encoder features for the drift loss.

        Args:
            x: latents of shape (B, C, H, W).

        Returns:
            Dict of (B, T, D) tensors, T = spatial token count, D = channels. Keys include
            conv1, layer{1-4}, and aggregated variants (*_mean, *_std, *_mean_{s}, *_std_{s}).
        """
        patch_mean_size = list(patch_mean_size or [])
        patch_std_size = list(patch_std_size or [])

        x = patch_input(x, self.input_patch_size)
        need_blocks = (
            isinstance(every_k_block, (int, float))
            and not math.isinf(float(every_k_block))
            and every_k_block >= 1
        )
        if need_blocks:
            feats, block_outputs = self.encoder(x, return_block_outputs=True)
        else:
            feats = self.encoder(x)
            block_outputs = {}

        out: Dict[str, torch.Tensor] = {}
        out["norm_x"] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]

        def process_feat(name: str, feat: torch.Tensor) -> None:
            _, _, h, w = feat.shape
            out[name] = _to_tokens(feat)
            if use_mean:
                out[f"{name}_mean"] = feat.mean(dim=(2, 3))[:, None, :]
            if use_std:
                out[f"{name}_std"] = safe_std(feat, dim=(2, 3))[:, None, :]
            for size in patch_mean_size:
                if h % size == 0 and w % size == 0:
                    out[f"{name}_mean_{size}"] = _patch_groups(feat, size).mean(dim=2)
            for size in patch_std_size:
                if h % size == 0 and w % size == 0:
                    out[f"{name}_std_{size}"] = safe_std(_patch_groups(feat, size), dim=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                for blk_idx, feat_i in enumerate(block_outputs.get(lname, []), start=1):
                    if blk_idx % k == 0:
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        return out


def activation_fn(model: MAEResNet, x: torch.Tensor, has_scale: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
    """MAE-only port of build_activation_function's activation_fn. x: (B, C, H, W)."""
    feats: Dict[str, torch.Tensor] = {}
    # JAX flattens NHWC; keep that element order
    feats["global"] = x.permute(0, 2, 3, 1).reshape(x.shape[0], 1, -1)
    if has_scale:
        feats["norm_x"] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]
    return {**feats, **model.get_activations(x, **kwargs)}


# --------------------- loading released JAX checkpoints ---------------------
_MSGPACK_EXT_NDARRAY = 1  # flax.serialization._MsgpackExtType.ndarray


def _msgpack_ext_hook(code, data):
    if code == _MSGPACK_EXT_NDARRAY:
        shape, dtype_name, buffer = msgpack.unpackb(data, raw=False)
        if dtype_name == "bfloat16":
            t = torch.frombuffer(bytearray(buffer), dtype=torch.bfloat16).float()
            return t.reshape(shape)
        arr = np.frombuffer(buffer, dtype=np.dtype(dtype_name)).reshape(shape)
        return torch.from_numpy(arr.copy())
    return msgpack.ExtType(code, data)


def _unchunk(tree):
    """Undo flax's chunking of arrays larger than MAX_CHUNK_SIZE."""
    if not isinstance(tree, dict):
        return tree
    if tree.get("__msgpack_chunked_array__"):
        shape = tuple(tree["shape"][str(i)] for i in range(len(tree["shape"])))
        chunks = [tree["chunks"][str(i)] for i in range(len(tree["chunks"]))]
        return torch.cat([c.reshape(-1) for c in chunks]).reshape(shape)
    return {k: _unchunk(v) for k, v in tree.items()}


def load_flax_msgpack(path: str) -> Dict:
    with open(path, "rb") as f:
        tree = msgpack.unpackb(f.read(), ext_hook=_msgpack_ext_hook, raw=False, strict_map_key=False)
    return _unchunk(tree)


def _flatten(tree: Dict, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], torch.Tensor]:
    flat = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix + (k,)))
        else:
            flat[prefix + (k,)] = v
    return flat


def _flax_path_to_torch(path: Tuple[str, ...]) -> str:
    parts = []
    for p in path[:-1]:
        if p.startswith("stages_"):
            parts += ["stages", p.split("_")[1]]
        elif p.startswith("layers_"):
            parts.append(p.split("_")[1])
        elif p.startswith("layer") and p.endswith("_norm"):
            parts += ["stage_norms", str(int(p[len("layer"):-len("_norm")]) - 1)]
        else:
            parts.append(p)
    leaf = {"kernel": "weight", "scale": "weight", "bias": "bias"}[path[-1]]
    return ".".join(parts + [leaf])


def flax_to_torch_state_dict(params: Dict, with_decoder: bool = True) -> Dict[str, torch.Tensor]:
    if "params" in params and isinstance(params["params"], dict):
        params = params["params"]
    state = {}
    for path, value in _flatten(params).items():
        if not with_decoder and path[0] == "decoder":
            continue
        if path[-1] == "kernel":
            if value.ndim == 4:  # (kh, kw, in, out) -> (out, in, kh, kw)
                value = value.permute(3, 2, 0, 1)
            elif value.ndim == 2:  # (in, out) -> (out, in)
                value = value.t()
        state[_flax_path_to_torch(path)] = value.contiguous()
    return state


def mae_from_metadata(metadata: Dict, with_decoder: bool = True) -> MAEResNet:
    cfg = dict(metadata.get("model_config", {}) or {})
    num_classes = int(cfg.pop("num_classes", 1000))
    return MAEResNet(num_classes=num_classes, with_decoder=with_decoder, **cfg)


def load_mae_from_jax(artifact_dir: str, with_decoder: bool = False) -> Tuple[MAEResNet, Dict]:
    """Build MAEResNet from a released artifact dir (metadata.json + ema_params.msgpack)."""
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        metadata = json.load(f)
    model = mae_from_metadata(metadata, with_decoder=with_decoder)
    params = load_flax_msgpack(os.path.join(artifact_dir, "ema_params.msgpack"))
    state = flax_to_torch_state_dict(params, with_decoder=with_decoder)
    del params
    model.load_state_dict(state, strict=True)
    return model, metadata
