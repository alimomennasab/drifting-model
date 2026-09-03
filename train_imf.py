"""Overfit a tiny set of latents to validate the training objective end-to-end.

We load n samples from a saved batch file and train on them for many steps,
expecting loss_u and loss_v to collapse near zero. Run twice -- once with DDE,
once with JVP -- to compare estimators.

Usage:
    CUDA_VISIBLE_DEVICES=2 python overfit_test.py --derivative dde
    CUDA_VISIBLE_DEVICES=2 python overfit_test.py --derivative jvp

    CUDA_VISIBLE_DEVICES=5 python overfit_test.py --derivative dde \\
        --experiment-name x_pred_lpips_ploss --p-loss --p-metric lpips

    CUDA_VISIBLE_DEVICES=7 python overfit_test.py --derivative dde \\
        --experiment-name x_pred_lpips_ploss \\
        --p-loss \\
        --p-metric lpips \\
        --steps 20000 \\
        --optimizer muon \\
        --batch-file /data/ali/imf_latents/train_overfit30_5classes.pt
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.optimizer import (
    build_optimizers,
    optimizer_state_dict,
    step_optimizers,
    zero_grad_optimizers,
)

from models.sdpa_jvp_patch import install as _install_sdpa_jvp
_install_sdpa_jvp()

from imf import iMeanFlow


def plot_losses(log_lines, checkpoint_path, perceptual_metric=None):
    """Plot logged losses next to the model checkpoint."""
    histories = {"step": [], "loss": [], "loss_u": [], "loss_v": [], "loss_p": []}
    for line in log_lines:
        fields = dict(re.findall(r"(\w+)=\s*([^\s]+)", line))
        histories["step"].append(int(fields["step"]))
        for name in ("loss", "loss_u", "loss_v"):
            histories[name].append(float(fields[name]))
        if "loss_p" in fields:
            histories["loss_p"].append(float(fields["loss_p"]))

    plots = [
        ("loss", "Total loss"),
        ("loss_u", "U loss"),
        ("loss_v", "V loss"),
    ]
    if histories["loss_p"]:
        metric_name = (perceptual_metric or "Perceptual").capitalize()
        plots.append(("loss_p", f"{metric_name} perceptual loss"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, title) in zip(axes.flat, plots):
        ax.plot(histories["step"], histories[name], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(plots):]:
        ax.axis("off")

    fig.suptitle(os.path.basename(checkpoint_path))
    fig.tight_layout()
    plot_path = f"{os.path.splitext(checkpoint_path)[0]}_losses.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--derivative", choices=["dde", "jvp"], required=True)
    p.add_argument("--data-root", default="/data/ali/imf_latents/train")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--muon-learning-rate", type=float, default=0.02)
    p.add_argument("--dde-eps", type=float, default=1e-3)
    p.add_argument("--amp", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/data/ali/imf_runs")
    p.add_argument(
        "--batch-file",
        default="/data/ali/imf_latents/train_overfit10.pt",
        help="Reuse the same n data samples for each experiment",
    )
    p.add_argument("--experiment-name", type=str)
    p.add_argument("--p-loss", action="store_true", help="Use p_loss to run with perceptual loss.")
    p.add_argument("--p-metric", type=str, choices=["lpips", "eucl"])

    args = p.parse_args()
    # Fixed optimizer defaults expected by utils.optimizer.build_optimizers.
    args.learning_rate = args.lr
    args.adam_b1 = 0.9
    args.adam_b2 = 0.95
    args.adam_eps = 1e-8
    args.weight_decay = 0.0
    args.muon_momentum = 0.95
    args.muon_weight_decay = 0.0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    batch = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    x_batch = batch["x_batch"].to(device)
    y_batch = batch["y_batch"].to(device)
    n_samples = len(x_batch)
    n_classes = int(y_batch.unique().numel())
    print(f"Overfit batch: x={x_batch.shape}, labels={y_batch.tolist()}")

    # We want a "real" batch matching iMF's setup (batch=1024 in paper) — but
    # since we only have n_samples, we'll tile them. iMF needs a non-trivial
    # batch because data_proportion / cfg_beta both sample per-element. We'll
    # use the smallest tile that keeps overfit sensible.
    tile = max(8, n_samples)
    x_tile = x_batch.repeat(tile // n_samples + 1, 1, 1, 1)[:tile].contiguous()
    y_tile = y_batch.repeat(tile // n_samples + 1)[:tile].contiguous()

    model = iMeanFlow(
        model_str="imfDiT_B_2",
        derivative=args.derivative,
        dde_eps=args.dde_eps,
        eval_mode=False,
        p_loss=args.p_loss,
        parametrization="xpred",
        perceptual_metric=args.p_metric,
    ).to(device)
    optimizers, optimizer_summary = build_optimizers(model, args)
    print(f"Optimizer: {optimizer_summary}")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    print(f"Starting {args.steps}-step overfit run with derivative={args.derivative}")
    t0 = time.time()
    log_lines = []
    for step in range(1, args.steps + 1):
        zero_grad_optimizers(optimizers)
        if amp_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                loss, logs = model(x_tile, y_tile)
        else:
            loss, logs = model(x_tile, y_tile)
        loss.backward()
        step_optimizers(optimizers, scaler=None)

        if step == 1 or step % args.log_every == 0:
            msg = (
                f"step={step:>5d} loss={logs['loss'].item():.4f} "
                f"loss_u={logs['loss_u'].item():.6f} "
                f"loss_v={logs['loss_v'].item():.6f} "
            )
            if args.p_loss:
                msg += f"loss_p={logs['loss_perceptual'].mean().item():.6f} "
            msg += f"t={time.time() - t0:.1f}s"
            print(msg, flush=True)
            log_lines.append(msg)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "overfit.log")
    with open(log_path, "a") as f:
        timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        f.write(
            f"\n=== {args.derivative.upper()} overfit ({args.steps} steps): "
            f"experiment {args.experiment_name} [{timestamp}] ===\n"
        )
        f.write("\n".join(log_lines) + "\n")

    checkpoint_path = os.path.join(
        args.out,
        f"overfit_{args.derivative}_{args.experiment_name}_{args.optimizer}_"
        f"{args.steps}steps_{n_samples}samples{n_classes}classes.pt",
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer_state_dict(optimizers),
            "step": args.steps,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved model checkpoint to {checkpoint_path}")
    plot_path = plot_losses(log_lines, checkpoint_path, args.p_metric)
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
