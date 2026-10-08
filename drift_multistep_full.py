"""
Drift sharpener experiment on larger Meanflow checkpoint (SiT-B/2 trained on full ImageNet)

export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth

    CUDA_VISIBLE_DEVICES=6 python -u drift_multistep_full.py \
    --data-root "/data/ali/imf_latents/val" \
    --class-ids 0,1,2,3,4,5,6,7,8,9 \
    --drift-steps 2 \
    --num-y 8 \
    --num-y-pos 50 \
    --mid-t 0.5 \
    --cfg-scale 1.0

    more gpu availability:
    CUDA_VISIBLE_DEVICES=6 nohup python -u drift_multistep_full.py \
    --class-ids 0,1,2,3,4,5,6,7,8,9 \
    --drift-steps 2 --num-y 8 --num-y-pos 50 --mid-t 0.5 \
    --gen-batch-size 256 --decode-batch-size 32 \
    > drift_multistep_full.log 2>&1
"""

import argparse
import json
import os
import sys

import torch
import utils.torch_util as tu
from diffusers.models import AutoencoderKL
from utils.drift_util import (
    ZHUYU_LATENT_SCALE,
    class_ids_from_val_root,
    compute_sharpener_drift,
    create_positive_bank,
)
from utils.plot import plot_class_rows
from utils.fid import (
    get_inception_model,
    extract_inception_features,
    compute_fid_from_features,
)

MEANFLOW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MeanFlow")
sys.path.insert(0, MEANFLOW_ROOT)
from sit import SiT_models  


DEFAULT_CKPT = "/data/ali/meanflow_zhuyu_ckpts/ImageNet256/sit_b_2_meanflow_ema.pt"


def decode_latents(vae, latents, device, batch_size=8):
    """ 
    https://github.com/zhuyu-cs/MeanFlow scales latents by 0.18125. 
    We match this here.
    """
    chunks = []
    with torch.no_grad():
        for start in range(0, len(latents), batch_size):
            batch = latents[start : start + batch_size].to(device)
            out = vae.decode(batch / ZHUYU_LATENT_SCALE).sample.float().cpu()
            chunks.append(out)
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def official_mf_step(model, z, t, r, labels, cfg_scale=1.0):
    """z_r = z_t - (t-r) u(z_t, r, t)"""
    bsz = z.shape[0]
    device = z.device
    t_b = torch.full((bsz,), t, device=device, dtype=z.dtype)
    r_b = torch.full((bsz,), r, device=device, dtype=z.dtype)
    if cfg_scale > 1.0:
        null_y = torch.full_like(labels, model.num_classes)
        u_cond, u_uncond = model(
            torch.cat([z, z], dim=0),
            torch.cat([r_b, r_b], dim=0),
            torch.cat([t_b, t_b], dim=0),
            y=torch.cat([labels, null_y], dim=0),
        ).chunk(2, dim=0)
        u = u_uncond + cfg_scale * (u_cond - u_uncond)
    else:
        u = model(z, r_b, t_b, y=labels)
    return z - (t - r) * u


def mf_step_chunks(model, z, t, r, labels, batch_size, device, cfg_scale):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            end = min(start + batch_size, len(z))
            chunk = official_mf_step(
                model,
                z[start:end].to(device),
                t,
                r,
                labels[start:end].to(device),
                cfg_scale=cfg_scale,
            ).cpu()
            chunks.append(chunk)
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-steps", type=int, default=10)
    p.add_argument(
        "--data-root",
        type=str,
        default="/data/ali/imf_latents/val",
        help="Cached val latent shards (shard_*.pt)",
    )
    p.add_argument(
        "--class-ids",
        type=str,
        default="",
        help="Comma-separated class ids. Default: every class_* folder under data-root",
    )
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT,
    )
    p.add_argument("--model", type=str, default="SiT-B/2", choices=list(SiT_models.keys()))
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-y", type=int, required=True, help="Gens per class")
    p.add_argument("--num-y-pos", type=int, required=True, help="Val images per class in the bank")
    p.add_argument("--out-dir", type=str, default="/data/ali/gmd_gens/")
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--plot-max", type=int, default=8)
    p.add_argument("--tau-start", type=float, default=5.0)
    p.add_argument("--tau-end", type=float, default=15.0)
    p.add_argument("--step-size", type=float, default=0.2)
    p.add_argument("--lambda-rep", type=float, default=0.1)
    p.add_argument("--sigma-r", type=float, default=1.5)
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Must stay 1.0 for the CFG-trained zhuyu checkpoint",
    )
    p.add_argument("--mid-t", type=float, default=0.5)
    p.add_argument("--fid", action="store_true")
    p.add_argument(
        "--fid-statistics-file",
        type=str,
        default=os.path.join(MEANFLOW_ROOT, "fid_stats/adm_in256_stats.npz"),
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[args.model](
        input_size=32,
        num_classes=args.num_classes,
        use_cfg=True,
        **block_kwargs,
    ).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint_path}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    for p_ in vae.parameters():
        p_.requires_grad = False

    if args.class_ids.strip():
        unique_labels = torch.tensor(
            [int(x) for x in args.class_ids.split(",") if x.strip()],
            dtype=torch.long,
        )
    else:
        unique_labels = torch.tensor(class_ids_from_val_root(args.data_root), dtype=torch.long)
    unique_labels = unique_labels.sort().values
    print(f"classes ({len(unique_labels)}): {unique_labels.tolist()}")

    k = args.num_y
    k_pos = args.num_y_pos
    steps = int(args.drift_steps)
    temperatures = torch.linspace(args.tau_start, args.tau_end, steps, device=device)
    mid_t = args.mid_t
    n_samples = len(unique_labels) * k
    gen_batch_size = args.gen_batch_size
    cfg_scale = args.cfg_scale

    seeds = torch.arange(k).repeat(len(unique_labels))
    gen_labels = unique_labels.repeat_interleave(k)

    y_pos_for_drift = create_positive_bank(
        args.data_root,
        unique_labels.tolist(),
        max_per_class=k_pos,
        latent_norm="zhuyu",
    )
    print(next(iter(y_pos_for_drift.values())).shape)

    print("PERFORMING FIRST GENERATION STEP")
    mid_chunks = []
    onestep_chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, gen_batch_size):
            end = min(start + gen_batch_size, n_samples)
            rng = tu.BatchGenerator(device=device, seeds=seeds[start:end])
            z = rng.randn((end - start, 4, 32, 32)).to(torch.float32)
            labels_chunk = gen_labels[start:end].to(device)
            mid_chunks.append(
                official_mf_step(model, z, 1.0, mid_t, labels_chunk, cfg_scale).cpu()
            )
            onestep_chunks.append(
                official_mf_step(model, z, 1.0, 0.0, labels_chunk, cfg_scale).cpu()
            )
            torch.cuda.empty_cache()
    mid_latents = torch.cat(mid_chunks, dim=0)
    onestep_latents = torch.cat(onestep_chunks, dim=0)
    print("mid latents shape: ", mid_latents.shape)
    print("1-step latents shape: ", onestep_latents.shape)

    generated_latents = mid_latents.clone()
    sharpened_latents = mid_latents.clone()

    print(f"PERFORMING {steps} DRIFT STEPS AT t={mid_t}")
    print(f"TEMPERATURES: {temperatures}")
    with torch.no_grad():
        for class_id in unique_labels.tolist():
            key = f"class_{class_id:04d}"
            mask = gen_labels == class_id
            y = sharpened_latents[mask].to(device).flatten(1)
            y_pos = y_pos_for_drift[key].to(device).flatten(1)
            y_pos_noised = (1 - mid_t) * y_pos + mid_t * torch.randn_like(y_pos)

            for step, tau in enumerate(temperatures, start=1):
                drift = compute_sharpener_drift(
                    y, y_pos_noised, tau, lambda_rep=args.lambda_rep, sigma_r=args.sigma_r
                )
                y = y + args.step_size * drift
                print(
                    f"{key} sharpener step {step:>2d}/{steps}: "
                    f"tau={tau.item():.3f} "
                    f"mean_drift_norm={drift.norm(dim=1).mean().item():.4f}"
                )
            sharpened_latents[mask] = y.reshape(sharpened_latents[mask].shape).cpu()

    print("PERFORMING SECOND GENERATION STEP")
    generated_latents = mf_step_chunks(
        model, generated_latents, mid_t, 0.0, gen_labels, gen_batch_size, device, cfg_scale
    )
    sharpened_latents = mf_step_chunks(
        model, sharpened_latents, mid_t, 0.0, gen_labels, gen_batch_size, device, cfg_scale
    )
    print("generated latents shape: ", generated_latents.shape)
    print("sharpened latents shape: ", sharpened_latents.shape)

    del model
    torch.cuda.empty_cache()

    print("Decoding")
    onestep_images = decode_latents(vae, onestep_latents, device, args.decode_batch_size)
    generated_images = decode_latents(vae, generated_latents, device, args.decode_batch_size)
    sharpened_images = decode_latents(vae, sharpened_latents, device, args.decode_batch_size)
    del onestep_latents, generated_latents, sharpened_latents
    torch.cuda.empty_cache()

    ckpt_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
    run_dir = os.path.join(
        args.out_dir,
        f"gmd_mid_zhuyu_{ckpt_name}_{len(unique_labels)}classes_{k}gens_{steps}steps"
        f"_tau{args.tau_start:g}-{args.tau_end:g}_cfg{cfg_scale:g}_midt{mid_t:g}",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing class plots to {run_dir}")

    plot_n = min(args.plot_max, k)
    for class_id in unique_labels.tolist():
        key = f"class_{class_id:04d}"
        mask = gen_labels == class_id
        mid_row = decode_latents(
            vae, mid_latents[mask][:plot_n], device, args.decode_batch_size
        )
        plot_path = plot_class_rows(
            [
                ("1-step generated", onestep_images[mask][:plot_n]),
                (f"After first step (t={mid_t:g})", mid_row),
                ("2-step sharpened", sharpened_images[mask][:plot_n]),
                ("2-step unsharpened", generated_images[mask][:plot_n]),
            ],
            os.path.join(run_dir, f"{key}.png"),
            class_id=class_id,
            subtitle=f"{steps} drift steps | t_mid={mid_t:g} | zhuyu {args.model}",
        )
        print(f"Saved {plot_path}")

    results = {"args": vars(args)}
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if args.fid:
        inception_model = get_inception_model(device)
        onestep = ((onestep_images + 1) / 2).clamp(0, 1)
        gen = ((generated_images + 1) / 2).clamp(0, 1)
        sharp = ((sharpened_images + 1) / 2).clamp(0, 1)
        feat_onestep, _ = extract_inception_features(onestep, inception_model, batch_size=16)
        feat_gen, _ = extract_inception_features(gen, inception_model, batch_size=16)
        feat_sharp, _ = extract_inception_features(sharp, inception_model, batch_size=16)
        stats = args.fid_statistics_file
        fid_onestep = compute_fid_from_features(feat_onestep, stats)
        fid_gen = compute_fid_from_features(feat_gen, stats)
        fid_sharp = compute_fid_from_features(feat_sharp, stats)
        print(f"FID 1-STEP GENERATED (ADM): {fid_onestep:.4f}")
        print(f"FID 2-STEP UNSHARPENED (ADM): {fid_gen:.4f}")
        print(f"FID 2-STEP SHARPENED (ADM): {fid_sharp:.4f}")
        results["fid"] = {
            "onestep": fid_onestep,
            "generated": fid_gen,
            "sharpened": fid_sharp,
            "fid_statistics_file": stats,
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
