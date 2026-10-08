"""
Drift sharpener on the larger MeanFlow checkpoint (SiT-B/2 trained on full ImageNet)

export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth

    CUDA_VISIBLE_DEVICES=6 nohup python -u drift_full.py \
    --drift-steps 1 
    --num-y 50 
    --num-y-pos 50 \
    --gen-batch-size 128 
    --decode-batch-size 8 \
    --fid \
    > drift_full_1step.log 2>&1 &
"""

import argparse
import json
import os

import torch
import utils.torch_util as tu
from diffusers.models import AutoencoderKL
from drift_multistep_full import (
    DEFAULT_CKPT,
    MEANFLOW_ROOT,
    SiT_models,
    decode_latents,
    official_mf_step,
)
from utils.drift_util import (
    class_ids_from_val_root,
    compute_sharpener_drift,
    create_positive_bank,
)
from utils.plot import plot_class_comparison
from utils.fid import (
    get_inception_model,
    extract_inception_features,
    compute_fid_from_features,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-steps", type=int, default=10)
    p.add_argument("--data-root", type=str, default="/data/ali/imf_latents/val", help="Cached val latent shards")
    p.add_argument("--class-ids", type=str, default="", help="Comma-separated class ids. Default: every class in data-root")
    p.add_argument("--checkpoint-path", type=str, default=DEFAULT_CKPT)
    p.add_argument("--model", type=str, default="SiT-B/2", choices=list(SiT_models.keys()))
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-y", type=int, required=True, help="Amount of generations produced **PER CLASS**")
    p.add_argument("--num-y-pos", type=int, required=True, help="Val images per class in the drift bank")
    p.add_argument("--out-dir", type=str, default="/data/ali/gmd_gens/")
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument("--gen-batch-size", type=int, default=64)
    p.add_argument("--plot-max", type=int, default=8, help="Max images per row in class PNGs")
    p.add_argument("--tau-start", type=float, default=5.0)
    p.add_argument("--tau-end", type=float, default=15.0)
    p.add_argument("--step-size", type=float, default=0.2)
    p.add_argument("--lambda-rep", type=float, default=0.1)
    p.add_argument("--sigma-r", type=float, default=1.5)
    p.add_argument("--cfg-scale", type=float, default=1.0, help="Must stay 1.0 for the CFG-trained zhuyu checkpoint")
    p.add_argument("--gen-steps", type=int, default=1, help="MeanFlow sampling steps")
    p.add_argument("--fid", action="store_true", help="FID from ADM ImageNet-256 train stats")
    p.add_argument(
        "--fid-statistics-file",
        type=str,
        default=os.path.join(MEANFLOW_ROOT, "fid_stats/adm_in256_stats.npz"),
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiT_models[args.model](
        input_size=32,
        num_classes=args.num_classes,
        use_cfg=True,
        fused_attn=False,
        qk_norm=False,
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
        unique_labels = torch.tensor([int(x) for x in args.class_ids.split(",") if x.strip()])
    else:
        unique_labels = torch.tensor(class_ids_from_val_root(args.data_root))
    unique_labels = unique_labels.sort().values
    print(f"classes ({len(unique_labels)}): {unique_labels.tolist()}")

    # config
    k = args.num_y  # gens per class
    k_pos = args.num_y_pos # reals per class
    steps = int(args.drift_steps)
    temperatures = torch.linspace(args.tau_start, args.tau_end, steps, device=device)
    step_size = args.step_size
    lambda_rep = args.lambda_rep
    sigma_r = args.sigma_r
    cfg_scale = args.cfg_scale
    n_samples = len(unique_labels) * k

    # create seeds and labels
    seeds = torch.arange(k).repeat(len(unique_labels)) # [0,1,2,3,4, 0,1,2,3,4, ...]
    gen_labels = unique_labels.repeat_interleave(k) # [c0,c0,c0,c0,c0, c1,c1,..., c10,...]

    y_pos_for_drift = create_positive_bank(
        args.data_root, unique_labels.tolist(), max_per_class=k_pos, latent_norm="zhuyu"
    )
    print(next(iter(y_pos_for_drift.values())).shape)

    # generate images in chunks for better memory usage
    gen_batch_size = args.gen_batch_size
    time_steps = torch.linspace(1.0, 0.0, args.gen_steps + 1).tolist()
    generated_chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, gen_batch_size):
            end = min(start + gen_batch_size, n_samples)
            rng = tu.BatchGenerator(device=device, seeds=seeds[start:end])
            z = rng.randn((end - start, 4, 32, 32)).to(torch.float32)
            labels_chunk = gen_labels[start:end].to(device)
            for t, r in zip(time_steps[:-1], time_steps[1:]):
                z = official_mf_step(model, z, t, r, labels_chunk, cfg_scale)
            generated_chunks.append(z.cpu())
            torch.cuda.empty_cache()
        generated_latents = torch.cat(generated_chunks, dim=0).to(device)
        print("generated latents shape: ", generated_latents.shape)

    # inference-time GMD sharpening in MeanFlow's latent space
    sharpened_latents = generated_latents.clone()
    diagnostics = {}
    print(f"PERFORMING {steps} DRIFT STEPS")
    print(f"TEMPERATURES: {temperatures}")
    with torch.no_grad():
        for class_id in unique_labels.tolist():
            key = f"class_{class_id:04d}"
            mask = (gen_labels == class_id).to(device)

            y = generated_latents[mask].flatten(1) # [k, D] generations w/ current label
            y_pos = y_pos_for_drift[key].to(device).flatten(1) # [k_pos, D] val bank w/ current label
            y_gen = y.clone()
            nearest_gen = torch.cdist(y_gen, y_pos).min(dim=1)

            for step, tau in enumerate(temperatures, start=1):
                drift = compute_sharpener_drift(
                    y,
                    y_pos,
                    tau,
                    lambda_rep=lambda_rep,
                    sigma_r=sigma_r,
                )
                y = y + step_size * drift
                print(
                    f"{key} sharpener step {step:>2d}/{steps}: "
                    f"tau={tau.item():.3f} "
                    f"mean_drift_norm={drift.norm(dim=1).mean().item():.4f}"
                )

            sharpened_latents[mask] = y.reshape(generated_latents[mask].shape)

            # retrieval diagnostics: if gen->bank distance is similar to bank->bank spacing,
            # the nearest bank image is a different photo, not a cleaner version of the gen
            bank_dist = torch.cdist(y_pos, y_pos)
            bank_dist.fill_diagonal_(float("inf"))
            nearest_sharp = torch.cdist(y, y_pos).min(dim=1)
            diagnostics[key] = {
                "gen_to_nearest_bank": nearest_gen.values.mean().item(),
                "bank_to_nearest_bank": bank_dist.min(dim=1).values.mean().item(),
                "sharp_to_nearest_bank": nearest_sharp.values.mean().item(),
                "distinct_bank_hits_gen": nearest_gen.indices.unique().numel(),
                "distinct_bank_hits_sharp": nearest_sharp.indices.unique().numel(),
                "relative_displacement": ((y - y_gen).norm(dim=1) / y_gen.norm(dim=1)).mean().item(),
            }
            print(key, {name: round(val, 3) for name, val in diagnostics[key].items()})

    # Free backbone memory
    del model
    torch.cuda.empty_cache()

    generated_images = decode_latents(vae, generated_latents, device, args.decode_batch_size)
    sharpened_images = decode_latents(vae, sharpened_latents, device, args.decode_batch_size)
    del generated_latents, sharpened_latents
    torch.cuda.empty_cache()

    ckpt_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
    run_dir = os.path.join(
        args.out_dir,
        f"gmd_gens_zhuyu_{ckpt_name}_{len(unique_labels)}classes_{k}gens_{steps}steps"
        f"_tau{args.tau_start:g}-{args.tau_end:g}_ss{step_size:g}_cfg{cfg_scale:g}_gen{args.gen_steps}",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing class plots to {run_dir}")

    plot_n = min(args.plot_max, k)
    for class_id in unique_labels.tolist():
        key = f"class_{class_id:04d}"
        mask = gen_labels == class_id

        positive_images = decode_latents(
            vae, y_pos_for_drift[key][:plot_n], device, args.decode_batch_size
        )
        plot_path = plot_class_comparison(
            steps,
            k_pos,
            positive_images[:plot_n],
            generated_images[mask][:plot_n],
            sharpened_images[mask][:plot_n],
            os.path.join(run_dir, f"{key}.png"),
            class_id=class_id,
        )
        print(f"Saved {plot_path}")

    results = {"args": vars(args), "diagnostics": diagnostics}
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if args.fid:
        inception_model = get_inception_model(device)
        # normalize & clamp: fid expects [0, 1] values
        gen = ((generated_images + 1) / 2).clamp(0, 1)
        sharp = ((sharpened_images + 1) / 2).clamp(0, 1)
        feat_gen, _ = extract_inception_features(gen, inception_model, batch_size=16)
        feat_sharp, _ = extract_inception_features(sharp, inception_model, batch_size=16)
        del gen, sharp
        stats = args.fid_statistics_file
        fid_gen = compute_fid_from_features(feat_gen, stats)
        fid_sharp = compute_fid_from_features(feat_sharp, stats)
        print(f"FID GENERATED (ADM): {fid_gen:.4f}")
        print(f"FID SHARPENED (ADM): {fid_sharp:.4f}")
        results["fid"] = {
            "generated": fid_gen,
            "sharpened": fid_sharp,
            "fid_statistics_file": stats,
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
