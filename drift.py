""""
export INCEPTION_WEIGHTS=/data/ali/weights/weights-inception-2015-12-05-6726825d.pth

    CUDA_VISIBLE_DEVICES=7 python drift.py \
    --data-root "/data/ali/imf_latents/val" \
    --drift-steps 10 \
    --train-batch "/data/ali/imf_latents/train_overfit30_5classes.pt" \
    --checkpoint-path "/data/ali/imf_runs/overfit_dde_x_pred_lpips_ploss_muon_20000steps_30samples5classes.pt" \
    --num-y 1000 \
    --num-y-pos 50 \

"""


import argparse
import json
import os

import torch
import utils.torch_util as tu
from imf import iMeanFlow
from utils.vae_util import VAEWrapper
from utils.drift_util import compute_sharpener_drift, create_positive_bank
from utils.plot import plot_class_comparison
from utils.fid import (
    get_inception_model,
    extract_inception_features,
    compute_fid_from_features,
)


def decode_latents(vae, latents, batch_size=8):
    """Decode latents w/ the VAE in small batches for better memory usage"""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(latents), batch_size):
            batch = latents[start : start + batch_size]
            chunks.append(vae.decode(batch).float().cpu())
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drift-steps", type=int, default=10)
    p.add_argument("--data-root", type=str, default="/data/ali/imf_latents/val", help="Cached val latent shards")
    p.add_argument("--train-batch", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, required=True)
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
    p.add_argument("--cfg-omega", type=float, default=2.0, help="MeanFlow was trained with omega in [1, 8]")
    p.add_argument("--t-min", type=float, default=0.4, help="CFG interval start")
    p.add_argument("--t-max", type=float, default=0.65, help="CFG interval end")
    p.add_argument("--gen-steps", type=int, default=1, help="MeanFlow sampling steps")
    p.add_argument("--fid", action="store_true", help="FID from ADM ImageNet-256 train stats")
    p.add_argument(
        "--fid-statistics-file",
        type=str,
        default="/home/ali/unite-meanflow/drifting-model/MeanFlow/fid_stats/adm_in256_stats.npz",
    )
    args = p.parse_args()

    # load meanflow model alongside VAE decoder & feature extractor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mf_chkpt = torch.load(args.checkpoint_path)
    mf = iMeanFlow(model_str="imfDiT_B_2", parametrization="xpred", derivative="dde")
    mf.load_state_dict(mf_chkpt['model'], strict=False)
    mf.to(device)
    mf.eval()

    vae = VAEWrapper(decode_batch_size=16)

    # load data
    ds = torch.load(args.train_batch, map_location="cpu")
    print(ds.keys())
    x_batch = ds["x_batch"] # images
    y_batch = ds["y_batch"] # labels
    print(x_batch.shape)
    print(y_batch.shape)


    # config
    k = args.num_y  # gens per class
    k_pos = args.num_y_pos # reals per class
    steps = int(args.drift_steps)
    temperatures = torch.linspace(args.tau_start, args.tau_end, steps, device=device)
    step_size = args.step_size
    lambda_rep = args.lambda_rep
    sigma_r = args.sigma_r
    unique_labels = y_batch.unique(sorted=True)  # [10]
    cfg_omega = args.cfg_omega
    interval_min = args.t_min
    interval_max = args.t_max
    n_samples = len(unique_labels) * k 

    # create seeds and labels
    seeds = torch.arange(k).repeat(len(unique_labels)) # [0,1,2,3,4, 0,1,2,3,4, ...]
    gen_labels = unique_labels.repeat_interleave(k) # [c0,c0,c0,c0,c0, c1,c1,..., c10,...]

    y_pos_for_drift = create_positive_bank(
        args.data_root, unique_labels.tolist(), max_per_class=k_pos
    )
    print(y_pos_for_drift.keys())
    print(next(iter(y_pos_for_drift.values())).shape)

    # generate images in chunks for better memory usage
    gen_batch_size = args.gen_batch_size
    generated_chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, gen_batch_size):
            end = min(start + gen_batch_size, n_samples)
            chunk = mf.generate(
                n_sample=end - start,
                rng=tu.BatchGenerator(device=device, seeds=seeds[start:end]),
                num_steps=args.gen_steps,
                omega=cfg_omega,
                t_min=interval_min,
                t_max=interval_max,
                labels=gen_labels[start:end].to(device),
            ).cpu()
            generated_chunks.append(chunk)
            torch.cuda.empty_cache()
        generated_latents = torch.cat(generated_chunks, dim=0).to(device)
        print("generated latents shape: ", generated_latents.shape)


    # inference-time GMD sharpening in MeanFlow's latent space
    sharpened_latents = generated_latents.clone()
    # retrieval control: each generation replaced by its nearest bank latent
    copy_latents = generated_latents.clone()
    diagnostics = {}
    print(f"PERFORMING {steps} DRIFT STEPS")
    print(f"TEMPERATURES: {temperatures}")
    with torch.no_grad():
        # select the current class's generations
        for class_id in unique_labels:
            key = f"class_{class_id:04d}"
            mask = (gen_labels == class_id)

            y = generated_latents[mask].flatten(1) # [k, D] generations w/ current label
            y_pos = y_pos_for_drift[key].to(device).flatten(1) # [k_pos, D] val bank w/ current label
            y_gen = y.clone()
            nearest_gen = torch.cdist(y_gen, y_pos).min(dim=1)
            copy_latents[mask] = y_pos[nearest_gen.indices].reshape(generated_latents[mask].shape)

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
                    f"Sharpener step {step:>2d}/{steps}: "
                    f"tau={tau.item():.3f} "
                    f"mean_drift_norm={drift.norm(dim=1).mean().item():.4f}"
                )

            sharpened_latents[mask] = y.reshape(generated_latents[mask].shape)

            # retrieval diagnostics: if gen->bank distance is similar to bank->bank spacing,
            # the nearest bank image is a different photo, not a cleaner version of the gen
            bank_dist = torch.cdist(y_pos, y_pos)
            bank_dist.fill_diagonal_(float("inf"))
            nearest_sharp = torch.cdist(y, y_pos).min(dim=1)
            # memorization: a generalizing model's gens sit about as far from the training set
            # as held-out reals do; a memorizing model's gens sit much closer
            train_c = x_batch[y_batch == class_id].to(device).float().flatten(1)
            diagnostics[key] = {
                "gen_to_nearest_train": torch.cdist(y_gen, train_c).min(dim=1).values.mean().item(),
                "bank_to_nearest_train": torch.cdist(y_pos, train_c).min(dim=1).values.mean().item(),
                "gen_to_nearest_bank": nearest_gen.values.mean().item(),
                "bank_to_nearest_bank": bank_dist.min(dim=1).values.mean().item(),
                "sharp_to_nearest_bank": nearest_sharp.values.mean().item(),
                "distinct_bank_hits_gen": nearest_gen.indices.unique().numel(),
                "distinct_bank_hits_sharp": nearest_sharp.indices.unique().numel(),
                "relative_displacement": ((y - y_gen).norm(dim=1) / y_gen.norm(dim=1)).mean().item(),
            }
            print(key, {name: round(val, 3) for name, val in diagnostics[key].items()})

    # Free backbone memory
    del mf
    torch.cuda.empty_cache()

    generated_images = decode_latents(
        vae, generated_latents, batch_size=args.decode_batch_size
    )
    sharpened_images = decode_latents(
        vae, sharpened_latents, batch_size=args.decode_batch_size
    )
    del generated_latents, sharpened_latents
    torch.cuda.empty_cache()

    run_dir = os.path.join(
        args.out_dir,
        f"gmd_gens_{len(x_batch)}samples_{y_batch.unique().numel()}classes_{k}gens_{steps}steps_tau{args.tau_start:g}-{args.tau_end:g}"
        f"_omega{cfg_omega:g}_int{interval_min:g}-{interval_max:g}_gen{args.gen_steps}",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing class plots to {run_dir}")

    plot_n = min(args.plot_max, k)
    for class_id in unique_labels.tolist():
        key = f"class_{class_id:04d}"
        mask = gen_labels == class_id

        positive_images = decode_latents(
            vae,
            y_pos_for_drift[key][:plot_n].to(device),
            batch_size=args.decode_batch_size,
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
        copy_images = decode_latents(vae, copy_latents, batch_size=args.decode_batch_size)
        feat_copy, _ = extract_inception_features(
            ((copy_images + 1) / 2).clamp(0, 1), inception_model, batch_size=16
        )
        stats = args.fid_statistics_file
        fid_gen = compute_fid_from_features(feat_gen, stats)
        fid_sharp = compute_fid_from_features(feat_sharp, stats)
        fid_copy = compute_fid_from_features(feat_copy, stats)
        print(f"FID GENERATED (ADM): {fid_gen:.4f}")
        print(f"FID SHARPENED (ADM): {fid_sharp:.4f}")
        print(f"FID COPY (nearest bank vs ADM): {fid_copy:.4f}")
        results["fid"] = {
            "generated": fid_gen,
            "sharpened": fid_sharp,
            "copy": fid_copy,
            "fid_statistics_file": stats,
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
