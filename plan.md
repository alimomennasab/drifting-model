# MeanFlow + Drift Sharpener: Plan

## Goal

Test whether a GMD-style inference-time drift improves one-step MeanFlow
samples without retraining the backbone. Small-data study on 10–15 ImageNet
classes; no full-ImageNet training or drifting.

## Setup

- Data: 200 ImageNet classes, up to 1,100 unique images/class
(`/data/ali/imagenet/train`), encoded on one GPU in file order to
`/data/ali/imf_latents/train` (217,965 latents).
- Backbones (MeanFlow, muon + LPIPS, 20k steps), checked via their saved
`batch_file` and class-embedding norms:

  | checkpoint                  | trained on                              | classes               |
  | --------------------------- | --------------------------------------- | --------------------- |
  | `..._30samples10classes.pt` | `train_overfit30_10classes.pt`, 3/class | 0–9                   |
  | `..._30samples5classes.pt`  | `train_overfit30_5classes.pt`, 6/class  | 1, 3, 4, 5, 7         |
  | `..._30samples15classes.pt` | `train_overfit30_15classes.pt`, 2/class | 15 classes (10 … 193) |

- 10-class backbone: its training images are files 0, 2, 4 of each class
(pixel-matched), so `skip_first = 5`.
- Drift/FID split per class: 5 skipped, 50 attraction images, 1,045 FID reals.
1,000 generations/class, FID-10k.



## Findings so far



### 1. The earlier sweep used the wrong checkpoint

The checkpoint in the first sweeps (then named `..._30samples10classes.pt`) was
the 5-class model. Classes 0, 2, 6, 8, 9 were never trained, so their
generations were noise. That made the MeanFlow FID look much worse than it is
(175.5) and made the drift look like a big win (93.1 at 10 steps): the drift
replaced noise with near-copies of real bank images.

### 2. With the correct checkpoint, drift makes FID worse


| drift steps | FID generated | FID sharpened |
| ----------- | ------------- | ------------- |
| 2           | 89.6          | 128.4         |
| 10          | 89.6          | 98.2          |


(5, 15, 20 steps pending.)

The 10-class backbone memorizes its 3 images/class, so its samples are already
sharp near-copies of real photos. There is nothing to sharpen; the drift only
blends them toward other images.

### 3. The current drift is nearest-neighbour snapping

- Squared latent distance to the nearest attraction image: ~5,000–6,000; gap to
the second nearest: ~180–280.
- At tau = 0.3 → 0.08 the attraction kernel is one-hot (top weight 1.0). It only
spreads over several images around tau ≈ 20 (top weight ≈ 0.13–0.21).
- Each step is effectively `y ← 0.8·y + 0.2·nearest_bank_latent`; after 10
steps a sample is ~89% one bank image. Samples of a class collapse onto a
handful of bank images.
- Repulsion is a no-op: with `sigma_r = 1.5`, `exp(-d²/4.5) ≈ 0`. Its sign is
also reversed (`v = attraction - y - λ·(y - ȳ_nbr)` pulls samples together).



## Next experiments



### A. Harder backbones (more images per class)

The drift can only help if the backbone's samples are novel but imperfect,
which requires a backbone that cannot memorize its training set.

- Train on classes 0–9 with 10, 50, 200 images/class via
`create_train_batch.py` on the current shards (~4 h per 20k-step run; can run
in parallel on separate GPUs).
- Use `skip_first` = images/class for each (files are in shard order).
- For each: baseline FID, visual check for memorization (nearest-training
distance), then the drift sweep.
- Expect baseline FID to get worse (less memorization) and the drift to have
room to help.



### B. Softer kernels (raise temperature)

Raise tau so each sample is pulled toward a weighted mix of bank images rather
than snapping onto one.

- Sweep tau ∈ {5, 10, 20, 40}; log mean top-1 kernel weight and entropy.
Target top-1 weight ~0.1–0.5.
- Fix the repulsion sign and set `sigma_r` near the median gen–gen distance
(~50–80); sweep `lambda_rep` ∈ {0, 0.1, 0.5, 1}.
- Tune step size × steps ({0.05, 0.1, 0.2} × {1, 3, 5, 10}) on a small dev set
(e.g. 100 gens/class, different seeds), then freeze.
- Also try `computeV()` (row/column-normalized, GMD-faithful).



### C. Retrieval controls (run alongside A and B)

- Copy baseline: replace each generation with its nearest attraction image.
- Random-perturbation control with the drift's per-sample norm.
- Diagnostics: distinct bank images hit per class, distance to nearest bank
image before vs after, relative displacement `||y_sharp - y_gen|| / ||y_gen||`.

A result only counts as sharpening if it beats the copy baseline and keeps
diversity.

### D. Later

- Kernel in a perceptual feature space (DINOv2 / CLIP), update in latent space.
- Bank size 5 → 250/class with fixed FID reals.
- Compute-matched baselines: 2–4-step MeanFlow sampling, CFG scale sweep
(current omega = 48 is very high).
- Retrain the 15-class backbone on a batch from the current shards.



## Metrics

FID alone mixes quality and diversity. Report:

- FID-10k and KID vs held-out reals;
- precision / recall or density / coverage;
- within-class diversity (pairwise LPIPS or DINOv2);
- copying: nearest-bank and nearest-training distances;
- requested-class accuracy from a pretrained classifier;
- per-class numbers.



## Code changes

1. Command-line args for tau schedule, step size, `sigma_r`, `lambda_rep`;
  include them in the output folder name.
2. Fix the repulsion sign in `compute_sharpener_drift`.
3. Log kernel top-1 weight / entropy and displacement per step.
4. Write `results.json` per run (args, split counts, FIDs).
5. Check the checkpoint's `args['batch_file']` against `--train-batch` and stop
  if they differ.
6. Remove the `['class_0000']` debug prints.



## Order

1. Start training runs for A (they take hours).
2. Meanwhile: code changes, then B and C on the existing 10-class backbone.
3. Run the best B settings plus C controls on each new backbone from A.

