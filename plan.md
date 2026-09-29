# MeanFlow + Drift Sharpener: Plan

## Goal

Test whether a GMD-style drift can sharpen one-step MeanFlow samples on
ImageNet latents the way it does on the 2D toy (25-Gaussian grid), without
reducing to retrieval of real images. Small-data study on 5–10 classes.

## Setup

- Data: 200 ImageNet classes, up to 1,100 unique images/class
(`/data/ali/imagenet/train`), encoded to SD-VAE latents (4×32×32) in
`/data/ali/imf_latents/train`. All drift math happens in this latent space
(flattened to 4,096-d); pixels are only used for plots and FID.
- Backbones (iMF DiT-B/2, x-pred, dde, muon + LPIPS, 20k steps):

  | checkpoint                  | trained on                              | train files / class | `skip_first` |
  | --------------------------- | --------------------------------------- | ------------------- | ------------ |
  | `..._30samples10classes.pt` | `train_overfit30_10classes.pt`, 3/class | 0, 2, 4             | 5            |
  | `..._30samples5classes.pt`  | `train_overfit30_5classes.pt`, 6/class  | 0, 2, 4, 6, 8, 10   | 11           |
  | `..._30samples15classes.pt` | `train_overfit30_15classes.pt`, 2/class | (not checked)       | —            |

  Train files were identified by pixel-matching decoded batch latents against
  the raw images. They are every other file because those batches were built
  from older shards encoded on 2 GPUs.
- Per class: `skip_first` skipped, 50 bank (attraction) images, the rest are
FID reals (1,039/class for the 5-class backbone → FID-5k; 1,045/class for the
10-class backbone → FID-10k). FIDs are only comparable within a backbone.
- `drift.py` flags: `--tau-start/--tau-end`, `--step-size`, `--lambda-rep`,
`--sigma-r`, `--cfg-omega`, `--t-min/--t-max`, `--gen-steps`,
`--skip-first`, batch sizes. Each run prints per-class retrieval diagnostics
and a copy-baseline FID and writes `results.json`.



## Findings



### 1. Early sweeps used the wrong checkpoint

The first sweeps loaded the 5-class model under the 10-class name. Half the
classes were never trained, their samples were noise, and the drift "improved"
FID (175.5 → 93.1) by replacing noise with near-copies of bank images.

### 2. The drift is nearest-neighbour retrieval

- In 4,096-d latent space the attraction kernel is one-hot up to τ ≈ 10: each
step is `y ← 0.8·y + 0.2·nearest_bank`, so drift norms shrink by exactly ×0.8
per step and after 10 steps a sample is ~89% one bank image.
- A generation's nearest bank image is as far away as two different real
photos are from each other (≈70–75 vs ≈72): the "nearest real image" is a
different photo, not a cleaner version of the generation.
- After drifting, samples sit ≈8.5 from a bank image, and only ~10 of the 50
bank images per class are ever used.
- Repulsion is a no-op (`sigma_r = 1.5` → weights ≈ 0) and its sign is reversed.



### 3. Temperature: retrieval or blur, no sharpening in between

5-class backbone, 10 drift steps, FID-5k (generated = 122.9 at ω = 48):


| τ schedule        | FID sharpened    | behaviour                               |
| ----------------- | ---------------- | --------------------------------------- |
| 0.3 → 0.08        | 92.9             | snap to nearest                         |
| 5, 10 (constant)  | 92.7, 91.9       | snap to nearest                         |
| 15 (constant)     | 104.1            | blend of several bank images, blur      |
| 20 → 320          | 259 – 292        | collapse toward class mean              |
| 5→10, 10→15, 5→20 | 92.7, 91.0, 92.5 | increasing ramps lock in the snap early |


Higher τ doesn't select more varied bank images. It averages several of them
into one target, which decodes to a blurry overlay. Fewer drift steps give a
partial blend of two images (2 steps: FID 181), which is worse than no drift.

### 4. On a memorizing backbone, drift hurts

10-class backbone (3 images/class): generated FID 89.6; drifted 128.4 (2 steps),
159.6 (5), 98.2 (10). Its samples are already near-copies of its training
images, so there is nothing to sharpen.

### 5. CFG: ω = 48 was out of range; ω = 1–2 and one step are best

The model is trained with ω ∈ [1, 8] (`sample_cfg_scale`, `s_max = 7`).
5-class backbone, τ = 10, interval 0.4–0.65, FID-5k:


| ω   | steps | generated | sharpened | copy  |
| --- | ----- | --------- | --------- | ----- |
| 1   | 1     | 112.1     | 96.5      | 109.3 |
| 2   | 1     | **111.5** | 96.3      | 110.8 |
| 4   | 1     | 114.5     | 95.0      | 110.3 |
| 8   | 1     | 130.6     | 90.6      | 105.9 |
| 48  | 1     | 122.9     | 91.9      | 106.8 |
| 1   | 2     | 138.5     | 109.5     | 115.4 |
| 2   | 2     | 135.2     | 107.9     | 114.5 |
| 4   | 2     | 138.0     | 107.7     | 114.3 |
| 8   | 2     | 140.3     | 108.5     | 112.5 |
| 48  | 2     | 139.4     | 105.1     | 111.6 |


- ω = 2, 1 step improves raw MeanFlow by ~11 FID over ω = 48 at no cost. Use it
as the baseline. (`--cfg-omega` still defaults to 48.)
- 2-step sampling is worse at every ω and hits fewer distinct bank images.
- ω = 8 generations sit far from all real images (nearest-bank ≈ 101 vs ≈ 72).



### 6. Drift beats exact copying only through jitter

Drift beats the copy baseline by 13–15 FID, but drifted samples are ~89% a bank
image. The copy baseline repeats ~10 images/class ~100× each, which FID
penalizes. The drift's 11% residual of the generation makes each copy slightly
different. That is added diversity, not added quality.

### 7. Why the toy worked

- A toy mode is a dense cloud (~200 real points, std 0.05). With τ ≈ 0.08 the
kernel averages many points of the same mode, so the target is the mode
centre. ImageNet images have no near neighbours, so the local mean is one
other photo or a blur.
- The toy's sharpness metric (distance to mode centre) rewards within-mode
collapse. FID does not.
- The toy's MeanFlow generalized and blurred between modes. Ours memorizes its
training images, so there is no blur to fix, only images to swap.

**Criterion for "not retrieval":** sharpened samples stay about as far from
the nearest bank image as raw generations (not ≈8), FID beats both the raw
generations and the copy baseline, and diversity is unchanged.

## Next experiments



### A. Retrain a backbone that generalizes (prerequisite)

All sharpening ideas need samples that are novel but imperfect.

- Classes 0–9, 100–300 images/class (e.g. 200 → 900 left for bank + FID).
Build the batch from the current single-GPU shards; confirm the training
files by pixel-matching (expected `skip_first` = images/class).
- Train longer than 20k steps; save checkpoints every ~10k.
- Memorization check per checkpoint: nearest-training distance vs  
training-to-training spacing.



### B. Controls (quick, current backbone)

- Copy + random noise with the same norm as the drift's residual. If FID ≈ the
drifted FID, the drift adds nothing but jitter.
- Kernel-density resampling: generate extra samples, keep the ones closest to
the bank (feature space). A no-retrieval baseline that any sharpener must beat.



### C. Inference-time sharpening that can't paste bank images

1. **Drift at a noisy intermediate step.** Re-noise the MeanFlow output to

t ≈ 0.3–0.5, drift the noisy state toward the bank noised to the same t
(bandwidth scaled with t), then take one MeanFlow step to t = 0. Blends are
valid inputs at high noise, and the model regenerates the detail. Baseline:
the same re-noise + step without drift.
2. **Drift the input noise.** Optimize ε so that `f(ε)` moves toward the bank
(gradients through the one-step generator). Outputs are always model samples.
3. **Feature-space gradient drift.** Kernel on DINOv2 features of decoded
samples, gradient steps on the latent, bounded step size. Evaluate with
Inception (a different network) to avoid gaming the metric.

### D. Training-time drift (the real GMD, main experiment)

On the backbone from A: `L = L_MF + λ·‖f(z) − sg[f(z) + V(f(z))]‖²`, with V
computed per minibatch in a feature space, small λ, and repulsion fixed (sign,
`sigma_r` ≈ median gen–gen distance). Averaged over batches, the model can't
copy individual images. The toy's GMD-only learned variant collapsed, so start
from a trained MeanFlow and increase λ slowly.

## Metrics

- FID and KID vs held-out reals, plus the copy-baseline FID.
- Retrieval diagnostics: gen→bank and sharpened→bank vs bank→bank distance;
distinct bank images hit per class; relative displacement.
- Precision / recall or density / coverage; within-class diversity (pairwise
LPIPS or DINOv2).
- Nearest-training distance (memorization).
- For the toy: add a within-mode spread metric next to sharpness.



## Code to-dos

1. Set the `--cfg-omega` default to 2.
2. Fix the repulsion sign in `compute_sharpener_drift`; set `sigma_r` from data.
3. Add the copy + noise control and the resampling baseline to `drift.py`.
4. Check the checkpoint's saved `batch_file` against `--train-batch` and stop

if they differ.
5. Remove leftover debug prints; keep `results.json` per run.

## Order

1. Start A as soon as a GPU is free (hours).
2. Meanwhile: B on the current backbone, and implement C1 and C2.
3. On the new backbone: baseline (ω = 2, 1 step), B, C1–C3, then D.

