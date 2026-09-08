# MeanFlow + Drift Sharpener Experimental Plan

## Goal

Evaluate whether a GMD-inspired inference-time drift can improve blurry
one-step MeanFlow outputs without retraining the MeanFlow backbone.

The current study is a small-data proof of concept, not a full ImageNet
benchmark.

## Available resources

- One MeanFlow trained on 30 images from 10 classes (3 images/class).
- One MeanFlow trained on 30 images from 15 classes (2 images/class).
- Both MeanFlows use LPIPS loss and remain fixed throughout the study.
- Approximately 2,000 real images from 200 classes are available for building
  reference banks (about 10 images/class).
- Training is limited to one GPU and approximately 20,000 steps, but inference
  can be performed repeatedly in small batches.

## Current findings

1. A MeanFlow overfit on 30 images may already behave in a mode-seeking or
   memorizing manner, leaving little visible blur for drift to correct.
2. LPIPS training already encourages perceptual sharpness, which further
   reduces the expected improvement from post-processing.
3. `drift.py` currently uses the MeanFlow training batch as the positive
   reference bank. Training, reference, and evaluation data should be
   disjoint.
4. `compute_attraction()` currently compares each generated sample with every
   supplied real sample. For class-conditional generation, positive references
   should come from the same class:

   `y_pos ~ p_data(. | class_label)`

5. Drift should be computed independently for each class:
   - attraction from same-class real references;
   - repulsion among same-class generated samples.
6. `drift.py` currently gives every class the same noise seed. Evaluation
   should use multiple unique seeds per class.
7. The current repulsion sign needs correction. `compute_repulsion()` returns
   `y_i - weighted_neighbor`, which already points away from neighboring
   generated samples. Subtracting that vector in `compute_sharpener_drift()`
   moves samples toward their neighbors.
8. The current sharpener is GMD-inspired rather than an exact implementation:
   - it operates on flattened raw VAE latents;
   - it uses a squared Gaussian-distance kernel;
   - it does not use the paper's row-and-column normalization;
   - it weights attraction and repulsion asymmetrically.
9. The GMD paper reports that a suitable feature encoder is important because
   raw high-dimensional distances can produce ineffective kernels.

## Data protocol

Only classes learned by each MeanFlow should be used in its primary
class-conditional evaluation. Other classes can be used as a negative control.

Create disjoint splits by underlying image identity:

### 10-class MeanFlow

- Training: 3 images/class (already used).
- Reference bank: 5 different images/class.
- Held-out evaluation: 2 different images/class.

### 15-class MeanFlow

- Training: 2 images/class (already used).
- Reference bank: 6 different images/class.
- Held-out evaluation: 2 different images/class.

If the available bank includes images used during MeanFlow training, remove
them before constructing these splits.

Save reference and evaluation files with both latents and labels, for example:

```python
{
    "x_batch": latents,
    "y_batch": labels,
}
```

## Generated evaluation set

For each fixed MeanFlow:

1. Generate 20 unique seeds per class.
2. Save the unmodified generated latents.
3. Apply every drift configuration to the exact same generated latents.

This produces:

- 200 generated samples for the 10-class model;
- 300 generated samples for the 15-class model.

Using fixed generated latents makes every comparison paired and avoids
regenerating MeanFlow outputs for each experiment.

## Required implementation changes

1. Add a separate reference-bank argument to `drift.py`.
2. Load reference latents and labels independently of the generated labels.
3. Group generated and reference latents by class.
4. Compute attraction only against same-class references.
5. Compute repulsion only among generated samples of the same class.
6. Correct or redefine the repulsion sign.
7. Support unique configurable generation seeds.
8. Save original and sharpened latents so metrics can be recomputed without
   rerunning generation.
9. Log drift diagnostics:
   - attraction norm;
   - repulsion norm;
   - total drift norm;
   - relative latent displacement;
   - nearest-reference distance;
   - kernel-weight entropy.

Kernel entropy is important: near-zero entropy indicates one-hot nearest
neighbor copying, while maximum entropy indicates an almost uniform,
uninformative kernel.

## Experiment 1: Correctness and component ablation

Use the largest disjoint same-class bank and compare:

1. No drift (fixed MeanFlow baseline).
2. Current all-class attraction (negative control).
3. Same-class attraction only.
4. Same-class repulsion only.
5. Same-class attraction plus corrected repulsion.
6. Random latent perturbation matched to the full drift's average norm.

This establishes whether improvements come from the intended drift structure
rather than from arbitrary latent perturbation.

## Experiment 2: Reference-bank size

With the best drift formulation from Experiment 1, compare:

- 1 reference/class;
- 2 references/class;
- 4 references/class;
- maximum available (5 or 6 references/class).

Use nested subsets or repeat each size with several deterministic subset seeds.
This tests whether a better empirical estimate of the real class distribution
improves the refinement.

## Experiment 3: Drift strength

Tune parameters using a small development set of 5 generated seeds/class:

- steps: 1, 3, 5, 10;
- step size: 0.02, 0.05, 0.1, 0.2;
- repulsion weight: 0, 0.1, 0.5, 1.0.

Do not run the full Cartesian product initially. First calibrate temperature
from observed same-class distances, then tune step size and number of steps.

Freeze the selected settings before evaluating the remaining generated seeds.

## Temperature calibration

The existing temperature schedule (`0.3` to `0.08`) should not be assumed to
work in a 4,096-dimensional raw latent space.

For each class:

1. Measure generated-to-reference distances.
2. Compute typical nearest-neighbor and median distances.
3. Select temperatures that produce non-uniform but non-one-hot weights.
4. Confirm this using kernel-weight entropy.

Feature normalization should be applied before interpreting absolute
temperature values.

## Experiment 4: Feature-space comparison

After validating the raw-latent baseline, compare:

1. Raw normalized VAE-latent distance.
2. A pretrained perceptual feature space such as DINOv2, CLIP, SimCLR, or
   MoCo.
3. The existing `computeV()` row-and-column normalized formulation.

A practical hybrid is to use perceptual features to select and weight real
neighbors, then apply the weighted update to their corresponding VAE latents.
This avoids training a new feature encoder, although it remains a custom
inference-time heuristic.

## Metrics

Compute paired before/after changes for every generated seed.

### Image quality

- CLIP-IQA or MUSIQ as the primary no-reference quality metric.
- Laplacian variance or high-frequency energy as a diagnostic only, because
  these metrics can reward noise and artifacts.

### Conditional correctness

- Requested-class top-1 accuracy.
- Requested-class classifier confidence.

### Realism

- DINOv2 or Inception feature distance to held-out same-class real images.
- Exploratory KID only if enough generated and real samples are available.
- Do not treat FID from approximately 30 real samples as reliable.

### Diversity and collapse

- Pairwise DINOv2 or LPIPS distance among generated samples within each class.
- Duplicate or near-duplicate rate.

### Content preservation and copying

- LPIPS between the original and sharpened output.
- Nearest-neighbor distance to the reference bank.
- Nearest-neighbor distance to the MeanFlow training images.
- Visualize each sharpened output beside its nearest reference to detect
  reference copying.

### Update magnitude

- Relative latent movement:

  `||y_sharp - y_base|| / ||y_base||`

## Statistical analysis

- Use the same base sample for every drift condition.
- Report paired metric differences rather than only absolute values.
- Report bootstrap 95% confidence intervals.
- Bootstrap by class or report per-class results so one easy class does not
  dominate the average.
- Select hyperparameters on development seeds and report final results on
  separate evaluation seeds.

## Comparing the two MeanFlows

The main cross-model hypothesis is:

> The 15-class model, with fewer training images per class, may benefit more
> from an external same-class reference bank than the 10-class model.

Compare relative before/after improvements within each model. If the models use
different classes, do not attribute absolute differences solely to the number
of classes because class difficulty is a confound. Prefer overlapping classes
when available.

## Success criteria

The sharpener is useful only if it:

1. improves perceptual quality or sharpness;
2. preserves or improves requested-class confidence;
3. does not substantially reduce within-class diversity;
4. does not merely copy a bank or training image;
5. outperforms a matched random perturbation;
6. behaves consistently across classes and seeds.

If it improves sharpness but reduces diversity or copies references, report
that as a trade-off rather than an unqualified improvement.

## Recommended first run

For each MeanFlow:

1. Generate 20 unique seeds/class and save the base latents.
2. Use the maximum disjoint same-class reference bank.
3. Compare no drift, attraction-only, corrected full drift, and matched random
   perturbation.
4. Start with 5 drift steps and a conservatively calibrated temperature and
   step size.
5. Measure quality, class confidence, diversity, copying, LPIPS change, and
   relative latent displacement.
6. Only proceed to bank-size and broader parameter sweeps if the corrected
   drift shows a measurable paired improvement.

