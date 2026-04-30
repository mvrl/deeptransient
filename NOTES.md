# DeepTransient — experiment notes

Snapshot of what's been run on this branch (`copilot/revamp-repo-with-modern-stack`)
and what's worth picking up next. The README has the formatted result tables; this
file is the working scratchpad.

## What's been run

### TransientNet (single 81/20 holdout, mean MSE × 100, ↓)

| Run dir                                       | Mean MSE × 100 | Notes                          |
|-----------------------------------------------|---------------:|--------------------------------|
| `transientnet_alexnet_imagenet`               | 4.38           | Paper-matching config (SGD)    |
| `transientnet_alexnet_places365`              | 4.36           | Paper-matching config (SGD)    |
| `transientnet_resnet18_imagenet_adam`         | 3.66           | Adam recovers from sigmoid+MSE |
| `transientnet_resnet50_imagenet_adam`         | 3.55           |                                |
| `transientnet_resnet50_places365_adam`        | 3.61           |                                |
| `transientnet_efficientnet_b0_imagenet_adam`  | 3.63           |                                |
| `transientnet_vit_b16_imagenet_adam`          | 3.54           |                                |
| `transientnet_clip_b32_frozen_adam`           | 3.85           | Frozen, linear probe           |
| `transientnet_clip_b32_finetune`              | **3.47**       | **Best**, FT from frozen ckpt  |
| `transientnet_clip_l14_frozen_adam`           | 3.78           | Frozen only                    |
| zero-shot CLIP ViT-B/32                       | 7.24           | No training                    |
| zero-shot CLIP ViT-L/14                       | 9.48           | No training                    |

Paper best (TransientNet-H, AlexNet+Hybrid): **3.83**.

### CloudyNet (5-fold normalised accuracy %, ↑)

| Run prefix                                | Norm. acc.        |
|-------------------------------------------|------------------:|
| `cloudynet_alexnet_imagenet_fold{0..4}`   | 85.9 ± 0.8        |
| `cloudynet_alexnet_places365_fold{0..4}`  | 86.6 ± 0.9        |
| `cloudynet_resnet50_imagenet_adam_fold*`  | **90.9 ± 1.1**    |
| `cloudynet_resnet50_places365_adam_fold*` | 89.8 ± 1.3        |
| `cloudynet_clip_b32_frozen_adam_fold*`    | 84.0 ± 0.5        |
| zero-shot CLIP ViT-B/32                   | 56.4 ± 0.9        |
| zero-shot CLIP ViT-L/14                   | 42.3 ± 1.2        |

Paper best (CloudyNet-H): **87.1 ± 0.3**. ResNet-50 with Adam clears it on both inits.

## Things that surprised me

1. **AlexNet underperforms the paper by ~0.3 on transient attrs** even with matching
   hyperparameters. Best hypothesis: torchvision AlexNet drops the original grouped
   convs in conv2/4/5, and the paper used Caffe-style BGR+per-channel-mean
   preprocessing. Closing that gap would probably mean shimming a Caffe-faithful
   AlexNet — not worth it; the trend Hybrid > Places > ImageNet still holds.
2. **Plain SGD stalls badly with sigmoid+MSE on modern backbones.** This isn't a
   subtle effect — ResNet-50 with SGD lands well behind Adam. Most of the time is
   spent in saturated tails. Adam is now the recommended path for non-AlexNet
   backbones.
3. **CLIP frozen linear-probe matches the paper on transient attrs in <1 min**, but
   *underperforms* trained ResNet-50 on CloudyNet. Sunny-vs-cloudy at this
   dataset's level of detail is a domain-specific decision boundary that CLIP's
   pre-trained features alone don't quite carve.
4. **Zero-shot CLIP ViT-L/14 is worse than ViT-B/32 on CloudyNet** (42 vs 56). Bigger
   model, sharper but more brittle alignment between the *words* "sunny"/"cloudy"
   and Lu et al.'s images.
5. **CLIP full fine-tune from the frozen-probe checkpoint beats every other
   configuration on transient attrs (3.47).** Two-stage training (linear probe →
   full FT) is meaningfully better than either alone in our runs.

## Suggested next steps

In rough priority order:

1. **CLIP ViT-L/14 full fine-tune.** Frozen L/14 is already at 3.78. Full FT from
   that checkpoint following the same recipe as B/32 might push below 3.47. Needs
   AMP + smaller batch size on the 4090s.
2. **CLIP fine-tune for CloudyNet.** Only frozen-probe (84.0) was run; given how
   well CLIP FT does for transient attrs, a 5-fold FT sweep is the obvious gap.
3. **Push best checkpoints to HuggingFace Hub.** `push_to_hub.py` already exists;
   the natural set is the four that the README highlights:
   `transientnet_clip_b32_finetune`, `transientnet_resnet50_imagenet_adam`,
   `transientnet_alexnet_imagenet`, `cloudynet_resnet50_imagenet_adam_fold0`
   (or all 5 folds packaged as one repo).
4. **Per-attribute breakdown.** `evaluate.py --per-attribute` is wired up. A short
   note in the README on which attributes CLIP-FT wins vs loses on relative to
   ResNet-50 would close the loop on the comparison.
5. **TTA / multi-crop at evaluation.** The paper disclaims none of this; we
   disclaim none of this. A 5-crop TTA likely shaves another 0.05–0.1 off
   transient MSE essentially for free.
6. **More CLIP variants.** SigLIP, OpenCLIP's ConvNeXt-Large, DFN-2B — all easy
   adds via open_clip. Likely diminishing returns at this point on this dataset,
   but cheap to confirm.
7. **Tighten the AlexNet repro story.** Either (a) get explicit about the
   architectural delta in the README and stop trying to match the paper number,
   or (b) shim a grouped-conv AlexNet + Caffe-style preprocessing. (a) is what
   the README currently does and is probably the right call.

## Repo state at handoff

- All training/eval/predict scripts work end-to-end.
- `scripts/run_all_transientnet.sh` and `scripts/run_all_cloudynet.sh` cover the
  full sweep, partitioned across two GPUs (queues `a` and `b`).
- `scripts/collect_results.py --write-readme` regenerates the README tables from
  `runs/` and the zero-shot JSON sidecars. Re-run after any new training.
- `runs/` is gitignored — checkpoints stay local.
- Branch is 5 commits ahead of `origin/copilot/revamp-repo-with-modern-stack` at
  this writing.
