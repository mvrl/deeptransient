# DeepTransient

**A fast method for estimating transient scene attributes from a single image
using deep convolutional neural networks.**

> Ryan Baltenberger, Menghua Zhai, Connor Greenwell, Scott Workman, Nathan Jacobs  
> *IEEE Winter Conference on Applications of Computer Vision (WACV), 2016*

This repository contains a clean PyTorch reimplementation of the paper,
supporting modern backbone architectures in addition to the original AlexNet.

---

## Overview

Outdoor scenes experience a wide range of lighting and weather conditions which
dramatically affect their appearance.  **DeepTransient** uses deep CNNs to
estimate two types of scene attributes from a single image:

| Model | Task | Output | Loss |
|-------|------|--------|------|
| **TransientNet** | Transient attribute estimation | 40-dim score vector ∈ [0,1] | MSE |
| **CloudyNet** | Sunny / cloudy classification | 2-class logits | Cross-entropy |

Three pre-training initialisations match the paper's naming convention:

| Suffix | Initialisation |
|--------|---------------|
| **-I** | ImageNet-1K (`--pretrained imagenet`) |
| **-P** | Places365 (`--pretrained places365`) |
| **-H** | Hybrid Places365+ImageNet (`--pretrained places365`, the hybrid checkpoint) |
| **-C** | CLIP (`--backbone clip_vit_b32` or `clip_vit_l14`; `--pretrained clip`) |

CLIP backbones use [OpenAI CLIP](https://github.com/openai/CLIP) weights loaded
via [open_clip](https://github.com/mlfoundations/open_clip).  With the backbone
frozen (`--freeze-backbone`) the CLIP linear-probe typically matches or exceeds
the best paper results in fewer than 10 epochs.

### Benchmark results (from the paper)

**Two-class weather classification** (normalised accuracy, higher is better):

| Method | Norm. Acc. |
|--------|-----------|
| Lu et al. (2014) | 53.1 ± 2.2 % |
| CloudyNet-I | 85.7 ± 0.5 % |
| CloudyNet-P | 86.1 ± 0.6 % |
| **CloudyNet-H** | **87.1 ± 0.3 %** |

**Transient attribute prediction** (mean MSE %, lower is better):

| Method | Avg. Error |
|--------|-----------|
| Laffont et al. (2014) | 4.2 % |
| TransientNet-I | 4.05 % |
| TransientNet-P | 3.87 % |
| **TransientNet-H** | **3.83 %** |

---

## Installation

```bash
git clone https://github.com/mvrl/deeptransient.git
cd deeptransient
pip install -e .
# Optional extras:
pip install pyyaml huggingface_hub
# CLIP backbones (clip_vit_b32, clip_vit_l14):
pip install open-clip-torch
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, torchvision ≥ 0.15.

---

## Datasets

### Transient Attributes Dataset

Download from the [official project page](https://transattr.cs.brown.edu/) and
arrange as follows:

```
<data_root>/
  imageLD/
    00000001/
      *.jpg
    00000002/
      ...
  annotations/
    annotations.tsv        # tab-separated: filename\tvalue,confidence ...
  holdout_split/
    training.txt           # one "webcam_id/filename" path per line
    test.txt
```

The official holdout split uses **81 webcams for training** and **20 for testing**.
If the `holdout_split/` directory is absent the code falls back to a
camera-id-based split (cameras 1–81 / 82–101).

### Two-Class Weather Dataset

Download from [Lu et al.'s project page](https://cs.brown.edu/~lbsun/sun/two_class_weather.html):

```
<data_root>/
  sunny/
    *.jpg
  cloudy/
    *.jpg
```

The dataset contains ~5 000 sunny and ~5 000 cloudy images.

---

## Training

### Using a config file

```bash
python train.py --config configs/transientnet_resnet50.yaml \
                --data-root /path/to/transient_attrs \
                --output-dir runs/transientnet_resnet50
```

### Using command-line flags

**TransientNet-I** (AlexNet, ImageNet init):
```bash
python train.py \
    --task transient_attrs \
    --backbone alexnet \
    --pretrained imagenet \
    --data-root /path/to/transient_attrs \
    --output-dir runs/transientnet_alexnet_imagenet
```

**CloudyNet-H** (AlexNet, Places365 init, fold 0 of 5):
```bash
python train.py \
    --task two_class_weather \
    --backbone alexnet \
    --pretrained places365 \
    --data-root /path/to/weather \
    --fold 0 \
    --output-dir runs/cloudynet_alexnet_places365_fold0
```

**Modern TransientNet** (ResNet-50):
```bash
python train.py \
    --task transient_attrs \
    --backbone resnet50 \
    --pretrained imagenet \
    --data-root /path/to/transient_attrs \
    --output-dir runs/transientnet_resnet50 \
    --amp
```

**CLIP TransientNet** (frozen backbone linear probe, recommended starting point):
```bash
python train.py \
    --config configs/transientnet_clip_vit_b32.yaml \
    --data-root /path/to/transient_attrs \
    --freeze-backbone \
    --output-dir runs/transientnet_clip_frozen
```

Optional full fine-tune from the linear-probe checkpoint:
```bash
python train.py \
    --config configs/transientnet_clip_vit_b32.yaml \
    --data-root /path/to/transient_attrs \
    --resume runs/transientnet_clip_frozen/checkpoint_best.pth \
    --lr 1e-5 --epochs 10 \
    --output-dir runs/transientnet_clip_finetune
```

Supported backbones: `alexnet`, `resnet18`, `resnet50`, `efficientnet_b0`,
`vit_b_16`, `clip_vit_b32`, `clip_vit_l14`.

---

## Evaluation

**TransientNet** (single test split):
```bash
python evaluate.py \
    --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \
    --data-root /path/to/transient_attrs \
    --per-attribute
```

**CloudyNet** (5-fold, requires one checkpoint per fold):
```bash
python evaluate.py \
    --task two_class_weather \
    --checkpoints \
        runs/cloudynet_fold0/checkpoint_best.pth \
        runs/cloudynet_fold1/checkpoint_best.pth \
        runs/cloudynet_fold2/checkpoint_best.pth \
        runs/cloudynet_fold3/checkpoint_best.pth \
        runs/cloudynet_fold4/checkpoint_best.pth \
    --data-root /path/to/weather
```

---

## Inference

```bash
python predict.py \
    --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \
    --image /path/to/image.jpg
```

Or load directly from HuggingFace Hub:
```bash
python predict.py \
    --hub-repo mvrl/deeptransient-transientnet-resnet50 \
    --image /path/to/image.jpg
```

Example output for TransientNet:
```
Image: /path/to/image.jpg
Top 10 attributes:
  day                  0.961  ████████████████████
  bright               0.894  █████████████████
  beautiful            0.812  ████████████████
  sunny                0.743  ██████████████
  warm                 0.721  ██████████████
  ...
```

Python API:
```python
import torch
from PIL import Image
from torchvision import transforms
from deeptransient.models import TransientNet

model = TransientNet(backbone="resnet50", pretrained="imagenet")
# Load weights:
ckpt = torch.load("checkpoint_best.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image = transform(Image.open("image.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    scores = model(image)  # (1, 40), values in [0, 1]
```

---

## Pushing to HuggingFace Hub

After training, push your best checkpoint to the Hub:

```bash
huggingface-cli login
python push_to_hub.py \
    --checkpoint runs/transientnet_resnet50/checkpoint_best.pth \
    --repo-id your-username/deeptransient-transientnet-resnet50
```

---

## Repository structure

```
deeptransient/           Python package
  models/
    transientnet.py      TransientNet, CloudyNet, backbone factory, get_transform
  data/
    transient_attrs.py   Transient Attributes Dataset loader
    two_class_weather.py Two-Class Weather Dataset loader
  metrics.py             Evaluation metrics
train.py                 Training script
evaluate.py              Evaluation script
predict.py               Single-image inference
push_to_hub.py           HuggingFace Hub upload
configs/                 YAML configuration files
  transientnet_alexnet.yaml
  transientnet_resnet50.yaml
  transientnet_clip_vit_b32.yaml
  cloudynet_alexnet.yaml
  cloudynet_resnet50.yaml
  cloudynet_clip_vit_b32.yaml
paper/                   Original WACV 2016 LaTeX source
```

---

## Citation

If you use this code or the pre-trained models, please cite:

```bibtex
@inproceedings{baltenberger2016fast,
  title     = {A Fast Method for Estimating Transient Scene Attributes},
  author    = {Baltenberger, Ryan and Zhai, Menghua and Greenwell, Connor
               and Workman, Scott and Jacobs, Nathan},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2016}
}
```

The Transient Attributes Dataset should also be cited:

```bibtex
@article{laffont2014transient,
  title   = {Transient Attributes for High-Level Understanding and Editing
             of Outdoor Scenes},
  author  = {Laffont, Pierre-Yves and Ren, Zhile and Tao, Xiaofeng
             and Qian, Chao and Hays, James},
  journal = {ACM Transactions on Graphics (SIGGRAPH)},
  volume  = {33},
  number  = {4},
  year    = {2014}
}
```

---

## License

MIT
