"""Two-Class Weather Dataset (Lu et al., 2014).

Dataset page: https://cs.brown.edu/~lbsun/sun/two_class_weather.html

Expected directory layout
--------------------------
::

    <root>/
      sunny/       # sunny images
        *.jpg
      cloudy/      # cloudy images
        *.jpg

The dataset contains ~5 000 sunny and ~5 000 cloudy images.
Following the protocol of Lu et al., we randomly partition each class
80 % / 20 % for training and testing, repeating this process five times
to report mean ± std normalised accuracy.

Label encoding: **0 = sunny**, **1 = cloudy**.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def _default_transform(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_DEFAULT_MEAN, _DEFAULT_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_DEFAULT_MEAN, _DEFAULT_STD),
        ]
    )


class TwoClassWeatherDataset(Dataset):
    """PyTorch Dataset for the two-class sunny/cloudy benchmark.

    Each item is an ``(image, label)`` pair where *label* is an integer:
    ``0`` for *sunny* and ``1`` for *cloudy*.

    Args:
        root: Path to the dataset root directory (see module docstring).
        split: ``"train"`` or ``"test"``.
        fold: Random-split index in ``[0, 4]`` (five folds total).  A
            different random seed is used for each fold so that the five
            train/test partitions are independent.
        transform: Optional image transform.  A sensible default is
            applied when ``None`` is given.
        train_ratio: Fraction of images used for training (default 0.8).
    """

    CLASSES: list[str] = ["sunny", "cloudy"]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        fold: int = 0,
        transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not 0 <= fold < 5:
            raise ValueError(f"fold must be in [0, 4], got {fold}")

        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.transform = transform if transform is not None else _default_transform(split)

        self._samples = self._build_samples(train_ratio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_images(self, class_dir: Path, label: int) -> list[tuple[Path, int]]:
        """Return sorted list of (path, label) pairs for one class directory."""
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                "Please download the dataset and place sunny/cloudy sub-directories "
                "inside the dataset root."
            )
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        paths = sorted(p for p in class_dir.iterdir() if p.suffix in exts)
        return [(p, label) for p in paths]

    def _build_samples(self, train_ratio: float) -> list[tuple[Path, int]]:
        """Build train or test sample list for the current fold."""
        sunny = self._collect_images(self.root / "sunny", label=0)
        cloudy = self._collect_images(self.root / "cloudy", label=1)

        rng = random.Random(self.fold)  # deterministic per fold

        def split_class(items: list) -> tuple[list, list]:
            shuffled = items.copy()
            rng.shuffle(shuffled)
            n_train = int(len(shuffled) * train_ratio)
            return shuffled[:n_train], shuffled[n_train:]

        sunny_train, sunny_test = split_class(sunny)
        cloudy_train, cloudy_test = split_class(cloudy)

        if self.split == "train":
            samples = sunny_train + cloudy_train
        else:
            samples = sunny_test + cloudy_test

        rng.shuffle(samples)
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self._samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
