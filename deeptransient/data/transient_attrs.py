"""Transient Attributes Dataset (Laffont et al., SIGGRAPH 2014).

Dataset page: https://transattr.cs.brown.edu/

Expected directory layout
--------------------------
::

    <root>/
      imageLD/               # images organised by webcam id
        00000001/
          *.jpg
        00000002/
          *.jpg
        ...
      annotations/
        annotations.tsv      # tab-separated: filename\\tattribute_value,confidence ...
      holdout_split/
        training.txt         # one "webcam_id/filename" path per line
        test.txt

The ``holdout_split/`` files define the official 81-webcam train /
20-webcam test split used in the paper.  If they are absent, the
dataset falls back to the camera-based split defined by
:data:`TRAIN_CAMERAS` and :data:`TEST_CAMERAS`.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Official webcam ids for train and test splits (1-indexed).
# Cameras 1-81 → training, cameras 82-101 → test (as in the paper).
TRAIN_CAMERAS: list[int] = list(range(1, 82))
TEST_CAMERAS: list[int] = list(range(82, 102))

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


class TransientAttributesDataset(Dataset):
    """PyTorch Dataset for the Transient Attributes benchmark.

    Each item is an ``(image, labels)`` pair where *image* is a
    ``(3, H, W)`` float tensor and *labels* is a ``(40,)`` float tensor
    of attribute scores in ``[0, 1]``.

    Args:
        root: Path to the dataset root directory (see module docstring).
        split: ``"train"`` or ``"test"``.
        transform: Optional image transform.  A sensible default is applied
            when ``None`` is given.
        image_dir: Name of the sub-directory containing images.
            Defaults to ``"imageLD"``; use ``"imageAlignedLD"`` for the
            larger-resolution variant of the dataset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_dir: str = "imageLD",
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else _default_transform(split)
        self.image_dir = image_dir

        self._annotations = self._load_annotations()
        self._samples = self._build_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_annotations(self) -> dict[str, list[float]]:
        """Return a dict mapping bare filename → list of 40 attribute values."""
        ann_path = self.root / "annotations" / "annotations.tsv"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {ann_path}\n"
                "Please download the dataset from https://transattr.cs.brown.edu/"
            )

        annotations: dict[str, list[float]] = {}
        with ann_path.open(newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                filename = row[0].strip()
                # Each attribute column is "value,confidence"; we use the value.
                attrs = [float(col.split(",")[0]) for col in row[1:] if col.strip()]
                annotations[filename] = attrs
        return annotations

    def _build_samples(self) -> list[tuple[Path, list[float]]]:
        """Build the list of (image_path, labels) for the chosen split."""
        holdout_file = self.root / "holdout_split" / (
            "training.txt" if self.split == "train" else "test.txt"
        )

        if holdout_file.exists():
            samples = self._samples_from_holdout(holdout_file)
        else:
            samples = self._samples_from_camera_split()

        if not samples:
            raise RuntimeError(
                f"No samples found for split={self.split!r} in {self.root}. "
                "Check that the dataset has been downloaded correctly."
            )
        return samples

    def _samples_from_holdout(
        self, holdout_file: Path
    ) -> list[tuple[Path, list[float]]]:
        """Use official holdout split files."""
        samples = []
        with holdout_file.open() as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                bare_name = os.path.basename(rel_path)
                if bare_name not in self._annotations:
                    continue
                img_path = self.root / self.image_dir / rel_path
                samples.append((img_path, self._annotations[bare_name]))
        return samples

    def _samples_from_camera_split(self) -> list[tuple[Path, list[float]]]:
        """Fall back to camera-id-based split when holdout files are absent."""
        camera_ids = TRAIN_CAMERAS if self.split == "train" else TEST_CAMERAS
        img_root = self.root / self.image_dir
        samples = []
        for cam_id in camera_ids:
            cam_dir = img_root / f"{cam_id:08d}"
            if not cam_dir.is_dir():
                continue
            for img_path in sorted(cam_dir.iterdir()):
                bare_name = img_path.name
                if bare_name in self._annotations:
                    samples.append((img_path, self._annotations[bare_name]))
        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, attrs = self._samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(attrs, dtype=torch.float32)
        return image, labels
