"""HuggingFace Hub utilities for DeepTransient.

This module exposes :func:`push_to_hub`, which uploads a trained checkpoint
and an auto-generated model card to a HuggingFace Hub repository.

It is used internally by ``train.py --push-to-hub`` and can also be called
directly from the standalone ``push_to_hub.py`` script.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import torch


def _make_model_card(config: dict, repo_id: str) -> str:
    task = config.get("task", "unknown")
    backbone = config.get("backbone", "unknown")
    pretrained = config.get("pretrained", "unknown")

    if task == "transient_attrs":
        task_display = "Transient Attribute Estimation (regression, 40 attributes)"
        tags = "transient-attributes, outdoor-scenes, regression"
    else:
        task_display = "Two-Class Weather Classification (sunny / cloudy)"
        tags = "weather-classification, outdoor-scenes, binary-classification"

    return textwrap.dedent(
        f"""
        ---
        license: mit
        tags:
          - pytorch
          - computer-vision
          - {tags}
        pipeline_tag: image-classification
        ---

        # DeepTransient – {task_display}

        This checkpoint is part of the [DeepTransient](https://github.com/mvrl/deeptransient)
        release, a PyTorch reimplementation of:

        > Baltenberger, R., Zhai, M., Greenwell, C., Workman, S., & Jacobs, N.
        > **A Fast Method for Estimating Transient Scene Attributes.**
        > WACV 2016.

        ## Model details

        | Property | Value |
        |----------|-------|
        | Task | {task_display} |
        | Backbone | `{backbone}` |
        | Pretrained init | `{pretrained}` |

        ## Usage

        ```python
        import torch
        from huggingface_hub import hf_hub_download
        from deeptransient.models import TransientNet, CloudyNet

        ckpt_path = hf_hub_download(repo_id="{repo_id}", filename="checkpoint_best.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = TransientNet.from_config(ckpt["config"])  # or CloudyNet
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        ```

        Or use the CLI::

            python predict.py --hub-repo {repo_id} --image /path/to/image.jpg

        ## Citation

        ```bibtex
        @inproceedings{{baltenberger2016fast,
          title     = {{A Fast Method for Estimating Transient Scene Attributes}},
          author    = {{Baltenberger, Ryan and Zhai, Menghua and Greenwell, Connor
                       and Workman, Scott and Jacobs, Nathan}},
          booktitle = {{IEEE Winter Conference on Applications of Computer Vision (WACV)}},
          year      = {{2016}}
        }}
        ```
        """
    ).strip()


def push_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    private: bool = False,
    example_image: str | None = None,
) -> None:
    """Upload a checkpoint and auto-generated model card to HuggingFace Hub.

    Args:
        checkpoint_path: Local path to the ``.pth`` checkpoint file.
        repo_id: HuggingFace Hub repository id, e.g. ``"myuser/my-model"``.
        private: Whether to create the repository as private.
        example_image: Optional local path to an example image to include.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    print(f"Uploading checkpoint to {repo_id} …")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="checkpoint_best.pth",
        repo_id=repo_id,
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj=json.dumps(config, indent=2).encode(),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
    )

    api.upload_file(
        path_or_fileobj=_make_model_card(config, repo_id).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    if example_image is not None:
        api.upload_file(
            path_or_fileobj=str(example_image),
            path_in_repo=f"example{Path(example_image).suffix}",
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"✓ Model pushed to https://huggingface.co/{repo_id}")

