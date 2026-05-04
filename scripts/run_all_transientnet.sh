#!/usr/bin/env bash
# Sequential per-GPU runner for the TransientNet sweep.
# Usage: ./scripts/run_all_transientnet.sh <gpu_id> <queue_name>
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

GPU="${1:?gpu id}"
QUEUE="${2:?queue name (a or b)}"
DATA_ROOT=/data/jacobsn/transient_attrs
mkdir -p runs

run() {
    local name="$1"; shift
    local out="runs/${name}"
    if [[ -f "${out}/checkpoint_best.pth" ]]; then
        echo "[${QUEUE}/skip] ${name} already trained"
        return 0
    fi
    echo "[${QUEUE}/start] ${name}"
    CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
        --data-root "${DATA_ROOT}" \
        --image-dir imageAlignedLD \
        --output-dir "${out}" \
        --workers 8 \
        "$@" 2>&1 | tee "runs/${name}.log"
    echo "[${QUEUE}/done] ${name}"
}

case "${QUEUE}" in
  a)
    # Queue A — paper-style AlexNet runs (SGD works fine for these because the
    # 4096-dim head has plenty of redundancy).
    run transientnet_alexnet_places365 \
        --task transient_attrs --backbone alexnet --pretrained places365 \
        --dropout 0.5 --epochs 50 --batch-size 64 --lr 1e-3 --weight-decay 5e-4 \
        --lr-step-size 20 --lr-gamma 0.1
    # Modern backbones with sigmoid+MSE need Adam — plain SGD stalls.
    run transientnet_resnet50_imagenet_adam \
        --task transient_attrs --backbone resnet50 --pretrained imagenet \
        --dropout 0.2 --epochs 20 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    run transientnet_resnet50_places365_adam \
        --task transient_attrs --backbone resnet50 --pretrained places365 \
        --dropout 0.2 --epochs 20 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    ;;
  b)
    run transientnet_resnet18_imagenet_adam \
        --task transient_attrs --backbone resnet18 --pretrained imagenet \
        --dropout 0.2 --epochs 20 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    run transientnet_efficientnet_b0_imagenet_adam \
        --task transient_attrs --backbone efficientnet_b0 --pretrained imagenet \
        --dropout 0.2 --epochs 20 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    run transientnet_vit_b16_imagenet_adam \
        --task transient_attrs --backbone vit_b_16 --pretrained imagenet \
        --dropout 0.2 --epochs 15 --batch-size 64 --lr 1e-4 --weight-decay 1e-4 \
        --optimizer adamw --scheduler cosine --amp
    run transientnet_clip_b32_finetune \
        --task transient_attrs --backbone clip_vit_b32 --pretrained clip \
        --dropout 0.0 --epochs 10 --batch-size 64 --lr 1e-5 \
        --weight-decay 1e-4 --optimizer adamw --scheduler cosine --amp \
        --init-from runs/transientnet_clip_b32_frozen_adam/checkpoint_best.pth
    run transientnet_clip_l14_frozen_adam \
        --task transient_attrs --backbone clip_vit_l14 --pretrained clip \
        --freeze-backbone --dropout 0.0 --epochs 30 --batch-size 128 \
        --lr 1e-3 --weight-decay 1e-4 --optimizer adam --scheduler cosine
    ;;
  *)
    echo "unknown queue: ${QUEUE}" >&2; exit 1;;
esac
