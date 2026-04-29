#!/usr/bin/env bash
# Sequential per-GPU runner for the CloudyNet 5-fold sweep.
# Usage: ./scripts/run_all_cloudynet.sh <gpu_id> <queue_name>
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

GPU="${1:?gpu id}"
QUEUE="${2:?queue name (a or b)}"
DATA_ROOT=/data/jacobsn/two_class_weather/weather_database
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
        --task two_class_weather --data-root "${DATA_ROOT}" \
        --output-dir "${out}" --workers 8 \
        "$@" 2>&1 | tee "runs/${name}.log"
    echo "[${QUEUE}/done] ${name}"
}

# Helper that loops over the 5 folds for a (backbone, init, optim) configuration.
sweep_5fold() {
    local prefix="$1"; shift
    for fold in 0 1 2 3 4; do
        run "${prefix}_fold${fold}" --fold "${fold}" "$@"
    done
}

case "${QUEUE}" in
  a)
    # Paper replication on AlexNet (SGD, matches the WACV configuration).
    sweep_5fold cloudynet_alexnet_imagenet \
        --backbone alexnet --pretrained imagenet \
        --dropout 0.5 --epochs 20 --batch-size 64 \
        --lr 1e-3 --weight-decay 5e-4 \
        --lr-step-size 10 --lr-gamma 0.1
    sweep_5fold cloudynet_alexnet_places365 \
        --backbone alexnet --pretrained places365 \
        --dropout 0.5 --epochs 20 --batch-size 64 \
        --lr 1e-3 --weight-decay 5e-4 \
        --lr-step-size 10 --lr-gamma 0.1
    # CLIP linear probe — fast, frozen backbone.
    sweep_5fold cloudynet_clip_b32_frozen_adam \
        --backbone clip_vit_b32 --pretrained clip --freeze-backbone \
        --dropout 0.0 --epochs 15 --batch-size 256 \
        --lr 1e-3 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    ;;
  b)
    # Modern backbone with Adam (SGD+CE works fine for AlexNet, but Adam is
    # more robust on ResNet-50; this gives a like-for-like comparison).
    sweep_5fold cloudynet_resnet50_imagenet_adam \
        --backbone resnet50 --pretrained imagenet \
        --dropout 0.2 --epochs 15 --batch-size 64 \
        --lr 3e-4 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    sweep_5fold cloudynet_resnet50_places365_adam \
        --backbone resnet50 --pretrained places365 \
        --dropout 0.2 --epochs 15 --batch-size 64 \
        --lr 3e-4 --weight-decay 1e-4 \
        --optimizer adam --scheduler cosine
    ;;
  *)
    echo "unknown queue: ${QUEUE}" >&2; exit 1;;
esac
