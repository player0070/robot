#!/bin/bash
# Diffusion Policy PushT 재현 스크립트
# 실행: bash reproduce_pusht.sh [lowdim|image] [train|eval]
# 예시: bash reproduce_pusht.sh lowdim train

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE=${1:-lowdim}   # lowdim | image
ACTION=${2:-train}  # train | eval

CONDA_ENV="robodiff"
DEVICE="cuda:0"
SEED=42

# conda 활성화 경로 탐색
CONDA_BASE=$(conda info --base 2>/dev/null) || \
CONDA_BASE=$(find /home /opt /root -maxdepth 4 -name "conda" -path "*/bin/conda" 2>/dev/null | head -1 | sed 's|/bin/conda||')
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "================================================"
echo " Diffusion Policy Reproduction"
echo " Mode   : $MODE"
echo " Action : $ACTION"
echo " Device : $DEVICE"
echo " Env    : $CONDA_ENV"
echo "================================================"

# ----- 환경 설치 확인 -----
setup_env() {
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        echo "[Setup] conda 환경 생성 중..."
        conda env create -f conda_environment_cuda12.yaml
    fi
    conda activate "$CONDA_ENV"
    # diffusion_policy 패키지 설치 (editable)
    if ! python -c "import diffusion_policy" 2>/dev/null; then
        echo "[Setup] diffusion_policy 패키지 설치 중..."
        pip install -e .
    fi
}

# ----- lowdim 학습 -----
train_lowdim() {
    echo "[Train] PushT lowdim (state-based) 학습 시작..."
    echo "  Config : train_diffusion_unet_lowdim_workspace"
    echo "  Data   : data/pusht/pusht_cchi_v7_replay.zarr"
    echo "  Epochs : 3000 (논문: 5000, 빠른 확인용으로 줄임)"
    echo ""

    python train.py \
        --config-name=train_diffusion_unet_lowdim_workspace \
        training.device="$DEVICE" \
        training.seed=$SEED \
        training.num_epochs=3000 \
        training.rollout_every=100 \
        training.checkpoint_every=100 \
        logging.mode=offline \
        hydra.run.dir="data/outputs/reproduce_pusht_lowdim"
}

# ----- image 학습 -----
train_image() {
    echo "[Train] PushT image-based (CNN) 학습 시작..."
    echo "  Config : train_diffusion_unet_hybrid_workspace"
    echo "  Data   : data/pusht/pusht_cchi_v7_replay.zarr"
    echo ""

    python train.py \
        --config-name=train_diffusion_unet_hybrid_workspace \
        training.device="$DEVICE" \
        training.seed=$SEED \
        training.num_epochs=3050 \
        training.rollout_every=50 \
        training.checkpoint_every=50 \
        dataloader.batch_size=64 \
        logging.mode=offline \
        hydra.run.dir="data/outputs/reproduce_pusht_image"
}

# ----- 평가 -----
eval_checkpoint() {
    local run_dir=$1
    CKPT=$(ls "$run_dir/checkpoints/"*.ckpt 2>/dev/null | grep -v last | sort -t= -k2 -n | tail -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls "$run_dir/checkpoints/last.ckpt" 2>/dev/null)
    fi
    if [ -z "$CKPT" ]; then
        echo "[Eval] 체크포인트를 찾지 못했습니다: $run_dir/checkpoints/"
        exit 1
    fi
    echo "[Eval] 체크포인트: $CKPT"
    python eval.py \
        --checkpoint "$CKPT" \
        --output_dir "data/outputs/eval_$(basename $run_dir)" \
        --device "$DEVICE"
}

# ----- 실행 -----
setup_env

if [ "$ACTION" = "train" ]; then
    if [ "$MODE" = "lowdim" ]; then
        train_lowdim
    elif [ "$MODE" = "image" ]; then
        train_image
    else
        echo "Usage: $0 [lowdim|image] [train|eval]"
        exit 1
    fi
elif [ "$ACTION" = "eval" ]; then
    if [ "$MODE" = "lowdim" ]; then
        eval_checkpoint "data/outputs/reproduce_pusht_lowdim"
    elif [ "$MODE" = "image" ]; then
        eval_checkpoint "data/outputs/reproduce_pusht_image"
    fi
else
    echo "Usage: $0 [lowdim|image] [train|eval]"
    exit 1
fi

echo ""
echo "[Done] 완료!"
