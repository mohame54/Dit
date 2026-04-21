#!/usr/bin/env bash
# ============================================================
# Dit  –  Distributed Training Launcher
#
# Usage:
#   bash train.sh [OPTIONS]
#
# All flags are optional; defaults are shown in the CONFIG
# section below. Pass any flag to override.
#
# Examples:
#   bash train.sh
#   bash train.sh --gpus 4 --epochs 500 --mp-dt float16
#   bash train.sh --resume checkpoint_300
#   bash train.sh --dist-mode fsdp --train-batch-sz 32 --push-hub False
# ============================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# DEFAULTS  –  every value here can be overridden via a flag
# ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/mohame54/Dit.git"
GDRIVE_DATA_ID="1I1uTVrPTwL3_rHyNVWpCMaFNWH0KpC2_"   # latent_data.zip
GDRIVE_CSV_ID="1HcVqvUxY0ddJHWj2g5P0fZQYB9DPpKjT"                                        # latent_vector_mapping.csv (leave empty to skip)
CUDA_VERSION="cu121"          # change to cu118 / cu124 / cu126 etc. as needed

NUM_GPUS=2
EPOCHS=1000
CUR_EPOCHS=0
TRAIN_BATCH_SZ=128
VAL_BATCH_SZ=128
MP_DT="bfloat16"
LOSS_TYPE="mse_loss"
DIST_MODE="ddp"
VAE_MODEL_ID="stabilityai/sdxl-vae"
NUM_GEN_STEPS=20
EPOCH_SAVE_FREQ=25
USE_MOUN="False"
PUSH_HUB="True"
USE_SCHEDULER="True"
SCHEDULER_TYPE="cosine"
SAVE_BEST="False"
FID_FREQ=0
FID_BATCH_SIZE=16
NUM_FID_SAMPLES=128
FID_FEATURE=64

DATA_DIR="content"
CHECKPOINTS_DIR="checkpoints"
LOGS_DIR="logs"
RESUME_DIR=""
RESUME_LOCAL=""       # local checkpoint path; skips HF Hub download (e.g. checkpoints/v2/checkpoint_400)
LR=""                 # empty = use value from scripts/opt_config.json
RUN_NAME=""           # empty = use CHECKPOINTS_DIR/LOGS_DIR as-is; set to scope under a subfolder (e.g. v2)
NUM_WORKERS=4         # DataLoader workers per GPU; increase if GPU idles waiting for data

HF_TOKEN="${HF_TOKEN:-}"

# ─────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash train.sh [OPTIONS]

Setup:
  --cuda-version VER      PyTorch CUDA wheel tag  (default: $CUDA_VERSION)
  --gpus N                GPUs per node           (default: $NUM_GPUS)
  --gdrive-csv-id ID      GDrive file ID for latent_vector_mapping.csv
                          (only downloaded if the file is missing)

Training:
  --epochs N              Total epochs            (default: $EPOCHS)
  --cur-epochs N          Starting epoch          (default: $CUR_EPOCHS)
  --train-batch-sz N      Train batch size        (default: $TRAIN_BATCH_SZ)
  --val-batch-sz N        Val batch size          (default: $VAL_BATCH_SZ)
  --mp-dt TYPE            Mixed precision dtype   (default: $MP_DT)
  --loss-type TYPE        Loss function           (default: $LOSS_TYPE)
  --dist-mode MODE        ddp | fsdp              (default: $DIST_MODE)
  --vae-model-id ID       VAE model id            (default: $VAE_MODEL_ID)
  --num-gen-steps N       Diffusion steps         (default: $NUM_GEN_STEPS)
  --epoch-save-freq N     Save every N epochs     (default: $EPOCH_SAVE_FREQ)
  --use-moun BOOL         Use Moun optimizer      (default: $USE_MOUN)
  --push-hub BOOL         Push to HF Hub          (default: $PUSH_HUB)
  --use-scheduler BOOL    Use LR scheduler        (default: $USE_SCHEDULER)
  --no-scheduler          Disable LR scheduler    (sets --use-scheduler False)
  --scheduler-type TYPE   cosine | step           (default: $SCHEDULER_TYPE)
  --save-best BOOL        Save best ckpt          (default: $SAVE_BEST)

FID:
  --fid-freq N            Eval FID every N epochs (default: $FID_FREQ)
  --fid-batch-size N      FID generation batch    (default: $FID_BATCH_SIZE)
  --num-fid-samples N     FID sample count        (default: $NUM_FID_SAMPLES)
  --fid-feature N         Inception feature dim   (default: $FID_FEATURE)

Paths:
  --data-dir PATH         Data directory          (default: $DATA_DIR)
  --dir-path-save PATH    Checkpoint save dir     (default: $CHECKPOINTS_DIR)
  --logs-save-dir PATH    Logs directory          (default: $LOGS_DIR)

Resume:
  --resume DIR            HF Hub checkpoint folder name  (e.g. checkpoint_300) – downloads from Hub
  --resume-local PATH     Local checkpoint directory     (e.g. checkpoints/v2/checkpoint_400) – no download
  --lr FLOAT              Override learning rate  (e.g. 5e-5); default: value in scripts/opt_config.json
  --run-name NAME         Scope checkpoints/logs under a subfolder (e.g. v2 → checkpoints/v2/, logs/v2/)
  --num-workers N         DataLoader workers per GPU     (default: $NUM_WORKERS)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)             usage ;;
        --cuda-version)        CUDA_VERSION="$2";       shift 2 ;;
        --gdrive-csv-id)       GDRIVE_CSV_ID="$2";      shift 2 ;;
        --gpus)                NUM_GPUS="$2";           shift 2 ;;
        --epochs)              EPOCHS="$2";             shift 2 ;;
        --cur-epochs)          CUR_EPOCHS="$2";         shift 2 ;;
        --train-batch-sz)      TRAIN_BATCH_SZ="$2";     shift 2 ;;
        --val-batch-sz)        VAL_BATCH_SZ="$2";       shift 2 ;;
        --mp-dt)               MP_DT="$2";              shift 2 ;;
        --loss-type)           LOSS_TYPE="$2";          shift 2 ;;
        --dist-mode)           DIST_MODE="$2";          shift 2 ;;
        --vae-model-id)        VAE_MODEL_ID="$2";       shift 2 ;;
        --num-gen-steps)       NUM_GEN_STEPS="$2";      shift 2 ;;
        --epoch-save-freq)     EPOCH_SAVE_FREQ="$2";    shift 2 ;;
        --use-moun)            USE_MOUN="$2";           shift 2 ;;
        --push-hub)            PUSH_HUB="$2";           shift 2 ;;
        --use-scheduler)       USE_SCHEDULER="$2";      shift 2 ;;
        --no-scheduler)        USE_SCHEDULER="False";   shift 1 ;;
        --scheduler-type)      SCHEDULER_TYPE="$2";     shift 2 ;;
        --save-best)           SAVE_BEST="$2";          shift 2 ;;
        --fid-freq)            FID_FREQ="$2";           shift 2 ;;
        --fid-batch-size)      FID_BATCH_SIZE="$2";     shift 2 ;;
        --num-fid-samples)     NUM_FID_SAMPLES="$2";    shift 2 ;;
        --fid-feature)         FID_FEATURE="$2";        shift 2 ;;
        --data-dir)            DATA_DIR="$2";           shift 2 ;;
        --dir-path-save)       CHECKPOINTS_DIR="$2";    shift 2 ;;
        --logs-save-dir)       LOGS_DIR="$2";           shift 2 ;;
        --resume)
            RESUME_DIR="$2"
            # Auto-derive cur-epochs from folder name if not explicitly set
            # e.g. checkpoint_300 → 300
            if [[ "$CUR_EPOCHS" == "0" ]]; then
                CUR_EPOCHS="${RESUME_DIR##*_}"
            fi
            shift 2
            ;;
        --resume-local)
            RESUME_LOCAL="$2"
            # Auto-derive cur-epochs from folder name if not explicitly set
            # e.g. checkpoints/v2/checkpoint_400 → 400
            if [[ "$CUR_EPOCHS" == "0" ]]; then
                CUR_EPOCHS="${RESUME_LOCAL##*_}"
            fi
            shift 2
            ;;
        --lr)                  LR="$2";                 shift 2 ;;
        --run-name)            RUN_NAME="$2";           shift 2 ;;
        --num-workers)         NUM_WORKERS="$2";        shift 2 ;;
        *)
            echo "Unknown option: $1  (run with --help to see all options)"
            exit 1
            ;;
    esac
done

# Apply run name scoping after all flags are parsed
if [[ -n "$RUN_NAME" ]]; then
    CHECKPOINTS_DIR="${CHECKPOINTS_DIR}/${RUN_NAME}"
    LOGS_DIR="${LOGS_DIR}/${RUN_NAME}"
fi

# ─────────────────────────────────────────────────────────────
# 1. Clone repository (skip if already inside the repo)
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$SCRIPT_DIR/main.py" ]]; then
    echo "==> Cloning repository..."
    git clone "$REPO_URL"
    cd Dit
else
    echo "==> Already inside repo, skipping clone."
    cd "$SCRIPT_DIR"
fi

# ─────────────────────────────────────────────────────────────
# 2. Install dependencies  (runs first, before any other work)
# ─────────────────────────────────────────────────────────────
if python3 -c "import torch" &>/dev/null; then
    echo "==> PyTorch already installed ($(python3 -c 'import torch; print(torch.__version__)')) – skipping."
else
    echo "==> Installing PyTorch (${CUDA_VERSION})..."
    pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" -q
fi

echo "==> Installing project requirements..."
pip install -r requirements.txt -q

# ─────────────────────────────────────────────────────────────
# 3. Set Hugging Face token
# ─────────────────────────────────────────────────────────────
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
    echo "==> HF_TOKEN is set."
else
    echo "WARNING: HF_TOKEN is not set – Hub uploads will fail."
    echo "         Export it before running: export HF_TOKEN=hf_..."
fi

# ─────────────────────────────────────────────────────────────
# 4a. Download latent_vector_mapping.csv (skip if already uploaded)
# ─────────────────────────────────────────────────────────────
if [[ ! -f "content/latent_vector_mapping.csv" ]]; then
    if [[ -n "$GDRIVE_CSV_ID" ]]; then
        echo "==> Downloading latent_vector_mapping.csv from Google Drive..."
        gdown "https://drive.google.com/uc?id=${GDRIVE_CSV_ID}" -O latent_vector_mapping.csv
    else
        echo "ERROR: latent_vector_mapping.csv not found and --gdrive-csv-id is not set."
        echo "       Either upload the file manually or pass --gdrive-csv-id <FILE_ID>."
        exit 1
    fi
else
    echo "==> latent_vector_mapping.csv already present, skipping download."
fi

# ─────────────────────────────────────────────────────────────
# 4b. Download and extract latent vectors zip (skip if already present)
# ─────────────────────────────────────────────────────────────
if [[ ! -d "latent_vectors" ]]; then
    echo "==> Downloading latent_data.zip from Google Drive..."
    gdown "https://drive.google.com/uc?id=${GDRIVE_DATA_ID}" -O latent_data.zip

    echo "==> Extracting latent vectors..."
    unzip -q latent_data.zip
    rm -f latent_data.zip
else
    echo "==> Latent vectors already present, skipping download."
fi

# ─────────────────────────────────────────────────────────────
# 5. Prepare train / val splits (skip if already present)
# ─────────────────────────────────────────────────────────────
if [[ ! -f "$DATA_DIR/train.csv" ]] || [[ ! -f "$DATA_DIR/val.csv" ]]; then
    echo "==> Generating train/val splits..."
    python3 scripts/prepare_data.py --data-dir "$DATA_DIR"
else
    echo "==> Splits already exist, skipping."
fi

# ─────────────────────────────────────────────────────────────
# 6. Launch distributed training
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Starting training"
printf "   %-20s %s\n" "GPUs:"          "$NUM_GPUS"
printf "   %-20s %s\n" "Epochs:"        "$EPOCHS"
printf "   %-20s %s\n" "Dist mode:"     "$DIST_MODE"
printf "   %-20s %s\n" "Mixed prec:"    "$MP_DT"
printf "   %-20s %s\n" "Train batch:"   "$TRAIN_BATCH_SZ"
printf "   %-20s %s\n" "Val batch:"     "$VAL_BATCH_SZ"
printf "   %-20s %s\n" "Scheduler:"     "$SCHEDULER_TYPE (enabled=$USE_SCHEDULER)"
printf "   %-20s %s\n" "FID freq:"      "$FID_FREQ"
printf "   %-20s %s\n" "Push to Hub:"   "$PUSH_HUB"
printf "   %-20s %s\n" "Checkpoints:"   "$CHECKPOINTS_DIR"
printf "   %-20s %s\n" "Logs:"          "$LOGS_DIR"
if [[ -n "$RESUME_DIR" ]]; then
    printf "   %-20s %s\n" "Resuming from:" "$RESUME_DIR  (cur_epochs=$CUR_EPOCHS)"
fi
if [[ -n "$LR" ]]; then
    printf "   %-20s %s\n" "LR override:" "$LR"
fi
echo "============================================================"
echo ""

RESUME_ARGS=""
if [[ -n "$RESUME_LOCAL" ]]; then
    RESUME_ARGS="--resume-local ${RESUME_LOCAL} --cur-epochs ${CUR_EPOCHS}"
elif [[ -n "$RESUME_DIR" ]]; then
    RESUME_ARGS="--resume-dir ${RESUME_DIR} --cur-epochs ${CUR_EPOCHS}"
fi

LR_ARGS=""
if [[ -n "$LR" ]]; then
    LR_ARGS="--lr ${LR}"
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
    python3 main_single_gpu.py \
        --epochs            "$EPOCHS"           \
        --train-batch-sz    "$TRAIN_BATCH_SZ"   \
        --val-batch-sz      "$VAL_BATCH_SZ"     \
        --data-dir-path     "$DATA_DIR"         \
        --dir-path-save     "$CHECKPOINTS_DIR"  \
        --logs-save-dir     "$LOGS_DIR"         \
        --epoch-save-freq   "$EPOCH_SAVE_FREQ"  \
        --num-gen-steps     "$NUM_GEN_STEPS"    \
        --mp-dt             "$MP_DT"            \
        --vae-model-id      "$VAE_MODEL_ID"     \
        --use-moun          "$USE_MOUN"         \
        --loss-type         "$LOSS_TYPE"        \
        --push-hub          "$PUSH_HUB"         \
        --use-scheduler     "$USE_SCHEDULER"    \
        --scheduler-type    "$SCHEDULER_TYPE"   \
        --save-best         "$SAVE_BEST"        \
        --fid-freq          "$FID_FREQ"         \
        --fid-batch-size    "$FID_BATCH_SIZE"   \
        --num-fid-samples   "$NUM_FID_SAMPLES"  \
        --fid-feature       "$FID_FEATURE"      \
        --num-workers       "$NUM_WORKERS"      \
        $RESUME_ARGS \
        $LR_ARGS
else
    torchrun --nproc_per_node="$NUM_GPUS" main.py \
        --epochs            "$EPOCHS"           \
        --train-batch-sz    "$TRAIN_BATCH_SZ"   \
        --val-batch-sz      "$VAL_BATCH_SZ"     \
        --data-dir-path     "$DATA_DIR"         \
        --dir-path-save     "$CHECKPOINTS_DIR"  \
        --logs-save-dir     "$LOGS_DIR"         \
        --epoch-save-freq   "$EPOCH_SAVE_FREQ"  \
        --num-gen-steps     "$NUM_GEN_STEPS"    \
        --mp-dt             "$MP_DT"            \
        --vae-model-id      "$VAE_MODEL_ID"     \
        --use-moun          "$USE_MOUN"         \
        --loss-type         "$LOSS_TYPE"        \
        --push-hub          "$PUSH_HUB"         \
        --dist-mode         "$DIST_MODE"        \
        --use-scheduler     "$USE_SCHEDULER"    \
        --scheduler-type    "$SCHEDULER_TYPE"   \
        --save-best         "$SAVE_BEST"        \
        --fid-freq          "$FID_FREQ"         \
        --fid-batch-size    "$FID_BATCH_SIZE"   \
        --num-fid-samples   "$NUM_FID_SAMPLES"  \
        --fid-feature       "$FID_FEATURE"      \
        --num-workers       "$NUM_WORKERS"      \
        $RESUME_ARGS \
        $LR_ARGS
fi
