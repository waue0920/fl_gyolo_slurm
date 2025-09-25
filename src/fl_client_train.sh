#!/bin/bash
#
# Shell Script to Submit All Client Training Jobs for a Specific Round
#
# This script simplifies the process of launching parallel training jobs.
# It loops from 1 to CLIENT_NUM and submits an sbatch job for each client.


set -e
set -o pipefail

# # 先 source 專案根目錄的 env.sh
# source "${WROOT}/src/env.sh"

# --- 1. Argument Parsing ---
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <EXP_DIR> <WROOT> <ROUND> <CLIENT_NUM> <DATASET_NAME>"
    exit 1
fi

EXP_DIR=$1
WROOT=$2
ROUND=$3
CLIENT_NUM=$4
DATASET_NAME=$5


# --- 2. Common Configuration ---
# Extract EXP_ID from EXP_DIR for more meaningful wandb names
EXP_ID=$(basename "${EXP_DIR}")

# Use full EXP_ID as wandb project name (new approach)
# This groups all rounds of the same experiment under one wandb project
WANDB_PROJECT="${EXP_ID}"

# These are parameters shared across all clients for this round.
OUTPUT_PROJECT_PATH="${EXP_DIR}/client_outputs/${EXP_ID}"

if [ -z "$TRAIN_EXTRA_ARGS" ]; then
    TRAIN_EXTRA_ARGS="--epochs 50 --batch 8 --img 640 --workers 4 --hyp hyp.yaml --optimizer AdamW --flat-cos-lr --no-overlap --close-mosaic 2 --save-period 1 --noplots"
fi

# echo "======================================================================"
# echo "## Submitting all client training jobs for Round ${ROUND}"
# echo "## Experiment ID: ${EXP_ID}"
# echo "## Wandb Project: ${OUTPUT_PROJECT_PATH}"
# echo "## Experiment Directory: ${EXP_DIR}"
# echo "## Number of Clients: ${CLIENT_NUM}"
# echo "======================================================================"

# --- 3. Loop and Submit Jobs ---
for c in $(seq 1 ${CLIENT_NUM}); do
    # Per-client configuration
    CLIENT_DATA_YAML="federated_data/${DATASET_NAME}_${CLIENT_NUM}/c${c}.yaml"
    OUTPUT_NAME="r${ROUND}_c${c}"  # Simplified naming: r1_c1, r2_c3, etc.
    # WANDB_RUN_NAME 實際會被傳給 gyolo/caption/train.py 的 --name 參數
    # 這個參數同時決定 wandb run name 及子目錄
    WANDB_RUN_NAME="r${ROUND}_c${c}"  # Clean wandb run name
    SLURM_LOG_OUT="${EXP_DIR}/slurm_logs/client_${c}_round_${ROUND}.out"
    SLURM_LOG_ERR="${EXP_DIR}/slurm_logs/client_${c}_round_${ROUND}.out"

    # echo "--> Submitting training job for Client ${c}..."
    # echo "    File output: ${OUTPUT_PROJECT_PATH}/${OUTPUT_NAME}"
    # echo "    Wandb project: ${OUTPUT_PROJECT_PATH} (derived from file path)"
    # echo "    Wandb run: ${WANDB_RUN_NAME}"

    # The sbatch command submits the client_train.sb script with all necessary
    # arguments for the client_train.sh wrapper.
    # Note: YOLOv9 will use Path(--project).stem as wandb project name
    # and --name as wandb run name

    SBATCH_CMD=(
        sbatch
        --export=ALL
        --job-name="fl_c${c}_r${ROUND}"
        --output="${SLURM_LOG_OUT}"
        --error="${SLURM_LOG_ERR}"
        --chdir="${WROOT}"
        --nodes="1"
        --gpus-per-node="${CLIENT_GPUS}"
        --cpus-per-task="${CLIENT_CPUS}"
        --partition="${SLURM_PARTITION}"
        --account="${SLURM_ACCOUNT}"
        "${WROOT}/src/client_train.sb"
        "./src/client_train.sh"
        --data-yaml "${CLIENT_DATA_YAML}"
        --project-out "${OUTPUT_PROJECT_PATH}"
        --name-out "${WANDB_RUN_NAME}"
        --extra-args "${TRAIN_EXTRA_ARGS}"
    )
    echo "${SBATCH_CMD[@]}"
    "${SBATCH_CMD[@]}"
done

# echo "----------------------------------------------------------------------"
# echo "## All ${CLIENT_NUM} client jobs for Round ${ROUND} have been submitted."
# echo "## Monitor job status with: squeue -u \\\${USER}"
# echo "======================================================================"
