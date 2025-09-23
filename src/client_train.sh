#!/bin/bash

# ===================================================================================
# Decoupled Executor Script for a Single Client Training Task
# ===================================================================================
# This script is a self-contained training executor. It is called by a Slurm
# script (like client_train.sb) and is responsible for executing the
# YOLOv9 training in a Singularity container.
#
# It is fully parameterized and does not depend on environment variables like
# EXP_ID or ROUND_NUM. All paths and hyperparameters are passed as arguments.
# ===================================================================================

set -e # Exit immediately if a command exits with a non-zero status.


# --- 0. Argument Parsing ---
# Initialize variables
DATA_YAML=""

## 參數 PROJECT_OUT 實際會被傳給 gyolo/caption/train.py 的 --project
## 這個參數同時決定 wandb project name 及訓練結果主目錄
PROJECT_OUT=""
## 參數 NAME_OUT 實際會被傳給 gyolo/caption/train.py 的 --name
## 這個參數同時決定 wandb run name 及子目錄
NAME_OUT=""
EXTRA_ARGS=""



while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data-yaml) DATA_YAML="$2"; shift ;;
        --project-out) PROJECT_OUT="$2"; shift ;;
        --name-out) NAME_OUT="$2"; shift ;;
        --extra-args) EXTRA_ARGS="$2"; shift ;; # Pass extra arguments as a single string
        *) echo "Unknown parameter passed: $1"; exit 1 ;; # Handle unknown parameters
    esac
    shift
done

# Verify required arguments
if [ -z "${DATA_YAML}" ]  || [ -z "${PROJECT_OUT}" ] || [ -z "${NAME_OUT}" ]; then
    echo "Usage: $0 \
    --data-yaml <path_to_client.yaml> \
    --project-out <path_to_output_project_dir> \
    --name-out <output_run_name> \
    [--extra-args \"--epochs 50 --batch 8\"]"
    exit 1
fi


# --- 1. Project Root and Environment Setup ---
# Use WROOT environment variable (set by user before execution)
if [ -z "${WROOT}" ]; then
    echo "Error: WROOT environment variable is not set"
    echo "Please run: export WROOT=/path/to/project/root"
    exit 1
fi

if [ -z "${SINGULARITY_IMG}" ]; then
    SINGULARITY_IMG="${WROOT}/gyolo_ngc2306.sif"
fi

if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    SLURM_GPUS_ON_NODE=$(nvidia-smi -L | wc -l)
fi

if [ -z "$MODEL_CFG" ]; then
    MODEL_CFG="${WROOT}/gyolo/models/caption/gyolo.yaml"
fi

if [ -z "$INITIAL_WEIGHTS" ]; then
    INITIAL_WEIGHTS="${WROOT}/gyolo.pt"
fi


## 下面這段多節點運算會導致port 衝突而failed
# --- NCCL/SLURM 多節點環境變數設定 --- 
# NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
# NNODES=${SLURM_NNODES:-1}
# NODE_RANK=${SLURM_NODEID:-0}
# if [ -z "$MASTER_ADDR" ]; then
#     MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# fi
# MASTER_PORT=9527
## 但由於 Gyolo 需要用torchrun (DPP) 來執行，不可用 python (DP) 單機，因此以下參數須設定
NGPU=${SLURM_GPUS_ON_NODE:-1}
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=9527


if [ -z "$CLIENT_GPUS" ]; then
    CLIENT_GPUS=$(nvidia-smi -L | wc -l)
fi
DEVICE_LIST=$(seq -s, 0 $(($CLIENT_GPUS-1)) | paste -sd, -)


echo "================================================================================"
echo ">> Starting Client Training (Decoupled)"
echo ">> Project Root:    ${WROOT}"
echo ">> Data YAML:       ${DATA_YAML}"
echo ">> Input Weights:   ${INITIAL_WEIGHTS}"
echo ">> Wandb & Output Project:  ${PROJECT_OUT}"
echo ">> Wandb & Output Name:     ${NAME_OUT}"
echo ">> Device List:     ${DEVICE_LIST}"
echo ">> Extra Args:      ${EXTRA_ARGS}"
echo "================================================================================"

# Check if files exist before proceeding

if [ ! -f "${WROOT}/${DATA_YAML}" ]; then
    echo "Error: Data YAML file not found at ${WROOT}/${DATA_YAML}"
    exit 1
fi
if [ ! -f "${SINGULARITY_IMG}" ]; then
    echo "Error: Singularity image not found at ${SINGULARITY_IMG}"
    exit 1
fi

# --- 2. Execute YOLOv9 Training inside Singularity ---
cd "${WROOT}/gyolo"

# python "${WROOT}/gyolo/caption/train.py"
### DP -> failed
DP_SRUN_CMD=(
    singularity exec --nv
    --bind "${WROOT}:${WROOT}"
    --bind "/home/waue0920/dataset/coco:/home/waue0920/dataset/coco"
    "${SINGULARITY_IMG}"
    python "caption/train.py"
    --weights "${INITIAL_WEIGHTS}"
    --data "${WROOT}/${DATA_YAML}"
    --cfg "${MODEL_CFG}"
    --project "${PROJECT_OUT}"
    --name "${NAME_OUT}"
    --device "${DEVICE_LIST}"
)
### DDP
SRUN_CMD=(
    singularity exec --nv
    --bind "${WROOT}:${WROOT}"
    --bind "/home/waue0920/dataset/coco:/home/waue0920/dataset/coco"
    "${SINGULARITY_IMG}"
    torchrun --nproc_per_node="$NGPU" --nnodes="$NNODES" 
    --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    "./caption/train.py"
    --weights "${INITIAL_WEIGHTS}"
    --data "${WROOT}/${DATA_YAML}"
    --cfg "${MODEL_CFG}"
    --project "${PROJECT_OUT}"
    --name "${NAME_OUT}"
    --device "${DEVICE_LIST}"
) # 其他記錄在 ${EXTRA_ARGS}"



if [ -n "${EXTRA_ARGS}" ]; then
    SRUN_CMD+=( ${EXTRA_ARGS} )
fi
echo "!! [client_train] FL exec @ Singularity !!"
echo "${SRUN_CMD[@]}"

"${SRUN_CMD[@]}"
EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Error: Gyolo training failed with exit code ${EXIT_CODE}."
    exit ${EXIT_CODE}
else
    echo "================================================================================"
    echo ">> Client training finished successfully."
    echo "================================================================================"
fi
