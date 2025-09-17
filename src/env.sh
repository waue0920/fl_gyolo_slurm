
#####################
## 全域專案與實驗參數
#####################
export WROOT="/home/waue0920/fl_gyolo_slurm"
export EXPERIMENTS_BASE_DIR="${WROOT}/experiments"

## Dataset
export DATASET_NAME="coco"
## Environment
export SINGULARITY_IMG="${WROOT}/gyolo_ngc2306.sif"

#####################
## Slurm 
#####################
export SLURM_PARTITION=gp2d
export SLURM_ACCOUNT="GOV113038"


#####################
## FL client 端的 slurm 參數
#####################
## FL
export CLIENT_NUM=4   # Client 端數量
export TOTAL_ROUNDS=3  # FL Rounds
export EPOCHS=10
## Gyolo
export BATCH_SIZE=64   # 需要是 gpu 數量的n數: 一般 GPUsx8 高 GPUsx16 
export WORKER=32   # cpu = gpu x 4
export IMG_SIZE=640
export HYP="${WROOT}/gyolo/data/hyps/hyp.scratch-cap-e.yaml" # hyp.scratch-cap.yaml or hyp.scratch-cap-e.yaml


## Parallel
#export CLIENT_NODES=1  # 目前每個Client只支援單節點多GPU運算，因為NCCL port 會衝突
export CLIENT_GPUS=8
export CLIENT_CPUS=32   # cpu = gpu x 4

#####################
## FL Server 端的參數
#####################
export SERVER_ALG="fedprox"   # 支持 fedavg, fedprox, scaffold
export SERVER_FEDPROX_MU=0.01  # FedProx 的 proximal term 係數
