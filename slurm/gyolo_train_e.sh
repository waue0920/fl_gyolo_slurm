#!/bin/bash

### 0_ 參數設定區 ###
HOSTNAME=$(hostname | cut -d '.' -f 1)
echo "[$HOSTNAME][0]==================="
echo " !! Now !! We are in gyolo_train_e.sh @ $HOSTNAME !!"
echo "[$HOSTNAME][0]-------------------"

## 工作目錄
WORKDIR=/home/waue0920/gyolo/src2/
cd $WORKDIR

## 設定 NCCL 詳細紀錄
#export NCCL_DEBUG=TRACE
#export NCCL_DEBUG_SUBSYS=ALL

## SLURM 環境
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
NNODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
if [ -z "$MASTER_ADDR" ]; then
    echo "$HOSTNAME : oh! why MASTER_ADDR not found!"
    MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
fi

#NGPU=$SLURM_GPUS_ON_NODE #這個值常抓不到
#NGPU=$NPROC_PER_NODE # NPROC_PER_NODE是gpu數但在這邊也抓錯
if [ -z "$NGPU" ]; then
    echo "$HOSTNAME : oh! why NPROC_PER_NODE not found!"
    NGPU=$(nvidia-smi -L | wc -l)  # 等於 $SLURM_GPUS_ON_NODE
fi

MASTER_PORT=9527
DEVICE_LIST=$(seq -s, 0 $(($NGPU-1)) | paste -sd, -) # 0,1,...n-1
### 1_ SLURM參數檢查 ###
echo "[$HOSTNAME][1]==================="
echo "[$HOSTNAME]: SLURM_NODEID: $NODE_RANK"
echo "[$HOSTNAME]: SLURM_NNODES: $NNODES"
echo "[$HOSTNAME]: SLURM_GPUS_ON_NODE: $NGPU"
echo "[$HOSTNAME]: Device: $DEVICE_LIST"
echo "[$HOSTNAME]: MASTER_ADDR: $MASTER_ADDR"
echo "[$HOSTNAME]: MASTER_PORT: $MASTER_PORT"
echo "[$HOSTNAME][1]-------------------"

### 客製化 conda env 選項 ### 


### 2_ Python環境檢查區 ### 

echo "[$HOSTNAME][2]==================="
echo "[$HOSTNAME]: Python Path and Version:"
which python
python --version
echo "[$HOSTNAME]: PYTHONPATH: $PYTHONPATH"
echo "[$HOSTNAME]: Activated Conda Environment:"
python -c "import sys; print('\n'.join(sys.path))"
wandb login
python -c 'import wandb'
python -c 'import torch; print(torch.__version__)'
python env.py
echo "[$HOSTNAME][2]-------------------"

### 3_ 執行訓練命令 ###
## 超參數設定
# NBatch=128    # v100 超過 254會failed
NBatch=112    # v100 超過 254會failed
NEpoch=60       # 
NWorker=16       # cpu = gpu x 4, worker < cpu


TRAIN_CMD="torchrun --nproc_per_node=$NGPU --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
caption/train.py --device $DEVICE_LIST --batch $NBatch --epochs $NEpoch --workers $NWorker \
--data data/coco.yaml --img 640 --cfg models/caption/gyolo-e.yaml \
--name gyolo-e-b$NBatch --weights '' --hyp data/hyps/hyp.scratch-cap.yaml \
--optimizer AdamW --flat-cos-lr --no-overlap --close-mosaic 2 --save-period 1 --noplots"

## 印出完整的訓練命令
echo "[$HOSTNAME][3]==================="
echo "[$HOSTNAME]: Executing Training Command:"
echo "[$HOSTNAME]: $TRAIN_CMD"
echo "[$HOSTNAME][3]-------------------"
$TRAIN_CMD


## 檢查執行結果
if [ $? -ne 0 ]; then
  echo "Error: TRAIN_CMD execution failed on node $HOSTNAME" >&2
  exit 1
fi

