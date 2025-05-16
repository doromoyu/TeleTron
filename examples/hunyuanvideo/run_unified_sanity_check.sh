#!/bin/bash

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_VISIBLE_DEVICES=2
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:$(pwd)/third_parties/Megatron-LM

GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
echo '$MASTER_ADDR'$MASTER_ADDR
# MASTER_PORT='12341'
NNODES=${WORLD_SIZE:-'1'}

echo '$NNODES' $NNODES
NODE_RANK=${RANK:-'0'}
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo '$WORLD_SIZE' $WORLD_SIZE

TP=$1
CP=$2
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
export NUM_LAYERS=3
export NUM_SINGLE_LAYERS=6
# 开启融合算子计算，需要先安装fused kernels
export FUSED_KERNELS=1

TENSORBOARD_LOGS_PATH=./logs
DATA_PATH=./checkpoint


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 3
    --hidden-size 3072        
    --num-attention-heads 24
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters 10
    --weight-decay 1e-2
    --init-method-std 0.006 
    --clip-grad 0.0
    --bf16
    --lr 1e-5 
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --use-distributed-optimizer
    --recompute-num-layers 42
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --sanity-check
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
)
DATA_ARGS=(
    --dataset-type FakeDataset
    --data-path $DATA_PATH 
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000 
    --eval-iters 10000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

rm test/test_data/tp${TP}cp${CP}_layer36.log
echo start tp${TP} cp${CP} training
torchrun ${DISTRIBUTED_ARGS[@]} examples/hunyuanvideo/pretrain_hunyuanvideo.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]}
