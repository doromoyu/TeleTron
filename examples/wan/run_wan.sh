#!/bin/bash

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENCODER_MODEL_PATH=<Specify path>
ENCODER_TOKENIZER_PATH=<Specify path to file>/google/umt5-xxl
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
echo '$MASTER_ADDR'$MASTER_ADDR
MASTER_PORT='11220'
NNODES=${WORLD_SIZE:-'1'}

echo '$NNODES' $NNODES
NODE_RANK=${RANK:-'0'}
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo '$WORLD_SIZE' $WORLD_SIZE

TP=1
CP=2
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
# GBS=8

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 20
    --hidden-size 5120        
    --num-attention-heads 40
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --model ParallelWanModel # TODO support more models
    --task-type wan_i2v_prone
    --micro-batch-size ${MBS}
    --train-iters 10000
    --weight-decay 1e-2
    --init-method-std 0.006 
    --clip-grad 0.0
    --bf16
    --lr 1e-5
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --activation-offload
    --use-distributed-optimizer
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size 1
    --consumer-models-num 1
)
DATA_ARGS=(
    --dataset-type FakeDataset
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
    --num-frames 9
)

EVAL_AND_LOGGING_ARGS=(
    --encoder-model-path ${ENCODER_MODEL_PATH}
    --encoder-tokenizer-path ${ENCODER_TOKENIZER-PATH}
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000 
    --eval-iters 10000
)


torchrun ${DISTRIBUTED_ARGS[@]} examples/wan/pretrain_wan.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@" 2>&1 | tee wan.log
