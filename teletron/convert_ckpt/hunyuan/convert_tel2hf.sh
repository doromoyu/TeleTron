export PYTHONPATH=$PYTHONPATH:/path/to/Megatron-LM

# megatron_checkpoint path
SOURCE_CKPT_PATH="/path/to/megatron_checkpoint"
# converted huggingface_checkpoint path
TARGET_CKPT_PATH="/path/to/output/huggingface_checkpoint"

# Model parallelism settings
TP=1  # Tensor Parallelism
PP=1  # Pipeline Parallelism

# Run the conversion script
python convert_hunyuanvideo.py \
--load ${SOURCE_CKPT_PATH} \
--save ${TARGET_CKPT_PATH} \
--target-params-dtype bf16 \
--target-tensor-model-parallel-size ${TP} \
--target-pipeline-model-parallel-size ${PP} \
--convert-checkpoint-from-megatron-to-transformers

