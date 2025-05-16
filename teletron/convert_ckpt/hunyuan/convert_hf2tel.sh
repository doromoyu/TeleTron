export PYTHONPATH=$PYTHONPATH:/path/to/Megatron-LM

# Path to the HuggingFace checkpoint and its transformer directory
HUGGINGFACE_CKPT_PATH="/path/to/huggingface/hunyuanvideo"
SOURCE_CKPT_PATH="${HUGGINGFACE_CKPT_PATH}/transformer"

# Output path for the converted Megatron-style checkpoint
TARGET_CKPT_PATH="/path/to/output/teletron_checkpoint/"

# Model parallelism settings
TP=1  # Tensor Parallelism
PP=1  # Pipeline Parallelism

# Run the conversion script
python convert_hunyuanvideo.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --hf-ckpt-path ${HUGGINGFACE_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP}
