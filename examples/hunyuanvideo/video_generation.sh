#!/bin/bash

# Hunyuan Video Generation Script
# Modify parameters below and run directly: ./video_generation.sh

python examples/hunyuanvideo/video_generation.py \
    --base_model_path "path/to/hunyuan_ckpt/hunyuanvideo_13b" \
    --transformer_model_path "path/to/hunyuan_ckpt/transformer" \
    --prompt "A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door" \
    --ref_frame "assets/oven.jpg" \
    --width 1280 \
    --height 720 \
    --num_frames 49 \
    --num_inference_steps 50 \
    --output_file "oven.mp4" \
    --device "cuda:0" \
    --seed 42 \
    --guidance_scale 4.0 \
    --embedded_guidance_scale 1.0