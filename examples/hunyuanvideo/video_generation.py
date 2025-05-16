# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
import torch
import argparse
from pipelines import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image
from typing import List
from torchvision.transforms import InterpolationMode, functional as F


def prepare_reference_images(width: int, height: int, num_frames: int, ref_image_path: str) -> List[Image.Image]:
    """Prepare reference images with only the first frame set from input image"""
    ref_images = [Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(num_frames)]
    original_img = Image.open(ref_image_path)
    resized_img = F.resize(original_img, (height, width), InterpolationMode.BILINEAR)
    ref_images[0] = resized_img
    return ref_images


def main():
    parser = argparse.ArgumentParser(description="Hunyuan Video Generation Script")
    parser.add_argument("--width", type=int, default=1280, help="Width of the output video")
    parser.add_argument("--height", type=int, default=720, help="Height of the output video")
    parser.add_argument("--num_frames", type=int, default=49, help="Total number of frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--transformer_model_path", type=str, required=True, help="Path to transformer model")
    parser.add_argument("--ref_frame", type=str, default="", help="Path to reference image for first frame")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--embedded_guidance_scale", type=float, default=1.0, help="Embedded guidance scale")

    args = parser.parse_args()

    ref_images = prepare_reference_images(
        args.width, 
        args.height,
        args.num_frames,
        args.ref_frame
    )

    device = torch.device(args.device)
    pipeline = HunyuanVideoPipeline.from_pretrained(
        args.base_model_path,
        transformer_model_path=args.transformer_model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipeline.vae.enable_tiling()

    video = pipeline(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        model_type='i2v',
        ref_images=ref_images,
        guidance_scale=args.guidance_scale,
        embedded_guidance_scale=args.embedded_guidance_scale,
    ).frames[0]

    export_to_video(video, args.output_file, fps=15)


if __name__ == "__main__":
    main()