# Copyright 2025 TeleAI-infra Team and HuggingFace Inc. All rights reserved.

import os
from typing import List, Optional, Union, Any, Mapping

import functools
from einops import rearrange
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from tqdm import tqdm
import logging
# from megatron.core.export.data_type import DataType
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from megatron.core import mpu

from teletron.models.hunyuanvideo.model import HunyuanVideoTransformer3DModel, HunyuanParams

from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler

from megatron.training import get_args
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)

def broadcast_timesteps(input: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

class HunyuanPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_process = mpu.is_pipeline_first_stage()
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None
        print("Initializing HunyuanPipeline...")
        self.vae = AutoencoderKLHunyuanVideo()
        self.vae.requires_grad_(False)
        args = get_args()
        if args.vae_slicing:
            self.vae.enable_slicing()
        if args.vae_tiling:
            self.vae.enable_tiling()

        hunyuanConfig = HunyuanParams()
        latent_channels = self.vae.config.latent_channels

        print("Load HunyuanVideoTransformer3DModel to cuda")
        self.config = config 
        self.transformer = HunyuanVideoTransformer3DModel(hunyuanConfig, config)
        print("Loaded HunyuanVideoTransformer3DModel to cuda")
        self.flow_scheduler = FlowMatchEulerDiscreteScheduler()

        self.dtype = torch.bfloat16
        print(f"hunyuanpipeline dtype: {self.dtype}")
        
    def forward(self, batch_dict):
        args = get_args()
        if self.pre_process:
            with torch.no_grad():
                drop_prob = 0
                # latents
                images = batch_dict["images"]
                batch_size, num_frames, _, height, width = images.shape
                
                timesteps = torch.tensor(0, device=torch.cuda.current_device()).long()
                
                # add noise from flow matching scheduler
                scheduler_sigmas = self.flow_scheduler.sigmas.clone()
                weights = compute_density_for_timestep_sampling(
                    weighting_scheme=args.flow_weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.flow_logit_mean,
                    logit_std=args.flow_logit_std,
                    mode_scale=args.flow_mode_scale,
                )
                indices = (weights * self.flow_scheduler.config.num_train_timesteps).long()
                sigmas = scheduler_sigmas[indices].to(device=torch.cuda.current_device())

                timesteps = (sigmas * 1000.0).long()
                broadcast_timesteps(timesteps)


                prompt_embeds = batch_dict["prompt_embeds"].to(dtype=self.dtype)
                pooled_prompt_embeds = batch_dict["clip_text_embed"].to(dtype=self.dtype)
                prompt_masks = (
                    batch_dict["prompt_masks"].to(dtype=self.dtype)
                    if "prompt_masks" in batch_dict
                    else None
                )
                
                latents = batch_dict["latents"]

                noise = torch.randn(
                    latents.shape,
                    device=torch.cuda.current_device(),
                )

                def expand_tensor_to_dims(tensor, ndim):
                    while len(tensor.shape) < ndim:
                        tensor = tensor.unsqueeze(-1)
                    return tensor

                sigmas = expand_tensor_to_dims(sigmas, ndim=latents.ndim)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

                # conditional_latents
                conditional_latents = None
                if "ref_images" in batch_dict:
                    ref_images = batch_dict["ref_images"]
                    conditional_latents = (
                        self.forward_vae(ref_images) * self.vae.config.scaling_factor
                    )
                    if drop_prob > 0:
                        random_p = torch.rand(batch_size, device=self.device)
                        image_mask = torch.logical_and(
                            random_p >= drop_prob, random_p < 3 * drop_prob
                        )
                        image_mask = 1 - image_mask.float()
                        image_mask = image_mask.to(conditional_latents.dtype)
                        conditional_latents = (
                            conditional_latents * image_mask[:, None, None, None, None]
                        )
                    noisy_model_input = torch.cat(
                        [noisy_model_input, conditional_latents], dim=1
                    )
                if "first_ref_image" in batch_dict and batch_dict["first_ref_image"] is not None:
                    first_ref_image = batch_dict["first_ref_image"]
                    conditional_latents = (
                        self.forward_vae(first_ref_image) * self.vae.config.scaling_factor
                    )
                    pad_size = noisy_model_input.size(2) - conditional_latents.size(2)
                    conditional_latents = F.pad(
                        conditional_latents,
                        (0, 0, 0, 0, 0, pad_size),
                        mode="constant",
                        value=0,
                    )
                    b, c, f, h, w = noisy_model_input.shape
                    mask = torch.zeros((b, 1, f, h, w), device=torch.cuda.current_device())
                    mask[:, :, 0] = 1
                    noisy_model_input = torch.cat(
                        [noisy_model_input, conditional_latents, mask], dim=1
                    )
                if "cn_images" in batch_dict:
                    cn_images = batch_dict["cn_images"].to(self.dtype)
                    cn_images = self.forward_vae(cn_images)
                # TODO guidance check
                guidance_scale = 1.0
                guidance = (
                    torch.tensor(
                        [guidance_scale] * latents.shape[0],
                        dtype=self.dtype,
                        device='cuda',
                    )
                    * 1000.0
                )

                # cn model
                if "cn_images" in batch_dict:
                    if hasattr(self.model, "guider"):
                        cn_model = functools.partial(self.model, "guider")
                        cn_images = rearrange(cn_images, "b t c h w -> b c t h w")
                        cn_latents = cn_model(cn_images)
                        # cn_latents = rearrange(cn_latents, "b c t h w -> b t c h w")
                    else:
                        cn_latents = cn_images
                    noisy_model_input = torch.cat([noisy_model_input, cn_latents], dim=1)
                latent_model_input = noisy_model_input.to(torch.bfloat16)

        # generate dummy input for sanity check
        if args.sanity_check:
            torch.manual_seed(1234)
            latent_model_input = torch.randn_like(latent_model_input)
            prompt_embeds = torch.randn_like(prompt_embeds)
            timesteps = torch.randint_like(timesteps, 0, 100)
            pooled_prompt_embeds = torch.randn_like(pooled_prompt_embeds)
            guidance = torch.randn_like(guidance)

        model_pred = self.transformer(
            hidden_states=latent_model_input,  # [1, 2, 16, 28, 48] -> [1, 16, 2, 28, 48]
            timestep=timesteps,  # [263]
            encoder_hidden_states=prompt_embeds,  # [1, 226, 4096]
            encoder_attention_mask=prompt_masks,  # [[1, 1, 1, 0, 0 ]]
            pooled_projections=pooled_prompt_embeds,  # [1, 1, 768]
            guidance=guidance,  # []
            return_dict=False,
        )[0]

        # loss
        if self.post_process:
            weights = compute_loss_weighting_for_sd3(
                weighting_scheme=args.flow_weighting_scheme,
                sigmas=sigmas,
            )
            target = noise - latents

            loss = weights.float() * (model_pred.float() - target.float()).pow(2)
            if args.sanity_check:
                loss = model_pred.norm() # dummy loss just for sanity check
            
        return [loss]
    
    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images).latent_dist.sample()
        return latents

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.transformer.state_dict(prefix=prefix, keep_vars=keep_vars)
    
    def set_input_tensor(self, input_tensor):
        pass