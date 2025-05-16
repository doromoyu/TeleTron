# Copyright (c) 2025, TeleAI-infra Team and NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import torch

from megatron.training import (
    get_args,
)
from megatron.core import mpu


def get_batch_on_this_tp_rank_vast(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None

        batch = {
            'images':data["images"].cuda(non_blocking = True),
            'first_ref_image': data["first_ref_image"].cuda(non_blocking = True) if "first_ref_image" in data else None,
            'prompt_embeds': data["prompt_embeds"].cuda(non_blocking = True),
            'clip_text_embed': None if "clip_text_embed" not in data else data["clip_text_embed"].cuda(non_blocking = True),
            'prompt_masks':  None if "prompt_masks" not in data else data["prompt_masks"].cuda(non_blocking = True)
        }

        # Step 1: 保存每部分的大小信息（只在 Rank 0 执行）
        sizes_info = {key: tensor.size() if tensor is not None else None for key, tensor in batch.items()}

        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        _broadcast(batch['images'])
        _broadcast(batch['first_ref_image'])
        _broadcast(batch['prompt_embeds'])
        _broadcast(batch['clip_text_embed'])
        _broadcast(batch['prompt_masks'])

    else:
        sizes_info = None 
        sizes_info_list = [sizes_info]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        images=torch.empty(sizes_info_list[0]['images'], dtype=torch.float32, device = torch.cuda.current_device())
        first_ref_image=torch.empty(sizes_info_list[0]['first_ref_image'], dtype=torch.float32, device = torch.cuda.current_device())
        prompt_embeds=torch.empty(sizes_info_list[0]['prompt_embeds'], dtype=torch.float32, device = torch.cuda.current_device())
        clip_text_embed=torch.empty(sizes_info_list[0]['clip_text_embed'], dtype=torch.bfloat16, device = torch.cuda.current_device())
        prompt_masks=torch.empty(sizes_info_list[0]['prompt_masks'], dtype=torch.int64, device = torch.cuda.current_device())


        _broadcast(images)
        _broadcast(first_ref_image)
        _broadcast(prompt_embeds)
        _broadcast(clip_text_embed)
        _broadcast(prompt_masks)

        batch = {
            'images':images,
            'first_ref_image': first_ref_image,
            'prompt_embeds': prompt_embeds,
            'clip_text_embed': clip_text_embed,
            'prompt_masks': prompt_masks
        }

    return batch


def get_batch_on_this_tp_cp_rank_vast(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
    
    if mpu.get_tensor_context_parallel_rank() == 0:
        if data_iterator is not None:
           data = next(data_iterator)
        else:
           data = None
        
        batch = {
            'images':data["images"].cuda(non_blocking = True),
            'first_ref_image': data["first_ref_image"].cuda(non_blocking = True) if "first_ref_image" in data else None,
            'prompt_embeds': data["prompt_embeds"].cuda(non_blocking = True),
            'clip_text_embed': None if "clip_text_embed" not in data else data["clip_text_embed"].cuda(non_blocking = True),
            'latents': data["latents"].cuda(non_blocking = True),
        }

        # Step 1: 保存每部分的大小信息（只在 Rank 0 执行）
        sizes_info = {key: tensor.size() if tensor is not None else None for key, tensor in batch.items()}

        # Step 2: 广播大小信息
        sizes_info = torch.distributed.broadcast_object_list([sizes_info],mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())
        
        _broadcast(batch['images'])
        _broadcast(batch['first_ref_image'])
        _broadcast(batch['prompt_embeds'])
        _broadcast(batch['clip_text_embed'])
        _broadcast(batch['latents'])

    else:
        sizes_info = None 
        sizes_info_list = [sizes_info]
        torch.distributed.broadcast_object_list(sizes_info_list,mpu.get_tensor_context_parallel_src_rank(), group=mpu.get_tensor_context_parallel_group())

        images=torch.empty(sizes_info_list[0]['images'], dtype=torch.float32, device = torch.cuda.current_device())
        first_ref_image=torch.empty(sizes_info_list[0]['first_ref_image'], dtype=torch.float32, device = torch.cuda.current_device())
        prompt_embeds=torch.empty(sizes_info_list[0]['prompt_embeds'], dtype=torch.float32, device = torch.cuda.current_device())
        clip_text_embed=torch.empty(sizes_info_list[0]['clip_text_embed'], dtype=torch.float32, device = torch.cuda.current_device())
        latents=torch.empty(sizes_info_list[0]['latents'], dtype=torch.bfloat16, device = torch.cuda.current_device())
        

        _broadcast(images)
        _broadcast(first_ref_image)
        _broadcast(prompt_embeds)
        _broadcast(clip_text_embed)
        _broadcast(latents)

        batch = {
            'images':images,
            'first_ref_image': first_ref_image,
            'prompt_embeds': prompt_embeds,
            'clip_text_embed': clip_text_embed,
            'latents':latents
        }

    return batch