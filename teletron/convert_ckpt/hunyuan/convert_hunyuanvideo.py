# Copyright (c) 2025 TeleAI-infra and Alibaba Pai-Megatron Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import torch
import json
import types
from pathlib import Path
from collections import OrderedDict
from diffusers import AutoencoderKLHunyuanVideo
from transformers import AutoModelForCausalLM, GPT2Config
from diffusers.utils import WEIGHTS_NAME,WEIGHTS_INDEX_NAME
from huggingface_hub import DDUFEntry, create_repo, split_torch_state_dict_into_shards
import numpy as np
from collections.abc import Mapping, Sequence
from diffusers import HunyuanVideoTransformer3DModel

@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device.
    """
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem

def add_extra_args(parser):

    parser.add_argument(
        '--convert-checkpoint-from-megatron-to-transformers',
        action='store_true',
        help=
        ('If True, convert a Megatron checkpoint to a Transformers checkpoint. '
         'If False, convert a Transformers checkpoint to a Megatron checkpoint.'
         ),
    )

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str, required=False
    )

    parser.add_argument(
        "--load-path",
        type=str
    )

    parser.add_argument(
        "--save-path",
        type=str
    )

    parser.add_argument(
        "--target-params-dtype",
        type=str
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


internal_to_output_mapping = {
    "self_attn.dense": "self_attention.linear_proj",
    "mlp.dense_h_to_4h_1": "mlp.linear_fc1_1",
    "mlp.dense_h_to_4h_2": "mlp.linear_fc1_2",
    "mlp.dense_4h_to_h": "mlp.linear_fc2"
}

megatron_to_transformers = {
    "self_attention.linear_proj": "self_attn.o_proj"
}


def get_checkpoint_sub_dir_name(tp_rank, pp_rank, pp_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1: sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    return sub_dir_name

DIT_IDENTICAL_WEIGHT = [
    "x_embedder.proj.weight",
    "x_embedder.proj.bias",
    "context_embedder.time_text_embed.timestep_embedder.linear_1.weight",
    "context_embedder.time_text_embed.timestep_embedder.linear_2.weight",
    "context_embedder.time_text_embed.timestep_embedder.linear_1.bias",
    "context_embedder.time_text_embed.timestep_embedder.linear_2.bias",
    "context_embedder.time_text_embed.text_embedder.linear_1.weight",
    "context_embedder.time_text_embed.text_embedder.linear_2.weight",
    "context_embedder.time_text_embed.text_embedder.linear_1.bias",
    "context_embedder.time_text_embed.text_embedder.linear_2.bias",
    "context_embedder.proj_in.weight",
    "context_embedder.proj_in.bias",
    "context_embedder.token_refiner.refiner_blocks.0.norm1.weight",
    "context_embedder.token_refiner.refiner_blocks.0.norm1.bias",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_q.weight",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_q.bias",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_k.weight",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_k.bias",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_v.weight",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_v.bias",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0.weight",
    "context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0.bias",
    "context_embedder.token_refiner.refiner_blocks.0.norm2.weight",
    "context_embedder.token_refiner.refiner_blocks.0.norm2.bias",
    "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj.weight",
    "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj.bias",
    "context_embedder.token_refiner.refiner_blocks.0.ff.net.2.weight",
    "context_embedder.token_refiner.refiner_blocks.0.ff.net.2.bias",
    "context_embedder.token_refiner.refiner_blocks.0.norm_out.linear.weight",
    "context_embedder.token_refiner.refiner_blocks.0.norm_out.linear.bias",
    "context_embedder.token_refiner.refiner_blocks.1.norm1.weight",
    "context_embedder.token_refiner.refiner_blocks.1.norm1.bias",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_q.weight",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_q.bias",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_k.weight",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_k.bias",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_v.weight",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_v.bias",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_out.0.weight",
    "context_embedder.token_refiner.refiner_blocks.1.attn.to_out.0.bias",
    "context_embedder.token_refiner.refiner_blocks.1.norm2.weight",
    "context_embedder.token_refiner.refiner_blocks.1.norm2.bias",
    "context_embedder.token_refiner.refiner_blocks.1.ff.net.0.proj.weight",
    "context_embedder.token_refiner.refiner_blocks.1.ff.net.0.proj.bias",
    "context_embedder.token_refiner.refiner_blocks.1.ff.net.2.weight",
    "context_embedder.token_refiner.refiner_blocks.1.ff.net.2.bias",
    "context_embedder.token_refiner.refiner_blocks.1.norm_out.linear.weight",
    "context_embedder.token_refiner.refiner_blocks.1.norm_out.linear.bias",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_1.bias",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "time_text_embed.timestep_embedder.linear_2.bias",
    "time_text_embed.guidance_embedder.linear_1.weight",
    "time_text_embed.guidance_embedder.linear_1.bias",
    "time_text_embed.guidance_embedder.linear_2.weight",
    "time_text_embed.guidance_embedder.linear_2.bias",
    "time_text_embed.text_embedder.linear_1.weight",
    "time_text_embed.text_embedder.linear_1.bias",
    "time_text_embed.text_embedder.linear_2.weight",
    "time_text_embed.text_embedder.linear_2.bias",
    "proj_out.weight",
    "proj_out.bias",
]


def get_model_path(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        raise ValueError(f"{model_name_or_path} does not exist")
    model_dir = "./models/"
    model_path = os.path.join(model_dir, model_name_or_path)
    if os.path.exists(model_path):
        return model_path
    return get_huggingface_model_path(model_name_or_path)

def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        # checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_name = [f for f in os.listdir(os.path.join(args.load_path, sub_dir_name)) if 'rng' in f][0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)

    return tp_state_dicts


def megatron_to_transformers_fix_query_key_value_ordering(
        param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
        param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    d = d[path]
    return d


def convert_checkpoint_from_transformers_to_megatron(args):

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path+ "/release", exist_ok=True)
    os.system("cp -rf " + args.load_path + "/*.json " + args.save_path)
    os.system("cp -rf " + args.hf_ckpt_path + "/tokeniz* " + args.save_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)
    
    config = GPT2Config.from_pretrained(args.load_path)

    megatron_args = {
        "attention_head_dim": config.attention_head_dim,
        "in_channels": config.in_channels,
        "num_layers": config.num_layers,
        "num_single_layers": config.num_single_layers,
        "num_refiner_layers": config.num_refiner_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.attention_head_dim * config.num_attention_heads,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size
    }

    # global_config = load_config(globalConfig)

    margs = types.SimpleNamespace()

    for k, v in megatron_args.items():
        setattr(margs, k, v)

    state_dict = HunyuanVideoTransformer3DModel.from_pretrained(args.load_path).state_dict()

    vae_pretrained_path = os.path.join(args.hf_ckpt_path, "vae")
    vae_dict = AutoencoderKLHunyuanVideo.from_pretrained(vae_pretrained_path, trust_remote_code=True).state_dict()
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    # layer_re = re.compile("transformer_blocks\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    hidden_size =  config.attention_head_dim*config.num_attention_heads
    num_heads = config.num_attention_heads
    hidden_size_per_head = config.attention_head_dim
    group_per_split = config.num_attention_heads // args.target_tensor_model_parallel_size

    internal_state_dict = {}

    for layer_id in range(config.num_layers):
        # 1
        q_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_q.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        k_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_k.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        v_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_v.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        q_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_q.bias'].view([num_heads, -1, hidden_size_per_head])
        k_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_k.bias'].view([num_heads, -1, hidden_size_per_head])
        v_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_v.bias'].view([num_heads, -1, hidden_size_per_head])

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=1).view(-1, hidden_size).contiguous()
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1, hidden_size).contiguous()

        # 2
        add_q_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_q_proj.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        add_k_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_k_proj.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        add_v_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_v_proj.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        add_q_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_q_proj.bias'].view([num_heads, -1, hidden_size_per_head])
        add_k_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_k_proj.bias'].view([num_heads, -1, hidden_size_per_head])
        add_v_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_v_proj.bias'].view([num_heads, -1, hidden_size_per_head])

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_qkv.weight'] = torch.cat([add_q_weight, add_k_weight, add_v_weight], dim=1).view(-1, hidden_size).contiguous()
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_qkv.bias'] = torch.cat([add_q_bias, add_k_bias, add_v_bias], dim=1).view(-1, hidden_size).contiguous()

        # 3
        to_out_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_out.0.weight']
        to_out_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_out.0.bias']

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.weight'] = to_out_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.bias'] = to_out_bias

        # 4
        to_add_out_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_add_out.weight']
        to_add_out_bias = state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_add_out.bias']

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.weight'] = to_add_out_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.bias'] = to_add_out_bias

        # 5
        norm_q_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.norm_q.weight']
        norm_k_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.norm_k.weight']

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.q_layernorm.weight'] = norm_q_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.k_layernorm.weight'] = norm_k_weight

        # 6
        norm_added_q_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.norm_added_q.weight']
        norm_added_k_weight = state_dict['transformer_blocks.' + str(layer_id) + '.attn.norm_added_k.weight']

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_q_layernorm.weight'] = norm_added_q_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_k_layernorm.weight'] = norm_added_k_weight

        # 7
        ff_0_proj_weight = state_dict['transformer_blocks.' + str(layer_id) + '.ff.net.0.proj.weight']
        ff_0_proj_bias = state_dict['transformer_blocks.' + str(layer_id) + '.ff.net.0.proj.bias']
        ff_2_weight = state_dict['transformer_blocks.' + str(layer_id) + '.ff.net.2.weight']
        ff_2_bias = state_dict['transformer_blocks.' + str(layer_id) + '.ff.net.2.bias']

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc1.weight'] = ff_0_proj_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc1.bias'] = ff_0_proj_bias
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.weight'] = ff_2_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.bias'] = ff_2_bias

        # 9
        ff_context_0_proj_weight = state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.net.0.proj.weight']
        ff_context_0_proj_bias = state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.net.0.proj.bias']
        ff_context_2_weight = state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.net.2.weight']
        ff_context_2_bias = state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.net.2.bias']

        internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc1.weight'] = ff_context_0_proj_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc1.bias'] = ff_context_0_proj_bias
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.weight'] = ff_context_2_weight
        internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.bias'] = ff_context_2_bias

    for layer_id in range(config.num_single_layers):
        # 1
        q_weight = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_q.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        k_weight = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_k.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        v_weight = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_v.weight'].view([num_heads, -1, hidden_size_per_head, hidden_size])
        q_bias = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_q.bias'].view([num_heads, -1, hidden_size_per_head])
        k_bias = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_k.bias'].view([num_heads, -1, hidden_size_per_head])
        v_bias = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_v.bias'].view([num_heads, -1, hidden_size_per_head])

        internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=1).view(-1, hidden_size).contiguous()
        internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1, hidden_size).contiguous()

        # 2
        norm_q_weight = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.norm_q.weight']
        norm_k_weight = state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.norm_k.weight']

        internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.q_layernorm.weight'] = norm_q_weight
        internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.k_layernorm.weight'] = norm_k_weight

    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append(OrderedDict())


    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32


    for pp_rank in range(args.target_pipeline_model_parallel_size):
    # for layer in range(num_layers):
        for i in range(args.target_tensor_model_parallel_size):
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "model")
            # 更新 DIT_IDENTICAL_WEIGHT 中的参数
            update_params_with_identical_weights(params_dict, state_dict, DIT_IDENTICAL_WEIGHT)

            # 更新 transformer_blocks 中的参数
            for layer_id in range(config.num_layers):
                seg = 6 * hidden_size // args.target_tensor_model_parallel_size
    
                params_dict['transformer_blocks.' + str(layer_id) + '.norm1.linear.weight'] = state_dict['transformer_blocks.' + str(layer_id) + '.norm1.linear.weight'][seg*i: seg*(i+1), :]
                params_dict['transformer_blocks.' + str(layer_id) + '.norm1.linear.bias']=state_dict['transformer_blocks.' + str(layer_id) + '.norm1.linear.bias'][seg*i: seg*(i+1)]
                params_dict['transformer_blocks.' + str(layer_id) + '.norm1_context.linear.weight'] = state_dict['transformer_blocks.' + str(layer_id) + '.norm1_context.linear.weight'][seg*i: seg*(i+1), :]
                params_dict['transformer_blocks.' + str(layer_id) + '.norm1_context.linear.bias']=state_dict['transformer_blocks.' + str(layer_id) + '.norm1_context.linear.bias'][seg*i: seg*(i+1)]

                # 拼接 q_weight, k_weight, v_weight 后进行一次视图重塑
                qkv_weight = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.weight'].view(num_heads, -1, hidden_size_per_head, hidden_size)
                # 对拼接后的 qkv 进行切片操作，按任务编号 i 划分数据
                qkv_weight = qkv_weight[group_per_split * i : group_per_split * (i + 1)]
                # 重塑为二维张量
                qkv_weight = qkv_weight.view(-1, hidden_size).contiguous()
                qkv_bias = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.bias'].view(num_heads, -1, hidden_size_per_head)
                # 对拼接后的 qkv 进行切片操作，按任务编号 i 划分数据
                qkv_bias = qkv_bias[group_per_split * i : group_per_split * (i + 1)]
                # 重塑为二维张量
                qkv_bias = qkv_bias.view(-1).contiguous()
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.weight'] = qkv_weight
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.bias'] = qkv_bias
                # ---------------
                added_qkv_weight = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_qkv.weight'].view(num_heads, -1, hidden_size_per_head, hidden_size)
                # 对拼接后的 qkv 进行切片操作，按任务编号 i 划分数据
                added_qkv_weight = added_qkv_weight[group_per_split * i : group_per_split * (i + 1)]
                # 重塑为二维张量
                added_qkv_weight = added_qkv_weight.view(-1, hidden_size).contiguous()
                added_qkv_bias = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_qkv.bias'].view(num_heads, -1, hidden_size_per_head)
                # 对拼接后的 qkv 进行切片操作，按任务编号 i 划分数据
                added_qkv_bias = added_qkv_bias[group_per_split * i : group_per_split * (i + 1)]
                # 重塑为二维张量
                added_qkv_bias = added_qkv_bias.view(-1).contiguous()
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_qkv.weight'] = added_qkv_weight
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_qkv.bias'] = added_qkv_bias

                seg = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.weight'].shape[1] // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.weight'][:, seg*i : seg*(i + 1)]
                # seg = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.bias'].shape[1] // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.bias'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.linear_proj.bias']
                seg = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.weight'].shape[1] // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.weight'][:, seg*i : seg*(i + 1)]

                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.bias'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_linear_proj.bias']

                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.q_layernorm.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.q_layernorm.weight']
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.k_layernorm.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.k_layernorm.weight']
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_q_layernorm.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_q_layernorm.weight']
                params_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_k_layernorm.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.self_attention.added_k_layernorm.weight']
                ffn_hidden_size = hidden_size * 4
                seg = ffn_hidden_size // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc1.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc1.weight'].view(-1, ffn_hidden_size, hidden_size)[:, seg*i: seg*(i+1), :].reshape(-1, hidden_size)
                params_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc1.bias']=internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc1.bias'][seg*i: seg*(i+1)]
                seg = internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.weight'].shape[1] // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.weight']=internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.weight'][:, seg*i : seg*(i + 1)]
                params_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.bias']=internal_state_dict['transformer_blocks.' + str(layer_id) + '.mlp.linear_fc2.bias']
                
                seg = ffn_hidden_size // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc1.weight']=internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc1.weight'].view(-1, ffn_hidden_size, hidden_size)[:, seg*i: seg*(i+1), :].reshape(-1, hidden_size)
                params_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc1.bias']=internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc1.bias'][seg*i: seg*(i+1)]
                seg = internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.weight'].shape[1] // args.target_tensor_model_parallel_size
                params_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.weight'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.weight'][:, seg*i : seg*(i + 1)]
                params_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.bias'] = internal_state_dict['transformer_blocks.' + str(layer_id) + '.ff_context.linear_fc2.bias']
                
            # 更新 single_transformer_blocks 中的参数
            for layer_id in range(config.num_single_layers):
                seg = 3 * hidden_size // args.target_tensor_model_parallel_size

                params_dict['single_transformer_blocks.' + str(layer_id) + '.norm.linear.weight'] = state_dict['single_transformer_blocks.' + str(layer_id) + '.norm.linear.weight'][seg*i: seg*(i+1), :].reshape(-1, hidden_size)
                params_dict['single_transformer_blocks.' + str(layer_id) + '.norm.linear.bias']=state_dict['single_transformer_blocks.' + str(layer_id) + '.norm.linear.bias'][seg*i: seg*(i+1)]

                qkv_weight = internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.weight'].view(num_heads, -1, hidden_size_per_head, hidden_size)
                # 对拼接后的 qkv 进行切片操作，按任务编号 i 划分数据
                qkv_weight = qkv_weight[group_per_split * i : group_per_split * (i + 1)]
                # 重塑为二维张量
                qkv_weight = qkv_weight.view(-1, hidden_size).contiguous()
                qkv_bias = internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.bias'].view(num_heads, -1, hidden_size_per_head)
                # 对拼接后的 qkv 进行切片操作，按任务编号 i 划分数据
                qkv_bias = qkv_bias[group_per_split * i : group_per_split * (i + 1)]
                # 重塑为二维张量
                qkv_bias = qkv_bias.view(-1).contiguous()
                params_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.weight'] = qkv_weight
                params_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.linear_qkv.bias'] = qkv_bias
                params_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.q_layernorm.weight'] = internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.q_layernorm.weight']
                params_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.k_layernorm.weight'] = internal_state_dict['single_transformer_blocks.' + str(layer_id) + '.self_attention.k_layernorm.weight']
                seg = hidden_size*4 // args.target_tensor_model_parallel_size
                params_dict[f'single_transformer_blocks.{layer_id}.proj_mlp.weight']=state_dict[f'single_transformer_blocks.{layer_id}.proj_mlp.weight'].view(-1, ffn_hidden_size, hidden_size)[:, seg*i: seg*(i+1), :].reshape(-1, hidden_size)
                params_dict[f'single_transformer_blocks.{layer_id}.proj_mlp.bias']=state_dict[f'single_transformer_blocks.{layer_id}.proj_mlp.bias'][seg*i: seg*(i+1)]
                #RowParallel
                seg_1 = hidden_size // args.target_tensor_model_parallel_size
                seg_2 = hidden_size*4 // args.target_tensor_model_parallel_size
                hidden_size_2=hidden_size*4
                
                first_part = state_dict[f'single_transformer_blocks.{layer_id}.proj_out.weight'][:,:hidden_size]
                
                second_part=state_dict[f'single_transformer_blocks.{layer_id}.proj_out.weight'][:,-hidden_size_2:]
               
                params_dict[f'single_transformer_blocks.{layer_id}.proj_out.weight'] = torch.cat(
                    (first_part[:, seg_1*i: seg_1*(i+1)], second_part[:, seg_2*i: seg_2*(i+1)]),
                    dim=1
                )

                params_dict[f'single_transformer_blocks.{layer_id}.proj_out.bias']=state_dict[f'single_transformer_blocks.{layer_id}.proj_out.bias']
                
            params_dict["norm_out.adaLN_modulation.1.weight"] = state_dict['norm_out.linear.weight']
            params_dict["norm_out.adaLN_modulation.1.bias"] = state_dict['norm_out.linear.bias']

            keys = list(params_dict.keys())
            for k in keys:
                params_dict[f"transformer.{k}"] = params_dict.pop(k)
            
            print("begin vae")
            for param_name, param_tensor in vae_dict.items():
                params_dict[f"vae.{param_name}"] = param_tensor

            # 打印参数名称和形状
            with open(f'convert-{i}.log', 'w') as f:
                for param_name, param_tensor in params_dict.items():
                    # 通过 file 参数将输出写入文件
                    print(f"layer_name: {param_name}, param.shape: {param_tensor.shape}", file=f)

    for tp_rank in range(args.target_tensor_model_parallel_size):
        output_state_dict[tp_rank]["checkpoint_version"] = 3.0
        output_state_dict[tp_rank]["args"] = margs
        checkpoint_dir = (
            f"mp_rank_{tp_rank:02d}"
            if args.target_pipeline_model_parallel_size == 1
            else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        )
        checkpoint_name = "model_optim_rng.pt"
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(clone_state_dict(output_state_dict[tp_rank]), checkpoint_path)

def update_params_with_identical_weights(params_dict, state_dict, weight_keys):
    for ori_key in weight_keys:
        if ori_key in state_dict:  # 确保 key 在 state_dict 中存在
            params_dict[ori_key] = state_dict[ori_key]

def update_params_with_identical_weights_for_megatron_to_transformers(params_dict, state_dict, weight_keys):
    for ori_key in weight_keys:
        if ori_key in state_dict:
            new_key_for_transformers=ori_key.replace("transformer.", "", 1)
            params_dict[new_key_for_transformers] = state_dict.pop(ori_key)

def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    os.makedirs(args.save_path, exist_ok=True)
    
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = [i for i in os.listdir(os.path.join(args.load_path, sub_dir)) if 'rng' in i][0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    dir_path=os.path.dirname(args.load_path)
    config = GPT2Config.from_pretrained(dir_path)

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)
    new_DIT_IDENTICAL_WEIGHT = [item for item in DIT_IDENTICAL_WEIGHT]
    path = 'model'
    update_params_with_identical_weights_for_megatron_to_transformers(output_state_dict,get_element_from_dict_by_path(tp_state_dicts[0], path), new_DIT_IDENTICAL_WEIGHT)

    print("Converting transformer layers")
    hidden_size =  config.attention_head_dim * config.num_attention_heads
    num_heads = config.num_attention_heads // tp_size
    hidden_size_per_head = config.attention_head_dim

    path='model'

    # Extract the layers.
    for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
        if key.startswith("transformer_blocks"):
            key_list = key.split('.')
            # layer_id = int(key_list[2])
            layer_id = int(key_list[1])
            if 'norm1' in key:
                params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], path)[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=0,
                    ).to(dtype)
                if 'context' in key:
                    if 'weight' in key:
                        output_state_dict[f'transformer_blocks.{layer_id}.norm1_context.linear.weight'] = params
                    else:
                        output_state_dict[f'transformer_blocks.{layer_id}.norm1_context.linear.bias'] = params
                else:
                    if 'weight' in key:
                        output_state_dict[f'transformer_blocks.{layer_id}.norm1.linear.weight'] = params
                    else:
                        output_state_dict[f'transformer_blocks.{layer_id}.norm1.linear.bias'] = params
            if 'q_layernorm' in key:
                if 'added' in key:
                    output_state_dict[f'transformer_blocks.{layer_id}.attn.norm_added_q.weight']=val
                else:
                    output_state_dict[f'transformer_blocks.{layer_id}.attn.norm_q.weight']=val
            if 'k_layernorm' in key:
                if 'added' in key:
                    output_state_dict[f'transformer_blocks.{layer_id}.attn.norm_added_k.weight']=val
                else:
                    output_state_dict[f'transformer_blocks.{layer_id}.attn.norm_k.weight']=val
            if 'linear_fc' in key:
                dim = 1 if 'linear_fc2' in key else 0
                if "weight" in key:
                    params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=dim,
                    ).to(dtype)
                    if 'linear_fc2' in key:
                        if "ff_context" in key:
                            output_state_dict[
                                f'transformer_blocks.{layer_id}.ff_context.net.2.weight'] = params
                        else:
                            output_state_dict[
                                f'transformer_blocks.{layer_id}.ff.net.2.weight'] = params
                    else:
                        if "ff_context" in key:
                            output_state_dict[
                                f'transformer_blocks.{layer_id}.ff_context.net.0.proj.weight'] = params
                        else:
                            output_state_dict[
                                f'transformer_blocks.{layer_id}.ff.net.0.proj.weight'] = params
                            print(f"ff.net.0.proj shape: {params.shape}")
                if "bias" in key:
                    print(f"bias: {key}")
                    if "linear_fc2" in key:
                        if "ff_context" in key:
                            output_state_dict[f'transformer_blocks.{layer_id}.ff_context.net.2.bias']=val.to(dtype)
                        else:
                            output_state_dict[f'transformer_blocks.{layer_id}.ff.net.2.bias']=val.to(dtype)
                    else:
                        params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=dim,
                    ).to(dtype)
                        if "ff_context" in key:
                            output_state_dict[f'transformer_blocks.{layer_id}.ff_context.net.0.proj.bias']=params
                        else:
                            output_state_dict[f'transformer_blocks.{layer_id}.ff.net.0.proj.bias']=params
                
            if 'linear_qkv' in key:
                # key_list = key.split('.')
                # layer_id = int(key_list[2])
                if 'weight' in key:
                    dim=0
                    # 拆成视图
                    val = val.view(num_heads, -1, hidden_size_per_head, hidden_size)
                    # 当前 rank 的 q, k, v 三个部分
                    groups = list(torch.split(val, 1, dim=1))
                    # 收集来自所有 tp_rank 的对应部分
                    for tp_rank in range(1, tp_size):
                        tp_val = get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        tp_val = tp_val.view(num_heads, -1, hidden_size_per_head, hidden_size)
                        tp_groups = list(torch.split(tp_val, 1, dim=1))
                        for i in range(3):
                            groups[i] = torch.cat([groups[i], tp_groups[i]], dim=dim)
                    print(f"qkv weight shape: {groups[0].shape}")
                    flattened_groups = [
                            group.reshape(-1, hidden_size).to(dtype).contiguous()  # 展平并转换数据类型
                            for group in groups
                        ]
                    if 'added_linear_qkv' in key:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_q_proj.weight']=flattened_groups[0]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_k_proj.weight']=flattened_groups[1]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_v_proj.weight']=flattened_groups[2]
                    else:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_q.weight']=flattened_groups[0]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_k.weight']=flattened_groups[1]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_v.weight']=flattened_groups[2]
                if 'bias' in key:  
                    val = val.view(num_heads, -1, hidden_size_per_head)
                    groups = list(torch.split(val, 1, dim=1))
                    # 收集来自所有 tp_rank 的对应部分
                    for tp_rank in range(1, tp_size):
                        tp_val = get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        tp_val = tp_val.view(num_heads, -1, hidden_size_per_head)
                        tp_groups = list(torch.split(tp_val, 1, dim=1))
                        for i in range(3):
                            groups[i] = torch.cat([groups[i], tp_groups[i]], dim=dim)

                    flattened_groups = [
                            group.reshape(-1).to(dtype).contiguous()  # 展平并转换数据类型
                            for group in groups
                        ]
                
                    if 'added_linear_qkv' in key:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_q_proj.bias']=flattened_groups[0]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_k_proj.bias']=flattened_groups[1]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.add_v_proj.bias']=flattened_groups[2]
                    else:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_q.bias']=flattened_groups[0]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_k.bias']=flattened_groups[1]
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_v.bias']=flattened_groups[2]
                continue
            
            if 'linear_proj' in key:
                dim=1
                if "weight" in key:
                    params = torch.cat(
                            [val]
                            + [
                                get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                                for tp_rank in range(1, tp_size)
                            ],
                            dim=dim,
                        ).to(dtype).contiguous()
                    if 'added_linear_proj' in key:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_add_out.weight']=params
                    else:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_out.0.weight']=params
                if "bias" in key:
                    if 'added_linear_proj' in key:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_add_out.bias']=val
                    else:
                        output_state_dict['transformer_blocks.' + str(layer_id) + '.attn.to_out.0.bias']=val
        # if key.startswith("transformer.single_transformer_blocks"):
        if key.startswith("single_transformer_blocks"):
            key_list = key.split('.')
            # layer_id = int(key_list[2])
            layer_id = int(key_list[1])
            if 'norm.linear' in key:
                params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], path)[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=0,
                    ).to(dtype)
                if 'weight' in key:
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.norm.linear.weight'] = params
                if 'bias' in key:
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.norm.linear.bias'] = params
            if 'linear_qkv' in key:
                if 'weight' in key:
                    dim=0
                    # 拆成视图
                    val = val.view(num_heads, -1, hidden_size_per_head, hidden_size)
                    # 当前 rank 的 q, k, v 三个部分
                    groups = list(torch.split(val, 1, dim=1))
                    # 收集来自所有 tp_rank 的对应部分
                    for tp_rank in range(1, tp_size):
                        tp_val = get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        tp_val = tp_val.view(num_heads, -1, hidden_size_per_head, hidden_size)
                        tp_groups = list(torch.split(tp_val, 1, dim=1))
                        for i in range(3):
                            groups[i] = torch.cat([groups[i], tp_groups[i]], dim=dim)

                    flattened_groups = [
                            group.reshape(-1, hidden_size).to(dtype).contiguous()  # 展平并转换数据类型
                            for group in groups
                        ]
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_q.weight']=flattened_groups[0]
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_k.weight']=flattened_groups[1]
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_v.weight']=flattened_groups[2]
                if 'bias' in key:  
                    val = val.view(num_heads, -1, hidden_size_per_head)
                    groups = list(torch.split(val, 1, dim=1))
                    # 收集来自所有 tp_rank 的对应部分
                    for tp_rank in range(1, tp_size):
                        tp_val = get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        tp_val = tp_val.view(num_heads, -1, hidden_size_per_head)
                        tp_groups = list(torch.split(tp_val, 1, dim=1))
                        for i in range(3):
                            groups[i] = torch.cat([groups[i], tp_groups[i]], dim=dim)

                    flattened_groups = [
                            group.reshape(-1).to(dtype).contiguous()  # 展平并转换数据类型
                            for group in groups
                        ]         

                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_q.bias']=flattened_groups[0]
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_k.bias']=flattened_groups[1]
                    output_state_dict['single_transformer_blocks.' + str(layer_id) + '.attn.to_v.bias']=flattened_groups[2]
                continue
            if 'proj' in key:
                dim=0
                if "weight" in key:
                    if 'proj_mlp' in key:
                        params = torch.cat(
                            [val]
                            + [
                                get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                                for tp_rank in range(1, tp_size)
                            ],
                            dim=dim,
                        ).to(dtype).contiguous()
                        output_state_dict['single_transformer_blocks.' + str(layer_id) + '.proj_mlp.weight']=params
                    else:
                        seg_1 = hidden_size // args.target_tensor_model_parallel_size
                        seg_2 = hidden_size*4 // args.target_tensor_model_parallel_size
                        params_first = torch.cat(
                                [val[:,:seg_1]]
                                + [
                                    get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key][:,:seg_1]
                                    for tp_rank in range(1, tp_size)
                                ],
                                dim=1,
                            ).to(dtype).contiguous()
                        params_second = torch.cat(
                                [val[:,-seg_2:]]
                                + [
                                    get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key][:,-seg_2:]
                                    for tp_rank in range(1, tp_size)
                                ],
                                dim=1,
                            ).to(dtype).contiguous()
                        output_state_dict['single_transformer_blocks.' + str(layer_id) + '.proj_out.weight']=torch.cat((params_first,params_second),dim=1).contiguous()
                if "bias" in key:
                    params = torch.cat(
                            [val]
                            + [
                                get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                                for tp_rank in range(1, tp_size)
                            ],
                            dim=dim,
                        ).to(dtype).contiguous()
                    if 'proj_mlp' in key:
                        output_state_dict['single_transformer_blocks.' + str(layer_id) + '.proj_mlp.bias']=params
                    else:
                        output_state_dict['single_transformer_blocks.' + str(layer_id) + '.proj_out.bias']=val
                continue
            if 'q_layernorm' in key:
                output_state_dict[f'single_transformer_blocks.{layer_id}.attn.norm_q.weight']=val
            if 'k_layernorm' in key:
                output_state_dict[f'single_transformer_blocks.{layer_id}.attn.norm_k.weight']=val
            continue

    output_state_dict['norm_out.linear.weight']=get_element_from_dict_by_path(tp_state_dicts[0], path)["transformer.norm_out.adaLN_modulation.1.weight"]
    output_state_dict['norm_out.linear.bias']=get_element_from_dict_by_path(tp_state_dicts[0], path)["transformer.norm_out.adaLN_modulation.1.bias"]
    
    log_file = open("output_state_dict_kv.txt", "w")
    for key, val in output_state_dict.items():
        log_file.write(f"{key}: shape={tuple(val.shape)}, dtype={val.dtype}\n")
    log_file.close()

    filename_pattern='diffusion_pytorch_model{suffix}.bin'
    state_dict_split = split_torch_state_dict_into_shards(
            output_state_dict,max_shard_size="42GB", filename_pattern=filename_pattern
            )
    # Save the model
    if not os.path.exists(args.save_path):
        os.system(f'mkdir -p {args.save_path}')
    # 保存每个shard
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: output_state_dict[tensor].contiguous() for tensor in tensors}
        filepath = os.path.join(args.save_path, filename)
        torch.save(shard, filepath)

    # 保存index文件
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(args.save_path, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))

        print(f"Sharded model saved successfully with index file at {args.save_path}")
    else:
        print(f"Model small enough, saved without sharding in {args.save_path}/{WEIGHTS_NAME}")

    config_path = '/'.join(args.load_path.split('/')[:-1])
    os.system("cp -rf "+config_path+"/config.json " + args.save_path)
    print("Conversion from Megatron-LM to Transformers is done!")

def main():
    parser = argparse.ArgumentParser()
    parser = add_extra_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
