# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import json
import torch
import teletron
from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu

from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0,get_model
from teletron.training.training import pretrain,initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.training.utils import (
    average_losses_across_data_parallel_group
)
import torch.distributed as dist
from megatron.training.global_vars import (
    get_args,
    get_timers,
)
from teletron.datasets.fake_dataset import FakeDataset
from teletron.models.hunyuanvideo.pipeline import HunyuanPipeline
from teletron.training.utils import get_batch_on_this_tp_cp_rank_vast

from teletron.datasets.build import build_dataset
from teletron.models.hunyuanvideo.producer import producer_process
from teletron.core.parallel_state import get_world_group
import yaml

class Config(dict):
    def __init__(self, d=None):
        if d is None:
            d = {}
        super().__init__(d)
        for k, v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def get_batch(data_iterator):
    # get batches based on the TP_CP rank you are on
    batch = get_batch_on_this_tp_cp_rank_vast(data_iterator)
    return batch


def extra_args_provider(parser):
    group = parser.add_argument_group(title='dataset')
    group.add_argument('--dataset-type', default="KoalaDataset")
    group.add_argument("--num-frames", type=int, default=9,
                       help='number of frames to train, must be of 4n+1, \
                        overloads yaml if using koala dataset. example: 45')
    group.add_argument("--video-resolution", nargs=2, type=int, default=[1280, 720], 
                       help='video resolution to train, overloads yaml if using koala dataset. \
                       width and height should satisfy: (width or height) // 8 % 2 == 0')
    group.add_argument("--koala-opt", type=str, default="./teletron/datasets/koala.yml", 
                        help="the koala dataset option file")


    group = parser.add_argument_group(title="diffusion")
    group.add_argument("--vae-slicing", action="store_false")
    group.add_argument("--vae-tiling", action="store_false")
    group.add_argument("--flow-resolution-shifting", action="store_true")
    group.add_argument("--flow-base-image-seq-len", type=int, default=256)
    group.add_argument("--flow-max-image-seg-len", type=int, default=4096)
    group.add_argument("--flow-base-shift", type=float, default=0.5)
    group.add_argument("--flow-max-shift", type=float, default=1.15)
    group.add_argument("--flow-shift", type=float, default=1.0)
    group.add_argument("--flow-weighting-scheme", type=str, default="none")
    group.add_argument("--flow-logit-mean", type=float, default=0.0)
    group.add_argument("--flow-logit-std", type=float, default=1.0)
    group.add_argument("--flow-mode-scale", type=float, default=1.29)
    
    group = parser.add_argument_group(title='debug')
    group.add_argument("--sanity-check", action="store_true")

    group.add_argument("--distributed-vae-world-size", type=int, default=0,required=False)
    group.add_argument("--tokenizer", type=str, required=False)
    group.add_argument("--tokenizer-mode", type=str, default="llama")
    return parser


def train_valid_test_datasets_provider(train_val_test_num_samples):

    args = get_args()

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    train_ds = build_dataset(args.dataset_type)
    valid_ds = None
    test_ds = None

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds

def init(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
):
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks
    )

    args = get_args()
    

def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> HunyuanPipeline:
    args = get_args()

    config = core_transformer_config_from_args(args)

    model = HunyuanPipeline(
        config=config
    )
    
    return model

def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}

def forward_step(data_iterator, model: HunyuanPipeline):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    batch = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor_list = model(batch)

    return output_tensor_list, loss_func

if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        producer_process=producer_process,
        extra_args_provider=extra_args_provider,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )