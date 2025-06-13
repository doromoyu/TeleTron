import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler,MegatronPretrainingRandomSampler


def build_pretraining_data_loader(dataset, consumed_samples, data_parallel_rank=None, data_parallel_size=None):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()
    if data_parallel_rank is None:
        data_parallel_rank = mpu.get_data_parallel_rank()
    if data_parallel_size is None:
        data_parallel_size = mpu.get_data_parallel_world_size()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size)
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )