# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry, build_module
from .fake_dataset import FakeDataset
from teletron.utils import (
    print_rank_0,
    get_args,
)
from teletron.train.utils import get_train_valid_test_num_samples


DATASETS = Registry()
DATASETS.register_module(FakeDataset)

def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)


def build_train_valid_test_datasets():
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    if args.dataset_type == "FakeDataset":
        train_ds = build_dataset(args.dataset_type)
        valid_ds = None
        test_ds = None
    else:
        raise NotImplementedError

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds