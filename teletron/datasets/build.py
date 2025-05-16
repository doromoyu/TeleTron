# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry, build_module
from .fake_dataset import FakeDataset
from .koala_dataset import KoalaDataset

DATASETS = Registry()
DATASETS.register_module(FakeDataset)
DATASETS.register_module(KoalaDataset)


def build_dataset(params_or_type, *args, **kwargs):
    return build_module(DATASETS, params_or_type, *args, **kwargs)