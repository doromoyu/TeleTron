# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry
from .wan.parallel_wan_model import ParallelWanModel



registor = Registry("model")
registor.register(ParallelWanModel)


def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


