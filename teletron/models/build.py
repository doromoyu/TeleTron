# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

from .registry import Registry
from .wan.parallel_wan_model import ParallelWanModel
from .hunyuan.parallel_model import ParallelHunyuanVideoModel


registor = Registry("model")
registor.register(ParallelWanModel)
registor.register(ParallelHunyuanVideoModel)

def build_model(name,config=None):
    if config is None:
        return registor.build(name)
    else:
        return registor.build(name,config)


