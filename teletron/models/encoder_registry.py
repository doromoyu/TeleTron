# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import torch
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Tuple, List
from teletron.utils import get_args

class BaseEncoder(ABC):

    def __init__(self, device: torch.device, **kwargs: Any):
        
        args = get_args()
        self.device = device
        self.moe = (args.consumer_models_num > 1)

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """
        init models
        """
        pass

    @abstractmethod
    def encode(self, raw_batch: Dict[str, Any]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """

        Args:
            raw_batch (Dict[str, Any])

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
            - Tensor List (List[torch.Tensor])
            - Sizes of Tensors (torch.Tensor)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_output_schema() -> List[str]:
        """
        返回一个有序列表，其中包含编码器输出的所有张量的名称。
        这个顺序必须与 encode() 方法返回的张量列表的顺序严格一致。
        这是一个静态方法，因此无需实例化即可调用。

        返回:
            List[str]: 一个包含张量名称的字符串列表，例如 ['context', 'clip,'image_feature', 'latents']。
        """
        pass

    @staticmethod
    def _pack_tensors(tensors_to_pack: List[torch.Tensor]) -> torch.Tensor:
        """
        将一个张量列表展平并拼接成一个单一的扁平化张量。
        这是一个辅助函数，可以在具体实现中被调用。
        """
        if not tensors_to_pack:
            return torch.tensor([])
        
        flattened_tensors = [torch.flatten(t) for t in tensors_to_pack]
        return torch.cat(flattened_tensors, dim=0)

    @staticmethod
    def _get_tensors_size(tensor_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        获取张量列表的形状信息，并将其作为一个整数张量返回。
        这是一个辅助函数，可以在具体实现中被调用。
        """
        size_info = ()
        for item in tensor_list:
            size_info += item.size()
        return torch.tensor(size_info, dtype=torch.int32, device=device)

_ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {}

def register_encoder(name: str):
    """
    一个装饰器，用于将编码器类注册到全局注册表中。

    Args:
        name (str): 编码器的唯一名称。
    """
    def decorator(cls: Type[BaseEncoder]):
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"错误: 编码器 '{name}' 已被注册。")
        if not issubclass(cls, BaseEncoder):
            raise TypeError(f"错误: 注册的类 '{cls.__name__}' 必须是 BaseEncoder 的子类。")
        
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator

def get_encoder(name: str, device: torch.device, **kwargs: Any) -> BaseEncoder:
    """
    根据名称从注册表中获取并实例化一个编码器。

    Args:
        name (str): 要获取的编码器的名称。
        device (torch.device): 编码器将被初始化的设备。
        **kwargs: 传递给编码器构造函数的其他参数。

    Returns:
        BaseEncoder: 所请求编码器的实例化对象。
    
    Raises:
        ValueError: 如果请求的名称在注册表中不存在。
    """
    if name not in _ENCODER_REGISTRY:
        raise ValueError(f"错误: 编码器 '{name}' 未在注册表中找到。可用选项: {list(_ENCODER_REGISTRY.keys())}")
    
    encoder_class = _ENCODER_REGISTRY[name]
    return encoder_class(device=device, **kwargs)

model_mapping = {
    "parallelwanmodel": "wan_encoder",
    "wanmodel": "wan_encoder"
}

def get_encoder_name(key):
    return model_mapping.get(key.lower(), "unknown_encoder")
    