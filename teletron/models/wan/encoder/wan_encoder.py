# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import os
import torch
from typing import Dict, Any, Tuple, List

from teletron.core.distributed.base_encoder import BaseEncoder
from .wan_prompter import WanPrompter
from .model_manager import ModelManager
from .wan_encoder_utils import get_encoder_features
from teletron.models.wan.pipelines.wan_video import WanVideoPipeline
from teletron.utils import get_args

def get_encoder_model_paths(path):
    filenames = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ]
    return [os.path.join(path, f) for f in filenames]

class WanVideoEncoder(BaseEncoder):
    """WAN视频模型的具体编码器实现。"""
    
    _OUTPUT_MOE_SCHEMA = ['context', 'img_clip_feature', 'img_emb_y', 'latents', 'noise']
    _OUTPUT_SCHEMA = ['context', 'img_clip_feature', 'img_emb_y', 'latents']

    @staticmethod
    def get_output_schema() -> List[str]:
        """返回此编码器输出张量的固定名称和顺序。"""
        args = get_args()
        is_moe = (args.consumer_models_num > 1)
        if is_moe is True:
            return WanVideoEncoder._OUTPUT_MOE_SCHEMA
        return WanVideoEncoder._OUTPUT_SCHEMA

    def __init__(self, device: torch.device, **kwargs: Any):
        super().__init__(device)
        args = get_args()
        kwargs['model_paths'] = get_encoder_model_paths(args.encoder_model_path)
        kwargs['tokenizer_path'] = args.encoder_tokenizer_path
        kwargs['tiler_kwargs'] = {
            "tiled": True, 
            "tile_size":  (34, 34), 
            "tile_stride": (18, 16)
        }
        self.model_paths = kwargs.get("model_paths")
        self.tokenizer_path = kwargs.get("tokenizer_path")
        self.tiler_kwargs = kwargs.get("tiler_kwargs", {})

        if not self.model_paths or not self.tokenizer_path:
            raise ValueError("WanVideoEncoder需要 'model_paths' 和 'tokenizer_path' 参数。")

        # 将模型组件初始化为None，它们将在setup()中被加载
        self.text_encoder = None
        self.image_encoder = None
        self.vae = None
        self.prompter = None

    def setup(self) -> None:
        """加载所有必需的WAN模型组件到指定设备。"""
        print(f"在设备 {self.device} 上设置 WanVideoEncoder...")
        
        model_manager = ModelManager(torch_dtype=torch.float32, device="cpu")
        model_manager.load_models(self.model_paths)
        
        pipe = WanVideoPipeline.from_model_manager(model_manager)
        
        self.text_encoder = pipe.text_encoder.to(device=self.device)
        self.image_encoder = pipe.image_encoder.to(device=self.device)
        self.vae = pipe.vae.to(device=self.device, dtype=torch.bfloat16)
        del pipe # 释放不再需要的内存

        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.tokenizer_path)
        print("WanVideoEncoder 设置完成。")


    def encode(self, raw_batch: Dict[str, Any]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        使用WAN模型对数据批次进行编码。
        """
        batch = dict(raw_batch)

        prompt_emb, image_emb, latents = get_encoder_features(
            batch, self.prompter, self.vae, self.tiler_kwargs, self.image_encoder
        )
        
        
        context = prompt_emb['context']
        img_clip_feature = image_emb["clip_feature"]
        img_emb_y = image_emb["y"]

        if self.moe is True:
            noise = torch.randn_like(latents, device=self.device)
            tensors_to_send = [context, img_clip_feature, img_emb_y, latents, noise]
        else:
            tensors_to_send = [context, img_clip_feature, img_emb_y, latents]

        size_info_tensor = self._get_tensors_size(tensors_to_send, device=self.device)

        return tensors_to_send, size_info_tensor