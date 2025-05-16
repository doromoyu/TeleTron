# Copyright 2025 TeleAI-infra Team, The HunyuanVideo Team, and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from dataclasses import dataclass
from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import HunyuanVideoLoraLoaderMixin
from diffusers.models import AutoencoderKLHunyuanVideo
from teletron.models.hunyuanvideo.model import HunyuanVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import os
from einops import rearrange
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        >>> from diffusers.utils import export_to_video

        >>> model_id = "tencent/HunyuanVideo"
        >>> transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        ...     model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
        >>> pipe.vae.enable_tiling()
        >>> pipe.to("cuda")

        >>> output = pipe(
        ...     prompt="A cat walks on the grass, realistic",
        ...     height=320,
        ...     width=512,
        ...     num_frames=61,
        ...     num_inference_steps=30,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=15)
        ```
"""


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}

@torch.cuda.amp.autocast(dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    r"""
    Output class for HunyuanVideo pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


class HunyuanVideoPipeline(DiffusionPipeline, HunyuanVideoLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`LlamaModel`]):
            [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
        tokenizer (`LlamaTokenizer`):
            Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
        transformer ([`HunyuanVideoTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLHunyuanVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder_2 ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer_2 (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        text_encoder: LlamaModel,
        tokenizer: LlamaTokenizerFast,
        transformer: HunyuanVideoTransformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio
            if hasattr(self, "vae") and self.vae is not None
            else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.normalize = transforms.Normalize([0.5], [0.5])

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        transformer_model_path=None,
        guider_model_path=None,
        torch_dtype=None,
        **kwargs,
    ):
        if transformer_model_path is None:
            transformer_model_path = os.path.join(model_path, "transformer")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            transformer_model_path, torch_dtype=torch_dtype, disable_mmap=True, **kwargs,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        guider = None
        if guider_model_path is not None:
            guider = GuiderModel.from_pretrained(
                guider_model_path, torch_dtype=torch_dtype
            )
        pipe = super().from_pretrained(
            model_path,
            transformer=transformer,
            scheduler=scheduler,
            guider=guider,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        return pipe

    def _get_llama_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
        padding: Union[bool, str] = "max_length",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding=padding,
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|eot_id|> token and placeholder {}
            crop_start -= 2

        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=padding not in [False, 'do_not_pad'],
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        if padding in [False, 'do_not_pad']:
            prompt_attention_mask = None
        else:
            prompt_attention_mask = text_inputs.attention_mask.to(device=device)
        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        if prompt_attention_mask is not None:
            if crop_start is not None and crop_start > 0:
                prompt_attention_mask = prompt_attention_mask[:, crop_start:]

            prompt_attention_mask = prompt_attention_mask.repeat(1, num_videos_per_prompt)
            prompt_attention_mask = prompt_attention_mask.view(
                batch_size * num_videos_per_prompt, seq_len
            )

        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_2.batch_decode(
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        ).pooler_output

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        padding: Union[bool, str] = "max_length",
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
                padding=padding,
            )

        if pooled_prompt_embeds is None:
            if prompt_2 is None and pooled_prompt_embeds is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_template=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif prompt_2 is not None and (
            not isinstance(prompt_2, str) and not isinstance(prompt_2, list)
        ):
            raise ValueError(
                f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}"
            )

        if prompt_template is not None:
            if not isinstance(prompt_template, dict):
                raise ValueError(
                    f"`prompt_template` has to be of type `dict` but is {type(prompt_template)}"
                )
            if "template" not in prompt_template:
                raise ValueError(
                    f"`prompt_template` has to contain a key `template` but only found {prompt_template.keys()}"
                )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negtive_prompt: str = "",
        negtive_prompt_2: str = "",
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negtive_prompt_embeds: Optional[torch.Tensor] = None,
        negtive_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negtive_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        embedded_guidance_scale: Optional[float] = 1.0,
        max_sequence_length: int = 256,
        seed: Optional[int] = -1,
        model_type: str = "t2v",
        ref_images: List[Image.Image] = None,
        cn_images: List[Image.Image] = None,
        first_last: bool = False,
        cfg_zero_star: bool = False,
        padding: Union[bool, str] = "max_length",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            height (`int`, defaults to `720`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `129`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, defaults to `6.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. Note that the only available HunyuanVideo model is
                CFG-distilled, which means that traditional guidance between unconditional and conditional latent is
                not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            seed (`int`, *optional*, defaults to -1):
                Seed for the random number generator.
            model_type (`str`, defaults to 't2v'):
                Type of model to use. Options: 't2v' (text-to-video), 'i2v' (image-to-video),
                's2v' (storyboard-to-video), 'i2vhy' (image-to-video).
            ref_images (`List[Image.Image]`, *optional*):
                Reference images for 'i2v' or 's2v' modes.
            cn_images (`List[Image.Image]`, *optional*):
                Content reference images for tasks like storyboard-to-video.
            first_last (`bool`, defaults to `False`):
                Whether to enable the first and last frame reference mode.
            padding (`Union[bool, str]`, defaults to `"max_length"`):
                Padding mode for the tokenizer. Supported values are [False/"do_not_pad", "max_length", True/"longest"].
        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
            padding=padding,
        )

        if do_classifier_free_guidance:
            negtive_prompt_embeds, negtive_pooled_prompt_embeds, negtive_prompt_attention_mask = self.encode_prompt(
                prompt=negtive_prompt,
                prompt_2=negtive_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negtive_prompt_embeds,
                pooled_prompt_embeds=negtive_pooled_prompt_embeds,
                prompt_attention_mask=negtive_prompt_attention_mask,
                device=device,
                max_sequence_length=max_sequence_length,
                padding=padding,
            )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        if do_classifier_free_guidance:
            negtive_prompt_embeds = negtive_prompt_embeds.to(transformer_dtype)
            if negtive_prompt_attention_mask is not None:
                negtive_prompt_attention_mask = negtive_prompt_attention_mask.to(transformer_dtype)
            if negtive_pooled_prompt_embeds is not None:
                negtive_pooled_prompt_embeds = negtive_pooled_prompt_embeds.to(transformer_dtype)
            prompt_embeds = [negtive_prompt_embeds, prompt_embeds]
            if prompt_attention_mask is not None:
                prompt_attention_mask = [negtive_prompt_attention_mask, prompt_attention_mask]
            else:
                prompt_attention_mask = [None, None]
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat([negtive_pooled_prompt_embeds, pooled_prompt_embeds])

        # 4. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
            if sigmas is None
            else sigmas
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_latent_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        if model_type == "t2v":
            ref_images = None
            mask_latents = None
        if ref_images is not None:
            ref_images = self.video_processor.preprocess(ref_images)
            if model_type == "i2v":
                ref_images = ref_images[:1, ...]
                ref_images = rearrange(
                    ref_images.unsqueeze(0), "b f c h w -> b c f h w"
                )
                ref_latents = self.vae.encode(
                    ref_images.to(dtype=self.vae.dtype, device=self.vae.device)
                ).latent_dist.sample()
                ref_latents = ref_latents * self.vae.config.scaling_factor
                b, c, f, h, w = latents.shape
                pad_size = latents.size(2) - ref_latents.size(2)
                ref_latents = F.pad(
                    ref_latents, (0, 0, 0, 0, 0, pad_size), mode="constant", value=0
                )
                mask_latents = torch.zeros(
                    (b, 1, f, h, w), dtype=self.vae.dtype, device=self.vae.device
                )
                mask_latents[:, :, 0] = 1
            
            elif model_type == 'token_replace':
                first_images = ref_images[:1, ...]
                first_images = rearrange(
                    first_images.unsqueeze(0), "b f c h w -> b c f h w"
                )
                first_images_latents = self.vae.encode(
                    first_images.to(dtype=self.vae.dtype, device=self.vae.device)
                ).latent_dist.sample()
                first_images_latents = first_images_latents * self.vae.config.scaling_factor
                if first_last:
                    last_images = ref_images[-1:, ...]
                    last_images = rearrange(
                        last_images.unsqueeze(0), "b f c h w -> b c f h w"
                    )
                    last_latents = self.vae.encode(
                        last_images.to(dtype=self.vae.dtype, device=self.vae.device)
                    ).latent_dist.sample()
                    last_latents = last_latents * self.vae.config.scaling_factor

                    latents = torch.cat(
                            [first_images_latents, latents[:,:,1:-1,:,:], last_latents], dim=2
                        )
                else:
                    latents = torch.cat(
                            [first_images_latents, latents[:,:,1:,:,:]], dim=2
                        )
                ref_images = None
                mask_latents = None

            else:
                ref_images = rearrange(
                    ref_images.unsqueeze(0), "b f c h w -> b c f h w"
                )
                ref_latents = self.vae.encode(
                    ref_images.to(dtype=self.vae.dtype, device=self.vae.device)
                ).latent_dist.sample()
                ref_latents = ref_latents * self.vae.config.scaling_factor
                mask_latents = None

        # 6. Prepare guidance condition
        if do_classifier_free_guidance:
            guidance = (
                torch.tensor(
                    [embedded_guidance_scale] * latents.shape[0] * 2,
                    dtype=transformer_dtype,
                    device=device,
                )
                * 1000.0
            )
        else:
            guidance = (
                torch.tensor(
                    [embedded_guidance_scale] * latents.shape[0],
                    dtype=transformer_dtype,
                    device=device,
                )
                * 1000.0
            )
        if do_classifier_free_guidance:
            if ref_latents is not None:
                ref_latents = torch.cat([ref_latents] * 2)
            if mask_latents is not None:
                mask_latents = torch.cat([mask_latents] * 2)
            if cn_images is not None:
                cn_latents = torch.cat([mask_latents] * 2)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latent_model_input = latents.to(transformer_dtype)
                latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.repeat(latent_model_input.shape[0]).to(latents.dtype)
                if ref_images is not None:
                    latent_model_input = torch.cat(
                        [latent_model_input, ref_latents], dim=1
                    )
                if mask_latents is not None:
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents], dim=1
                    )
                if cn_images is not None:
                    latent_model_input = torch.cat(
                        [latent_model_input, cn_latents], dim=1
                    )
                if do_classifier_free_guidance:
                    noise_pred_list = []
                    for idx in range(latent_model_input.shape[0]):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input[idx:idx+1],
                            timestep=timestep[idx:idx+1],
                            encoder_hidden_states=prompt_embeds[idx],
                            encoder_attention_mask=prompt_attention_mask[idx],
                            pooled_projections=pooled_prompt_embeds[idx:idx+1],
                            guidance=guidance[idx:idx+1],
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred_list.append(noise_pred_uncond)
                    noise_pred = torch.cat(noise_pred_list, dim=0)
                else:
                    noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            pooled_projections=pooled_prompt_embeds,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                    )[0]
                
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if cfg_zero_star:
                        if i <= 1:
                            noise_pred = noise_pred_text * 0
                        else:
                            batch_size = noise_pred_text.shape[0]
                            positive_flat = noise_pred_text.view(batch_size, -1)
                            negative_flat = noise_pred_uncond.view(batch_size, -1)

                            alpha = optimized_scale(positive_flat, negative_flat)
                            alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                            alpha = alpha.to(noise_pred_text.dtype)
                            noise_pred = noise_pred_uncond * alpha + guidance_scale * (noise_pred_text - noise_pred_uncond * alpha)
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if model_type == "token_replace":
                    if first_last:
                        latents = torch.cat(
                            [first_images_latents, latents[:,:,1:-1,:,:], last_latents], dim=2
                        )
                    else:
                        latents = torch.cat(
                            [first_images_latents, latents[:,:,1:,:,:]], dim=2
                        )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
