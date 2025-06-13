# train_data_iterator, valid_data_iterator, test_data_iterator \
#             = build_train_valid_test_data_iterators(
#                 train_valid_test_dataset_provider)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import sys
import collections
from megatron.core import mpu
from diffusers import AutoencoderKLHunyuanVideo
from einops import rearrange
from vast.models import utils as gm_utils
from config.hunyuanvideo_i2vhy import config
from megatron.training import get_args
from teletron.models.hunyuanvideo.text_encoder import PromptEncoder
from teletron.models.hunyuanvideo.clip_transform import CLIPTextTransform
import copy
from teletron.core.parallel_state import get_world_group


NUM_ITEMS_PER_CONSUMER = 100000
MAX_QUEUE_PER_CONSUMER_ON_PRODUCER = 2
MAX_OUTSTANDING_SENDS_PER_CONSUMER = 1 

def cleanup_dist():
    # pass
    if dist.is_initialized():
        rank = dist.get_rank()
        dist.destroy_process_group()
        print(f"Rank {rank}: 进程组已销毁。")
    else:
        print("进程组未初始化或已被销毁。")

class ImagesVAE:
    def __init__(
        self,
        model_dir,
        vae_slicing,
        vae_tiling,
        device
        ):
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(model_dir,torch_dtype=torch.bfloat16)
        self.vae.to(device)
        self.vae.requires_grad_(False)
        if vae_slicing:
                self.vae.enable_slicing()
        if vae_tiling:
               self.vae.enable_tiling()
    
    def __call__(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, "b f c h w -> b c f h w")
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

class PromptToTransformerEmbedding:
    """
    extract text embedding from prompts
    """

    def __init__(
        self,
        model_name,
        model_path,
        max_length=None,
        with_attention_mask=False,
        padding="max_length",
        device='cpu',
        dtype = torch.bfloat16
    ):
        self.prompt_encoder = PromptEncoder(
            model_name, gm_utils.get_model_path(model_path), device=device,dtype=dtype
        )
        self.max_length = max_length
        self.with_attention_mask = with_attention_mask
        self.padding = padding

    def __call__(self, prompt):
        prompt_embeds, prompt_masks = self.prompt_encoder(
            prompt,
            max_length=self.max_length,
            with_attention_mask=self.with_attention_mask,
            padding=self.padding,
        )
        return prompt_embeds[0][None,:]

class PromptToClipEmbedding:
    def __init__(self, model_path, dtype=torch.bfloat16, device='cpu') -> None:
        self.clip_transform = CLIPTextTransform(
            gm_utils.get_model_path(model_path), dtype=dtype, device=device
        )

    def __call__(self, prompt):
        clip_text_embed = self.clip_transform(
            prompt, mode="after_pool", to_numpy=False
        )[0][None, :]

        return clip_text_embed


def producer_process(rank,world_size,build_train_valid_test_data_iterators, train_valid_test_dataset_provider, train_ds=None):
    args = get_args()
    import os
    dit_world_size = args.dit_world_size
    rank = rank - dit_world_size
    
    torch.manual_seed(1234)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and dist.get_backend() == 'nccl' else "cpu")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # load VAE, Llama, CLIP
    vae = ImagesVAE(config['models']['vae_path'], vae_slicing=args.vae_slicing, vae_tiling=args.vae_tiling,device=device)

    tf_embed_model = PromptToTransformerEmbedding(
                                                model_name=config['models']['PromptToTransformerEmbedding']['model_name'],
                                                model_path = config['models']['PromptToTransformerEmbedding']['model_path'],
                                                max_length=config['models']['PromptToTransformerEmbedding']['max_length'],
                                                with_attention_mask=config['models']['PromptToTransformerEmbedding']['with_attention_mask'],
                                                padding=config['models']['PromptToTransformerEmbedding']['padding'],
                                                device=device)
    
    cp_embed_model = PromptToClipEmbedding(
        model_path=config['models']['clip_path'],
        device=device
    )


    from teletron.core.parallel_state import get_comm_pair
    comm_pairs = get_comm_pair()

    consumers_data = torch.zeros(
        (len(comm_pairs), 1), dtype=int, device=torch.cuda.current_device()
    )
    consumers_queue = []
    idx=0
    for ccs in comm_pairs:
        req = dist.irecv(tensor=consumers_data[idx], src=ccs.consumer, tag=0)
        consumers_queue.append(req)
        idx+=1

    while len(consumers_queue) > 0:
        for ccs_req in  consumers_queue:
            if ccs_req.is_completed():
                consumers_queue.remove(ccs_req)
        time.sleep(0.1)

    args.iteration = consumers_data[0][0]

    train_data_iterators = {comm_pair.consumer: None for comm_pair in comm_pairs}
    valid_data_iterators = {comm_pair.consumer: None for comm_pair in comm_pairs}
    test_data_iterators = {comm_pair.consumer: None for comm_pair in comm_pairs}

    for comm_pair in comm_pairs:
        train_data_iterators[comm_pair.consumer], valid_data_iterators[comm_pair.consumer], test_data_iterators[comm_pair.consumer] \
                = build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, 
                    is_tp_first = True, 
                    dp_rank = comm_pair.dp_rank, 
                    dp_size = comm_pair.dp_size, 
                    train_ds_prev =train_ds )
        

    producer_gpu_queues = {
        crank.consumer: collections.deque() for crank in comm_pairs
    }

    items_initiated_send_for_consumer = {crank.consumer: 0 for crank in comm_pairs}

    send_requests_in_flight = []

    try:
        while any(items_initiated_send_for_consumer[crank.consumer] < NUM_ITEMS_PER_CONSUMER for crank in comm_pairs):

            new_send_requests_in_flight = []
            for req_list, cr, item_idx_req, tensor2send in send_requests_in_flight:
                if req_list.is_completed() is True:

                    del tensor2send
                    
                else:
                    new_send_requests_in_flight.append((req_list, cr, item_idx_req, tensor2send))
            send_requests_in_flight = new_send_requests_in_flight


            for current_consumer_rank in comm_pairs:
     
                items_in_queue = len(producer_gpu_queues[current_consumer_rank.consumer])
                
                if items_in_queue < MAX_QUEUE_PER_CONSUMER_ON_PRODUCER:
                    
                    data = next(train_data_iterators[current_consumer_rank.consumer])     

                    t1 = torch.flatten(vae(data['images'].cuda()))
                    t2 = torch.flatten(tf_embed_model(data['prompt']).to(torch.bfloat16))
                    token_length = t2.size(0)//4096
                    t_length = torch.empty(1, dtype=torch.bfloat16, device=device)
                    t_length[0] = token_length
                    t2 = torch.nn.functional.pad(t2, (0, 4096*(config['models']['PromptToTransformerEmbedding']['max_length'] - token_length)), value=0.0)
                    t3 = torch.flatten(cp_embed_model(data['prompt']).to(torch.bfloat16))
                    t4 = torch.flatten(vae(data['first_ref_image'].cuda()))
                    result = torch.cat((t_length,t2, t3, t4, t1), dim=0)
                    producer_gpu_queues[current_consumer_rank.consumer].append(result)
                    
            for current_consumer_rank in comm_pairs:
                outstanding_sends_for_this_consumer = sum(
                    1 for _, cr, _, _ in send_requests_in_flight if cr == current_consumer_rank.consumer
                )
                if producer_gpu_queues[current_consumer_rank.consumer] and \
                    outstanding_sends_for_this_consumer < MAX_OUTSTANDING_SENDS_PER_CONSUMER:
                    
                    batch_to_send = producer_gpu_queues[current_consumer_rank.consumer].popleft()
                    current_item_idx_to_send = items_initiated_send_for_consumer[current_consumer_rank.consumer]
                    
                    req = dist.isend(tensor=batch_to_send, dst=current_consumer_rank.consumer, tag=1)
                    send_requests_in_flight.append((req, current_consumer_rank.consumer, current_item_idx_to_send, batch_to_send))
                    items_initiated_send_for_consumer[current_consumer_rank.consumer] += 1

            time.sleep(0.05)


        for req_obj, cr, item_idx_req,_ in send_requests_in_flight:
            req_obj.wait()

        dist.barrier(group=get_world_group())

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        cleanup_dist()


