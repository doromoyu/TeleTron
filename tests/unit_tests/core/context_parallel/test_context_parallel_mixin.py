from unittest import TestCase
import os 
import torch
import torch.nn.functional as F
from einops import rearrange
from unittest.mock import patch, Mock
from unit_tests.test_utils import spawn
import logging

SPLIT_SUCCESS = "split input success rank"
SPLIT_FAIL = "split input fail rank"
GATHER_SUCCESS = "gather output success rank"
GATHER_FAIL = "gather output fail rank"
ATTN_SUCCESS = "cp attn compute success rank"
ATTN_FAIL = "cp attn compute fail rank"

def forward_attn(cp_model, cp_size, cp_rank, que):
    with torch.no_grad():
        q_ori = torch.zeros((1, 98, 128)) + cp_size
        k_ori = torch.zeros((1, 98, 128)) + cp_size
        v_ori = torch.zeros((1, 98, 128)) + cp_size

        # original attn compute result
        q = rearrange(q_ori, "b s (n d) -> b s n d", n=16)
        k = rearrange(k_ori, "b s (n d) -> b s n d", n=16)
        v = rearrange(v_ori, "b s (n d) -> b s n d", n=16)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).flatten(2, 3).contiguous().cuda()
        
        # cp attn compute result
        q_split = cp_model.split_input(q_ori, dim=1).cuda()
        k_split = cp_model.split_input(k_ori, dim=1).cuda()
        v_split = cp_model.split_input(v_ori, dim=1).cuda()
        output_split = cp_model.forward_attn(q_split, k_split, v_split).cuda()
        output = cp_model.gather_output(output_split, dim=1)
        # use logging.info to print things to the terminal instead of print(), print stdout will be eaten by pytest
        logging.info(f"{x.shape}, {cp_size}")
    if torch.all(x == output):
        global ATTN_SUCCESS
        que.put(f"{ATTN_SUCCESS}{cp_rank}")
    else:
        global ATTN_FAIL
        que.put(f"{ATTN_FAIL}{cp_rank}")

def gather_output(cp_model, cp_size, cp_rank, q):
    with torch.no_grad():
        input = torch.zeros((1, 100, 128)) + cp_rank
        input = input.cuda()
        x_list = []
        for i in range(cp_size):
            x = torch.zeros((1, 100, 128)) + i
            x_list.append(x)
        output = torch.cat(x_list, dim=1).cuda()
        # use logging.info to print things to the terminal instead of print(), print stdout will be eaten by pytest
        logging.info(f"{output.shape}, {cp_size}")
    input_gather = cp_model.gather_output(input, dim=1)
    if torch.all(input_gather == output):
        global GATHER_SUCCESS
        q.put(f"{GATHER_SUCCESS}{cp_rank}")
    else:
        global GATHER_FAIL
        q.put(f"{GATHER_FAIL}{cp_rank}")

def split_input(cp_model, cp_size, cp_rank, q):
    with torch.no_grad():
        x_list = []
        for i in range(cp_size):
            x = torch.zeros((1, 100, 128)) + i
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        # use logging.info to print things to the terminal instead of print(), print stdout will be eaten by pytest
        logging.info(f"{x.shape}, {cp_size}")
    x_split = cp_model.split_input(x, dim=1)
    if torch.all(x_split == x_list[cp_rank]):
        global SPLIT_SUCCESS
        q.put(f"{SPLIT_SUCCESS}{cp_rank}")
    else:
        global SPLIT_FAIL
        q.put(f"{SPLIT_FAIL}{cp_rank}")

@patch("teletron.utils.get_args")
def setupContextParallelMixin(cp_rank, cp_size, q, mock_teletron):
    torch.cuda.set_device(cp_rank)
    from megatron.core import mpu
    from teletron.core.context_parallel import ContextParallelMixin
    from teletron.core.parallel_state import initialize_model_parallel_base
    args = Mock()
    args.recompute_method = "block"
    args.recompute_granularity = "full"
    args.recompute_num_layers = 1
    args.activation_offload = True
    args.num_layers = 1 
    args.num_attention_heads = 4
    args.distributed_vae = False
    args.consumer_models_num = 1
    mock_teletron.return_value = args
    
    class ContextParallelModel(ContextParallelMixin):
        def __init__(self, split_dim=1, gather_dim=1):
            self.cp_size = mpu.get_context_parallel_world_size()
            self.cp_group = mpu.get_context_parallel_group()
            self.split_dim = split_dim
            self.gather_dim = gather_dim 
            self.num_heads = 16 
            self.use_pad = None

    torch.distributed.init_process_group(world_size=cp_size, rank=cp_rank)
    initialize_model_parallel_base(
            tensor_model_parallel_size = 1,
            pipeline_model_parallel_size = 1,
            virtual_pipeline_model_parallel_size = None,
            pipeline_model_parallel_split_rank = None,
            use_sharp = False,
            context_parallel_size = cp_size,
            expert_model_parallel_size = 1,
            nccl_communicator_config_path = None,
            distributed_timeout_minutes = 30,
        )
    
    cp_model = ContextParallelModel()
    split_input(cp_model, cp_size, cp_rank, q)
    gather_output(cp_model, cp_size, cp_rank, q)
    forward_attn(cp_model, cp_size, cp_rank, q)


class testContextParallelMixin(TestCase):
    def test_everything(self):
        cp_size = 4
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(cp_size, setupContextParallelMixin)
        global GATHER_SUCCESS
        gather_success_msgs = [f"{GATHER_SUCCESS}{cp_rank}" for cp_rank in range(cp_size)]
        global SPLIT_SUCCESS
        split_success_msgs = [f"{SPLIT_SUCCESS}{cp_rank}" for cp_rank in range(cp_size)]
        global ATTN_SUCCESS
        attn_success_msgs = [f"{ATTN_SUCCESS}{cp_rank}" for cp_rank in range(cp_size)]
        responses = []
        while not q.empty():
            responses.append(q.get())
        # breakpoint()
        self.assertEqual(sorted(responses)[0:cp_size], attn_success_msgs)
        self.assertEqual(sorted(responses)[cp_size:cp_size*2], gather_success_msgs)
        self.assertEqual(sorted(responses)[cp_size*2:cp_size*3], split_success_msgs)