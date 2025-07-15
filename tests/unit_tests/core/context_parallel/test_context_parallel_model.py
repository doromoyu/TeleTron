from unittest import TestCase
import os 
import torch
from unit_tests.test_utils import spawn
import logging
from unittest.mock import patch, Mock

SPLIT_SUCCESS = "split input success rank"
SPLIT_FAIL = "split input fail rank"


def split_input(cp_manager, cp_size, cp_rank, q):
    with torch.no_grad():
        x_list = []
        for i in range(cp_size):
            x = torch.zeros((1, 100, 128)) + i
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        # use logging.info to print things to the terminal instead of print(), print stdout will be eaten by pytest
        logging.info(f"{x.shape}, {cp_size}")
    x_split = cp_manager.split_input(x)
    if torch.all(x_split == x_list[cp_rank]):
        global SPLIT_SUCCESS
        q.put(f"{SPLIT_SUCCESS}{cp_rank}")
    else:
        global SPLIT_FAIL
        q.put(f"{SPLIT_FAIL}{cp_rank}")

@patch("teletron.utils.get_args")
def setupContextParallelModelManager(cp_rank, cp_size, q, mock_teletron):
    from teletron.core.context_parallel import ContextParallelModelManager
    from teletron.core.parallel_state  import initialize_model_parallel_base    
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
    
    cp_manager = ContextParallelModelManager()
    split_input(cp_manager, cp_size, cp_rank, q)


class testContextParallelModelManager(TestCase):
    def test_everything(self):
        cp_size = 4
        os.environ['WORLD_SIZE'] = str(cp_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12445'
        q = spawn(cp_size, setupContextParallelModelManager)
        global SPLIT_SUCCESS
        success_msgs = [f"{SPLIT_SUCCESS}{cp_rank}" for cp_rank in range(cp_size)]
        responses = []
        while not q.empty():
            responses.append(q.get())
        # breakpoint()
        self.assertEqual(sorted(responses), success_msgs)

