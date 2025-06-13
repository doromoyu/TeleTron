import gc
import dataclasses
from datetime import datetime
import math
import logging
import os
import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
# from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
from teletron.core.data_loader import build_pretraining_data_loader
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.pipeline_parallel import get_forward_backward_func
from teletron.core.parallel_state import get_transformer_model_group


from megatron.training.utils import print_rank_0


from megatron.training.global_vars import (
    get_args,
    get_timers,
    get_one_logger
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import _initialize_distributed, _set_random_seed,_init_autoresume, _compile_dependencies, _initialize_tp_communicators
from megatron.training.training import print_datetime, setup_model_and_optimizer, train, evaluate_and_print_results, build_train_valid_test_datasets, cyclic_iter

def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
):

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)
    if args.distributed_vae:
        args.world_size -= args.distributed_vae_world_size
        args.dit_world_size = args.world_size
    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)
    
    set_global_variables(args)
    if args.distributed_vae:
        args.world_size += args.distributed_vae_world_size
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))

        from teletron.core.parallel_state import get_transformer_model_group
        isDiTRank = get_transformer_model_group()
        if isDiTRank is not None:
            _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        
        finish_mpu_init()

        
        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        from teletron.core.parallel_state import get_transformer_model_group
        isConsumerRank = get_transformer_model_group()
        if isConsumerRank is not None:
            _compile_dependencies()

        

        if args.tp_comm_overlap:
           _initialize_tp_communicators()


        # No continuation function
        return None



def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             producer_process=None,
             extra_args_provider=None,
             args_defaults={}):

    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)
    
    
    args = get_args()
    timers = get_timers()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()
    
    transformer_group = get_transformer_model_group()
    if transformer_group is None:
        

        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                train_valid_test_dataset_provider)

        producer_process(dist.get_rank(), dist.get_world_size(),build_train_valid_test_data_iterators, train_valid_test_dataset_provider, train_ds=train_ds)
        exit()

    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])

    # Data stuff.
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(train_valid_test_dataset_provider)

        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider, train_ds_prev=train_ds)

    if args.distributed_vae:
        
        consumer_config = torch.zeros(
            (1), dtype=int, device=torch.cuda.current_device()
        )
        consumer_config[0] = args.iteration

        from teletron.core.parallel_state import get_comm_pair
        comm_pair = get_comm_pair()

        if comm_pair is not None:
            req = dist.isend(tensor=consumer_config, dst=comm_pair.producer , tag=0)
            req.wait()

    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config)

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider, is_tp_fist = None, dp_rank = None, dp_size = None,  train_ds_prev = None):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)
    # print(f"cs rank: {dist.get_rank()}, is_distributed: {is_distributed}")

    # Construct the data pipeline
    
    # if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
    if is_tp_fist is None:
        is_tp_fist = (mpu.get_tensor_model_parallel_rank() == 0)
    
    if is_distributed or is_tp_fist:
        # Build datasets.
        if train_ds_prev is not None:
            train_ds = train_ds_prev
            valid_ds = None
            test_ds = None
        else:
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                build_train_valid_test_datasets_provider)
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples, dp_rank, dp_size )
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0, dp_rank, dp_size )
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples, dp_rank, dp_size )
        test_dataloader = build_pretraining_data_loader(test_ds, 0,  dp_rank, dp_size )

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    if dp_rank is None or dp_size is None:
        torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider, is_tp_first = None, dp_rank = None, dp_size = None, train_ds_prev=None):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    print("Building loaders.")
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider, is_tp_first,dp_rank,dp_size, train_ds_prev)
    
    # Build iterators.
    print("Building iterators.")
    dl_type = args.dataloader_type

    assert dl_type in ['single', 'cyclic', 'external']

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator