# Copyright (c) 2025 TeleAI-infra and Nvidia Megatron-LM Team. All rights reserved.

from functools import wraps

import torch
import megatron.core.parallel_state   as parallel_state

from typing import Optional
from datetime import timedelta
from functools import reduce
import operator
import os

_TENSOR_CONTEXT_PARALLEL_GROUP = None
_MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_CONTEXT_PARALLEL_RANK = None

def initialize_model_parallel_decorators(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
                tensor_model_parallel_size: int = 1,
                pipeline_model_parallel_size: int = 1,
                virtual_pipeline_model_parallel_size: Optional[int] = None,
                pipeline_model_parallel_split_rank: Optional[int] = None,
                use_sharp: bool = False,
                context_parallel_size: int = 1,
                expert_model_parallel_size: int = 1,
                nccl_communicator_config_path: Optional[str] = None,
                distributed_timeout_minutes: int = 30,
    ):
        # Calling the original Megatron's initialize_model_parallel function is to initialize the global parameters.
        # context_parallel_size ==1 , because the original Megatron does not have a good adaptation for this parameter, and it will throw an error if it is greater than 1.
        initialize_model_parallel(              
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            use_sharp,
            1,                                                              
            expert_model_parallel_size,
            nccl_communicator_config_path,
        )

        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()

        if (
            world_size
            % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
            != 0
        ):
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
                f"x context_parallel_size ({context_parallel_size})"
            )

        data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )

        if data_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
            )

        if expert_model_parallel_size > 1 and context_parallel_size > 1:
            raise RuntimeError(
                f"combination of expert model prallellism and context parallelism is not supported"
            )

        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 2:
                raise RuntimeError(
                    "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
                )
            # global parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
            # global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
            parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

        if pipeline_model_parallel_split_rank is not None:
            global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
            _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

        rank = torch.distributed.get_rank()

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            try:
                import yaml
            except ImportError:
                raise RuntimeError(
                    "Cannot import `yaml`. Setting custom nccl communicator configs "
                    "requires the yaml package."
                )

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        timeout = timedelta(minutes=distributed_timeout_minutes)

        # Build the data-parallel groups.
        all_data_parallel_group_ranks_with_cp = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('dp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
                if rank in ranks:
                    parallel_state._DATA_PARALLEL_GROUP = group
                    parallel_state._DATA_PARALLEL_GROUP_GLOO = group_gloo
                    parallel_state._DATA_PARALLEL_GLOBAL_RANKS = ranks
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                group_with_cp = torch.distributed.new_group(
                    ranks_with_cp, timeout=timeout, pg_options=parallel_state.get_nccl_options('dp_cp', nccl_comm_cfgs)
                )
                group_with_cp_gloo = torch.distributed.new_group(
                    ranks_with_cp, timeout=timeout, backend="gloo"
                )
                if rank in ranks_with_cp:
                    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                    parallel_state._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                    parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        # Apply SHARP to DP process groups
        if use_sharp:
            if rank == 0:
                print(
                    "The number of process groups to use SHARP with depends on the type "
                    "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                    "process groups and QM2 supports up to 256 process groups. We apply "
                    "SHARP to the communications of the data-parallel domain. If the "
                    "number of data-parallel process groups is larger than the max "
                    "process groups that the network switch supports, the communication "
                    "will fall back to non-SHARP operators. To enable SHARP, "
                    "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
                )
            torch.distributed.barrier(
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                device_ids=[torch.cuda.current_device()],
            )
            # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
            os.environ["NCCL_COLLNET_ENABLE"] = "0"

        # Build the context-parallel groups.
        for i in range(pipeline_model_parallel_size):
            for j in range(data_parallel_size):
                start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
                )
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    group = torch.distributed.new_group(
                        ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('cp', nccl_comm_cfgs)
                    )
                    if rank in ranks:
                        parallel_state._CONTEXT_PARALLEL_GROUP = group
                        parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

        # Build the model-parallel groups.
        for i in range(data_parallel_size * context_parallel_size):
            ranks = [
                data_parallel_group_ranks_with_cp[i]
                for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
            ]
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('mp', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._MODEL_PARALLEL_GROUP = group

        # Build the tensor model-parallel groups.
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('tp', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP = group

        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('pp', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = group
                parallel_state._PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(
                embedding_ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('embd', nccl_comm_cfgs)
            )
            if rank in embedding_ranks:
                parallel_state._EMBEDDING_GROUP = group
            if rank in ranks:
                parallel_state._EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(
                position_embedding_ranks,
                timeout=timeout,
                pg_options=parallel_state.get_nccl_options('embd', nccl_comm_cfgs),
            )
            if rank in position_embedding_ranks:
                parallel_state._POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        # Build the tensor + data parallel groups.
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

            for j in range(context_parallel_size):
                ranks = []
                for k in range(data_parallel_size):
                    start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * tensor_model_parallel_size
                        + k * tensor_model_parallel_size * context_parallel_size
                    )
                    end_rank = start_rank + tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('tp_dp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP = group

        # # Build the tensor + context parallel groups
        global _TENSOR_CONTEXT_PARALLEL_GROUP
        assert (
        _TENSOR_CONTEXT_PARALLEL_GROUP is None
        ), 'Tensor + context parallel group is already initialized'
        tensor_and_context_group_size: int = tensor_model_parallel_size * context_parallel_size
        num_tensor_and_context_groups: int = world_size // tensor_and_context_group_size
        print(f"world_size: {world_size}, {tensor_and_context_group_size}")
        for i in range(num_tensor_and_context_groups):
            start_rank = i * tensor_and_context_group_size
            end_rank = start_rank + tensor_and_context_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('tp_cp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_CONTEXT_PARALLEL_GROUP = group

        # Build the tensor + expert parallel groups
        tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size // expert_model_parallel_size
        for i in range(num_tensor_and_data_groups):
            for j in range(num_expert_groups):
                # TPxEP Group
                start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('tp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group
                for k in range(tensor_model_parallel_size):
                    ranks = range(
                        start_rank + k, end_rank, tensor_model_parallel_size
                    )
                    group = torch.distributed.new_group(
                        ranks, pg_options=parallel_state.get_nccl_options('exp', nccl_comm_cfgs)
                    )
                    if rank in ranks:
                        parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

        for i in range(num_tensor_and_data_groups):
            start_rank = i * tensor_and_data_group_size
            end_rank = (i + 1) * tensor_and_data_group_size
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=parallel_state.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

        # Initialize global memory buffer
        # This isn't really "parallel state" but there isn't another good place to
        # put this. If we end up with a more generic initialization of megatron-core
        # we could stick it there
        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

    return wrapper


def get_tensor_context_parallel_group(check_initialized=True):
    """Get the tensor context parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_CONTEXT_PARALLEL_GROUP is not None
        ), 'tensor context parallel group is not initialized'
    return _TENSOR_CONTEXT_PARALLEL_GROUP

def get_tensor_context_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_context_parallel_group())

def get_tensor_context_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    if _MPU_TENSOR_CONTEXT_PARALLEL_RANK is not None:
        return _MPU_TENSOR_CONTEXT_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_context_parallel_group())

def get_tensor_context_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_context_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
    
def destroy_model_parallel_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()

        global _TENSOR_CONTEXT_PARALLEL_GROUP
        _TENSOR_CONTEXT_PARALLEL_GROUP = None
        global _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE
        _MPU_TENSOR_CONTEXT_PARALLEL_WORLD_SIZE = None
        global _MPU_TENSOR_CONTEXT_PARALLEL_RANK
        _MPU_TENSOR_CONTEXT_PARALLEL_RANK= None
