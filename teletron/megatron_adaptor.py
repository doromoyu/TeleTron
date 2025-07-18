# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
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

def exe_adaptation():

    import megatron.core
    from teletron.core.parallel_state import initialize_model_parallel_decorators
    from teletron.core.parallel_state import destroy_model_parallel_wrapper
    from teletron.core.parallel_state import (get_tensor_context_parallel_group, 
                                            get_tensor_context_parallel_rank, 
                                            get_tensor_context_parallel_src_rank,
                                            get_tensor_context_parallel_world_size)
    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorators(
        megatron.core.parallel_state.initialize_model_parallel
    )
    megatron.core.parallel_state.destroy_model_parallel = destroy_model_parallel_wrapper(
        megatron.core.parallel_state.destroy_model_parallel
    )
    megatron.core.parallel_state.get_tensor_context_parallel_group = get_tensor_context_parallel_group
    megatron.core.parallel_state.get_tensor_context_parallel_rank = get_tensor_context_parallel_rank
    megatron.core.parallel_state.get_tensor_context_parallel_world_size = get_tensor_context_parallel_world_size
    megatron.core.parallel_state.get_tensor_context_parallel_src_rank = get_tensor_context_parallel_src_rank
    megatron.core.mpu = megatron.core.parallel_state


exe_adaptation()