# Copyright (c) 2025 TeleAI-infra and Nvidia Megatron-LM Team. All rights reserved.

import dataclasses
from functools import wraps

from megatron.training.utils import (
    print_rank_0,
    unwrap_model)

from megatron.training.global_vars  import (
    get_args,
    get_timers)

from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.checkpointing import load_checkpoint
import megatron.training.training as training



def setup_model_and_optimizer_decorators(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(
                model_provider_func,
                model_type,
                no_wd_decay_cond=None,
                scale_lr_cond=None,
                lr_mult=1.0
    ):
        """Setup model and optimizer."""
        args = get_args()
        timers = get_timers()

        model = training.get_model(model_provider_func, model_type)
        unwrapped_model = unwrap_model(model)

        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = timers
        optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                        scale_lr_cond, lr_mult)
        opt_param_scheduler = training.get_optimizer_param_scheduler(optimizer)

        if args.load is not None or args.pretrained_checkpoint is not None:
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                model, optimizer, opt_param_scheduler, strict=False)
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0

        # get model without FP16 and/or DDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        return model, optimizer, opt_param_scheduler
    
    return wrapper