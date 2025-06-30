# Copyright (c) 2025 TeleAI-infra Team and Nvidia Megatron-LM Team. All rights reserved.

_GLOBAL_ARGS=None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None

from teletron.utils.microbatches import build_num_microbatches_calculator
from teletron.utils.tokenizer import build_tokenizer

def set_args(args, build_tokenizer=True):
    assert args is not None
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _build_num_microbatches_calculator(args)
    _GLOBAL_ARGS = args
    if build_tokenizer:
        _ = _build_tokenizer(args)

def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER

def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        args)

def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS



def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)

def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)

