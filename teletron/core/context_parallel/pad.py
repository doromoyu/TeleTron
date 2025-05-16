# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import torch

# Global variables
ORIGIN_LENGTH = None
TARGET_LENGTH = None

def set_origin_length(length):
    global ORIGIN_LENGTH
    ORIGIN_LENGTH = length

def set_target_length(length):
    global TARGET_LENGTH
    TARGET_LENGTH = length

def pad_for_context_parallel(tensor, dim):
    """
    Pads the input tensor along the specified dimension to TARGET_LENGTH.
    """
    if TARGET_LENGTH is None:
        raise ValueError("TARGET_LENGTH is not set.")
    current_length = tensor.size(dim)
    pad_size = int(TARGET_LENGTH - current_length)

    if pad_size <= 0:
        return tensor  # No padding needed

    # Create pad tuple: (dim_n_before, dim_n_after, ..., dim_0_before, dim_0_after)
    pad = [0] * (2 * tensor.dim())
    pad[-(2 * dim + 1)] = pad_size  # pad after the dimension
    return torch.nn.functional.pad(tensor, pad)

def remove_pad_for_context_parallel(tensor, dim):
    """
    Removes padding from the input tensor along the specified dimension to ORIGIN_LENGTH.
    """
    if ORIGIN_LENGTH is None:
        raise ValueError("ORIGIN_LENGTH is not set.")
    return tensor.narrow(dim, 0, ORIGIN_LENGTH)

def remove_pad_with_encoder_for_context_parallel(tensor: torch.Tensor, encoder_length: int, dim: int):
    total_length = tensor.size(dim)

    split_point = total_length - encoder_length
    first_raw = tensor.narrow(dim, 0, split_point)
    first = first_raw.narrow(dim, 0, ORIGIN_LENGTH)

    second = tensor.narrow(dim, split_point, encoder_length)

    result = torch.cat([first, second], dim=dim)
    return result
