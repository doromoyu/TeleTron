# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import os
import torch
import torch.distributed as dist
import collections
import time
from typing import Callable, Any, Dict

from teletron.core.parallel_state import get_comm_pair, get_world_group, CommPair
from teletron.utils import get_args
from teletron.train.checkpoint import ensure_directory_exists
from teletron.models.encoder_registry import get_encoder


NUM_ITEMS_PER_CONSUMER = 100000
MAX_QUEUE_PER_CONSUMER_ON_PRODUCER = 2
MAX_OUTSTANDING_SENDS_PER_CONSUMER = 1


def cleanup_dist():
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: 销毁进程组。")
        dist.destroy_process_group()


def merge_commpairs(commpairs: list) -> Dict[int, CommPair]:
    merge_dict = {}
    for cp in commpairs:
        key = (cp.producer, cp.dp_rank, cp.dp_size)
        if key not in merge_dict:
            merge_dict[key] = []
        if isinstance(cp.consumer, int):
            merge_dict[key].append([cp.consumer])
        else: 
            merge_dict[key].append(cp.consumer)
    
    merged_list = {idx: None for idx in range(len(merge_dict))}

    idx=0
    for key, consumers_list in merge_dict.items():
        flat_consumers = []
        for sublist in consumers_list:
            if isinstance(sublist, list):
                flat_consumers.extend(sublist)
            else:
                flat_consumers.append(sublist)
        new_cp = CommPair(
            producer=key[0],
            consumer=flat_consumers,
            dp_rank=key[1],
            dp_size=key[2]
        )
        merged_list[idx] = new_cp
        idx+=1
    return merged_list


def producer_process(
    rank: int,
    world_size: int,
    encoder_name:str,
    device,
    build_train_valid_test_data_iterators: Callable,
    train_ds: Any = None,
    valid_ds: Any = None, # TODO
):
    """
    通用的分布式数据生产者进程。

    Args:
        rank (int): 当前进程的排名
        world_size (int): 全局进程总数
        encoder_name: 编码器名称
        device: encoder device
        build_train_valid_test_data_iterators (Callable): 用于构建数据迭代器的函数。
        train_ds (Any, optional): 预加载的训练数据集。
    """
    encoder = get_encoder(name=encoder_name, device=device)
    encoder.setup()

    args = get_args()
    comm_pairs = get_comm_pair()

    consumers_data = torch.zeros((len(comm_pairs), 3), dtype=torch.int64, device=device)
    req_queue = [dist.irecv(tensor=consumers_data[i], src=cp.consumer) for i, cp in enumerate(comm_pairs)]
    for req in req_queue:
        req.wait()
    args.iteration = consumers_data[0][0]
    args.consumed_train_samples = consumers_data[0][1]
    args.consumed_valid_samples = consumers_data[0][2]

    merged_comm_pairs = merge_commpairs(comm_pairs)
    data_iterators = {}
    same_data_group = {}
    train_ds0 = train_ds
    valid_ds0 = valid_ds
    for idx, mcp in merged_comm_pairs.items():
        data_iterators[idx], _, _,  train_ds0, valid_ds0 = build_train_valid_test_data_iterators(
            is_tp_first=True, 
            dp_rank=mcp.dp_rank, 
            dp_size=mcp.dp_size, 
            train_ds_prev=train_ds0, 
            valid_ds_prev=valid_ds0, 
            return_ds=True
        )
        first_consumer = mcp.consumer[0]
        same_data_group[first_consumer] = mcp.consumer

    consumers_data_queues = {cp.consumer: collections.deque() for cp in comm_pairs}
    consumers_size_queues = {cp.consumer: collections.deque() for cp in comm_pairs}
    items_initiated_send = {cp.consumer: 0 for cp in comm_pairs}
    send_size_reqs = []  # (req, consumer_rank, item_idx, size_tensor)
    send_data_reqs = []  # (req, consumer_rank, item_idx, data_tensor)

    if args.producer_profile:
        prof_save_path = os.path.join(args.profile_path, f"producer/rank_{dist.get_rank()}.json")
        ensure_directory_exists(prof_save_path)
        def trace_handler(p):
            p.export_chrome_trace(prof_save_path)

        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=True,
            on_trace_ready=trace_handler,
            record_shapes=True
        )
    index = 0

    try:
        while any(items_initiated_send[cp.consumer] < NUM_ITEMS_PER_CONSUMER for cp in comm_pairs):
            if args.producer_profile and index == args.profile_step_start:
                prof.start()
            if args.producer_profile and index == args.profile_step_end:
                prof.stop()
            index += 1

            # --- 阶段 A: 处理已完成的发送请求 ---
            send_size_reqs = [r for r in send_size_reqs if not r[0].is_completed()]
            send_data_reqs = [r for r in send_data_reqs if not r[0].is_completed()]

            # --- 阶段 B: 为队列填充新数据 ---
            for idx, mcp in merged_comm_pairs.items():
                first_consumer = mcp.consumer[0]
                # 如果队列未满，则生成新数据
                if len(consumers_data_queues[first_consumer]) < MAX_QUEUE_PER_CONSUMER_ON_PRODUCER:
                    raw_batch = next(data_iterators[idx])
                    
                    # 使用注入的编码器处理数据
                    tensors_to_send, size_info_tensor = encoder.encode(raw_batch)
                    packed_tensor = encoder._pack_tensors(tensors_to_send)

                    # 将编码后的数据分发给所有需要相同数据的消费者
                    for consumer_rank in same_data_group[first_consumer]:
                        consumers_size_queues[consumer_rank].append(size_info_tensor)
                        consumers_data_queues[consumer_rank].append(packed_tensor)

            # --- 阶段 C: 启动新的发送操作 ---
            # 首先发送尺寸信息
            for cp in comm_pairs:
                cr = cp.consumer
                outstanding_sends = sum(1 for _, c, _, _ in send_size_reqs if c == cr)
                if consumers_size_queues[cr] and outstanding_sends < MAX_OUTSTANDING_SENDS_PER_CONSUMER:
                    size_to_send = consumers_size_queues[cr].popleft()
                    item_idx = items_initiated_send[cr]
                    
                    req = dist.isend(tensor=size_to_send, dst=cr)
                    send_size_reqs.append((req, cr, item_idx, size_to_send))

                    # 只有在尺寸发送后才发送数据
                    tensor_to_send = consumers_data_queues[cr].popleft()
                    req_data = dist.isend(tensor=tensor_to_send, dst=cr)
                    send_data_reqs.append((req_data, cr, item_idx, tensor_to_send))

                    items_initiated_send[cr] += 1
            
            time.sleep(0.01)

        # 6. 等待所有挂起的通信完成
        print("所有数据项已启动发送，等待最终完成...")
        for req, _, _, _ in send_size_reqs + send_data_reqs:
            req.wait()
        
        dist.barrier(group=get_world_group())

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 强制中止，避免部分进程挂起
        dist.abort(group=get_world_group())
    finally:
        cleanup_dist()
