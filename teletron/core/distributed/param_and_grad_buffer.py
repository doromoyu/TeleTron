import torch
def start_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the grad buffer.

    When overlap_grad_reduce is set to True, dispatches asynchronous communication
    calls. When overlap_grad_reduce is set to False, calls synchronous
    communication ops.
    """
    for bucket in self.buckets:
        bucket.grad_data = bucket.grad_data.to(torch.float32)
        bucket.start_grad_sync()
        bucket.grad_data = bucket.grad_data.to(torch.bfloat16)