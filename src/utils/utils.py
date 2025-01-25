

import os
import sys
import torch
import torch.distributed as dist

import logging
logger = logging.getLogger(__name__)

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def to_embedding(model, mini_batch_size, data):

    outputs = []

    # break down the batch to avoid OOM
    while True:
        try:
            for i in range(0, data.shape[0], mini_batch_size):
                inputs = data[i:i+mini_batch_size]

                # forward pass, make sure to detach the output to avoid memory leak
                outputs.append(model(inputs).detach().cpu())

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if mini_batch_size == 1:
                raise e
            mini_batch_size = mini_batch_size // 2

            logger.info(f"OOM, reducing mini_batch_size to {mini_batch_size}")
            continue
        break

    # free cuda memory
    torch.cuda.empty_cache()

    outputs = torch.cat(outputs, dim=0)

    return outputs