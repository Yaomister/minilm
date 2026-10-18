import os
import torch
import logging
import torch.distributed as dist
from utils.common import get_device_type

logger = logging.getLogger(__name__)


def ddp(device=None):
    if device is None:
        device = get_device_type()
    assert device in ["cpu", "cuda", "mps"], "invalid device."
    if device == "cuda":
        assert torch.cuda.is_available(), "cuda is not available."
    if device == "mps":
        assert torch.backends.mps.is_available(), "mps is not available."

    torch.manual_seed(100)
    if device == "cuda":
        torch.cuda.manual_seed(100)

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    is_ddp_available, ddp_rank, ddp_local_rank, ddp_world_size = get_ddp_information()
    if is_ddp_available and device == "cuda":
        device_id = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device_id)
    else:
        device_id = torch.device(device)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_available, ddp_rank, ddp_local_rank, ddp_world_size, device


def is_ddp_available():
    return all([k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")])


def get_ddp_information():
    if is_ddp_available():
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
