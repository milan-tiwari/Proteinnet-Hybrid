from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def init_distributed() -> DistInfo:
    """Initialize torch.distributed if launched with torchrun.

    Returns:
        DistInfo describing the distributed context.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo(enabled=False, rank=0, world_size=1, local_rank=0)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)

    return DistInfo(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def is_main_process(dist_info: DistInfo) -> bool:
    return (not dist_info.enabled) or dist_info.rank == 0


def barrier(dist_info: DistInfo) -> None:
    if dist_info.enabled:
        dist.barrier()


def cleanup_distributed(dist_info: DistInfo) -> None:
    if dist_info.enabled and dist.is_initialized():
        dist.destroy_process_group()
