from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def init_distributed(enabled: bool, local_rank: int | None = None) -> DistributedContext:
    if not enabled:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistributedContext(False, 0, 1, 0, device)
    if not torch.cuda.is_available():
        raise RuntimeError("--distributed requires CUDA/NCCL")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    resolved_local_rank = int(os.environ.get("LOCAL_RANK", local_rank if local_rank is not None else 0))
    torch.cuda.set_device(resolved_local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return DistributedContext(True, rank, world_size, resolved_local_rank, torch.device("cuda", resolved_local_rank))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank_zero() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def reduce_epoch_totals(values: torch.Tensor) -> torch.Tensor:
    """Sum scalar/vector totals across ranks, leaving the caller's tensor unchanged."""
    result = values.detach().clone()
    if is_distributed():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def reduce_max(values: torch.Tensor) -> torch.Tensor:
    """Take the element-wise maximum across ranks without mutating the input."""
    result = values.detach().clone()
    if is_distributed():
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result


def destroy_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
