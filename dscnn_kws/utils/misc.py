from __future__ import annotations

import random

import numpy as np
import torch


def parameter_number(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def prepare_device(n_gpu_use: int):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu
    device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    device_ids = list(range(n_gpu_use))
    return device, device_ids


def set_random_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
