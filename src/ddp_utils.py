import os
import sys

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_logger(rank=-1, world_size=1):
    """Setup logger for distributed training"""
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": (
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>Rank {extra[rank]}/{extra[world_size]}</cyan> | "
                    "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                "level": "DEBUG",
                "colorize": True,
            }
        ]
    }

    try:  # Remove all existing handlers
        logger.configure(**config)
    except ValueError:
        pass

    # Create a new logger instance with rank information
    return logger.bind(rank=rank, world_size=world_size)


def setup_distributed(params):
    """Initialize distributed training environment with explicit device mapping"""
    if not params.distributed:
        return False

    try:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            params.rank = int(os.environ["RANK"])
            params.world_size = int(os.environ["WORLD_SIZE"])
            params.local_rank = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            params.rank = int(os.environ["SLURM_PROCID"])
            params.local_rank = params.rank % torch.cuda.device_count()
            params.world_size = int(os.environ["SLURM_NTASKS"])
        else:
            raise ValueError("Not running with distributed environment variables set")

        torch.cuda.set_device(params.local_rank)
        init_method = "env://"
        backend = params.dist_backend
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=params.world_size,
                rank=params.rank,
            )
            torch.cuda.set_device(params.local_rank)
            dist.barrier(device_ids=[params.local_rank])

        return True
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed training resources safely with device mapping"""
    if dist.is_initialized():
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.barrier(device_ids=[local_rank])
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error during distributed cleanup: {e}")
