import argparse
import time
import os
from typing import Any

import wandb
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import imageio
from loguru import logger
import torch.optim as optim
import torch_optimizer as jettify_optim

from ddp_utils import setup_logger


def init_wandb(params, hash):
    if params.use_wandb:
        if wandb.run is not None:
            wandb.finish()

        run_id = None
        if params.local_rank == 0:
            wandb.init(
                project=params.wandb_project,
                config=vars(params),
                name=f"{params.dataset.upper()}_PROGRESS{params.train_seq}_{hash}",
                settings=wandb.Settings(start_method="thread", _disable_stats=True),
            )
            run_id = wandb.run.id

        if params.distributed:
            object_list = [run_id if params.local_rank == 0 else None]
            dist.broadcast_object_list(object_list, src=0)
            run_id = object_list[0]

        if params.local_rank != 0:
            wandb.init(
                project=params.wandb_project,
                id=run_id,
                resume="allow",
                settings=wandb.Settings(start_method="thread", _disable_stats=True),
            )


def exec_time(func):
    """Decorates a function to measure its execution time in hours and minutes."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        logger = kwargs.get("logger")  # Get logger from kwargs
        if not logger:  # Find logger in positional arguments
            for arg in args:
                if isinstance(arg, type(setup_logger())):
                    logger = arg
                    break

        if logger:
            logger.success(
                f"Training completed ({int(execution_time // 3600)}h {int((execution_time % 3600) // 60)}min)"
            )
        return result

    return wrapper


def add_dataclass_args(parser: argparse.ArgumentParser, dataclass_type: Any):
    for field_info in dataclass_type.__dataclass_fields__.values():
        # Skip properties (those methods marked with @property)
        if isinstance(field_info.type, property):
            continue

        # bool type
        if field_info.type is bool:
            parser.add_argument(
                f"--{field_info.name}",
                action="store_true" if not field_info.default else "store_false",
                help=f"Set {field_info.name} to {not field_info.default}",
            )
        # tuple, list, float
        elif isinstance(field_info.default, tuple):
            parser.add_argument(
                f"--{field_info.name}",
                type=lambda x: tuple(map(float, x.split(","))),
                default=field_info.default,
                help=f"Set {field_info.name} to a tuple of floats (e.g., 0.9,0.999)",
            )
        elif isinstance(field_info.default, list):
            parser.add_argument(
                f"--{field_info.name}",
                type=lambda x: list(map(float, x.split(","))),
                default=field_info.default,
                help=f"Set {field_info.name} to a list of floats (e.g., 0.1,0.2,0.3)",
            )
        else:
            parser.add_argument(
                f"--{field_info.name}",
                type=field_info.type,
                default=field_info.default,
                help=f"Set {field_info.name} to a value of type {field_info.type.__name__}",
            )


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device, logger):
    """Load a model checkpoint with proper device management."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        # Load model state dict
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Load optimizer state dict
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Loaded checkpoint from '{checkpoint_path}' (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})"
        )

        return model, optimizer, scaler
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return model, optimizer, scaler


def save_reconstruction(inputs, reconstructions, save_path):
    """Save a grid of original and reconstructed images"""
    batch_size = min(8, inputs.shape[0])
    inputs = inputs[:batch_size].float()
    reconstructions = reconstructions[:batch_size].float()
    comparison = torch.cat([inputs[:batch_size], reconstructions[:batch_size]])

    # Denormalize and convert to numpy
    images = comparison.cpu().detach()
    images = torch.clamp(images, 0, 1)
    grid = torchvision.utils.make_grid(images, nrow=batch_size)
    image = grid.permute(1, 2, 0).numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, (image * 255).astype("uint8"))


def save_input_image(inputs: torch.Tensor, save_dir: str, seq: int, size: int = 96) -> str:
    input_path = os.path.join(save_dir, f"traverse_input_seq{seq}.png")
    os.makedirs(save_dir, exist_ok=True)

    input_img = inputs[0].cpu().float()
    input_img = torch.clamp(input_img, 0, 1)

    if input_img.shape[-1] != size:
        input_img = F.interpolate(
            input_img.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if input_img.shape[0] == 1:
        input_img = input_img.repeat(3, 1, 1)

    input_array = (input_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    imageio.imwrite(input_path, input_array)
    return input_path


def get_optimizer(model, params):
    """Get the optimizer based on the parameter settings"""
    optimizer_params = {
        "params": model.parameters(),
        "lr": params.learning_rate,
    }

    # Adam, Lamb, DiffGrad
    extra_args_common = {
        "betas": getattr(params, "betas", (0.9, 0.999)),
        "eps": getattr(params, "eps", 1e-8),
        "weight_decay": getattr(params, "weight_decay", 0),
    }

    extra_args_adamw = {
        "betas": getattr(params, "betas", (0.9, 0.999)),
        "eps": getattr(params, "eps", 1e-8),
        "weight_decay": getattr(params, "weight_decay", 0.01),
    }

    # SGD
    extra_args_sgd = {
        "momentum": getattr(params, "momentum", 0),
        "dampening": getattr(params, "dampening", 0),
        "weight_decay": getattr(params, "weight_decay", 0),
        "nesterov": getattr(params, "nesterov", False),
    }

    # MADGRAD
    extra_args_madgrad = {
        "momentum": getattr(params, "momentum", 0.9),
        "weight_decay": getattr(params, "weight_decay", 0),
        "eps": getattr(params, "eps", 1e-6),
    }

    optimizers = {
        "adam": (optim.Adam, extra_args_common),
        "adamw": (optim.AdamW, extra_args_adamw),
        "sgd": (optim.SGD, extra_args_sgd),
        "lamb": (jettify_optim.Lamb, extra_args_common),
        "diffgrad": (jettify_optim.DiffGrad, extra_args_common),
        "madgrad": (jettify_optim.MADGRAD, extra_args_madgrad),
    }

    optimizer_cls, extra_args = optimizers.get(params.optim.lower(), (optim.Adam, extra_args_common))
    if params.optim.lower() not in optimizers:
        logger.warning(f"Unsupported optimizer '{params.optim}', using 'Adam' optimizer instead.")
    optimizer = optimizer_cls(**optimizer_params, **extra_args)

    return optimizer
