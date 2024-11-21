import argparse
import os
from dataclasses import dataclass, field

import math
import imageio.v3 as imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import wandb
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from provlae import ProVLAE
from dataset import get_dataset
from ddp_utils import cleanup_distributed, setup_distributed, setup_logger
from utils import (
    init_wandb,
    get_optimizer,
    add_dataclass_args,
    exec_time,
    save_input_image,
    save_reconstruction,
    load_checkpoint,
)


@dataclass
class OptimizerParameters:
    betas: tuple = field(default=(0.9, 0.999))
    eps: float = field(default=1e-08)
    weight_decay: float = field(default=0)

    momentum: float = field(default=0)  # sgd, madgrad
    dampening: float = field(default=0)  # sgd


@dataclass
class HyperParameters:
    z_dim: int = field(default=3)
    num_ladders: int = field(default=3)
    beta: float = field(default=8.0)
    learning_rate: float = field(default=5e-4)
    fade_in_duration: int = field(default=5000)
    image_size: int = field(default=64)
    chn_num: int = field(default=3)
    batch_size: int = field(default=100)
    num_epochs: int = field(default=1)
    mode: str = field(default="seq_train")
    train_seq: int = field(default=1)  # progress stage
    hidden_dim: int = field(default=32)
    coff: float = field(default=0.5)
    pre_kl: bool = field(default=True)
    use_kl_annealing: bool = field(default=False)
    kl_annealing_mode: str = field(default="linear")
    cycle_period: int = field(default=4)
    max_kl_weight: float = field(default=1.0)
    min_kl_weight: float = field(default=0.1)
    ratio: float = field(default=1.0)
    use_capacity_increase: bool = field(default=False)
    gamma: float = field(default=1000.0)
    max_capacity: int = field(default=25)
    capacity_max_iter: float = field(default=1e-5)


@dataclass
class TrainingParameters:
    dataset: str = field(default="shapes3d")
    data_path: str = field(default="./data")
    num_workers: int = field(default=4)  # data loader

    # output dirs
    output_dir: str = field(default="output")  # results dir
    checkpoint_dir: str = field(default="checkpoints")
    recon_dir: str = field(default="reconstructions")
    traverse_dir: str = field(default="traversals")
    input_dir: str = field(default="inputs")

    # PyTorch optimization
    compile_mode: str = field(default="default")  # or max-autotune-no-cudagraphs
    on_cudnn_benchmark: bool = field(default=True)
    optim: str = field(default="adam")

    # Distributed training parameters
    distributed: bool = field(default=False)
    local_rank: int = field(default=-1)
    world_size: int = field(default=1)
    dist_backend: str = field(default="nccl")
    dist_url: str = field(default="env://")

    # wandb
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="provlae")


def parse_arguments():
    parser = argparse.ArgumentParser()
    add_dataclass_args(parser, HyperParameters)
    add_dataclass_args(parser, OptimizerParameters)
    add_dataclass_args(parser, TrainingParameters)

    return parser.parse_args()


def create_latent_traversal(model, data_loader, save_path, device, params):
    """Create and save organized latent traversal GIF with optimized layout"""
    model.eval()

    if hasattr(model, "module"):
        model = model.module

    model.fade_in = 1.0
    with torch.no_grad():
        inputs, _ = next(iter(data_loader))  # Get a single batch of images
        inputs = inputs[0:1].to(device)

        # save traverse inputs
        input_path = save_input_image(
            inputs.cpu(), os.path.join(params.output_dir, params.input_dir), params.train_seq, params.image_size
        )

        # Get latent representations
        with torch.amp.autocast(device_type="cuda", enabled=False):
            latent_vars = [z[0] for z in model.inference(inputs)]

        traverse_range = torch.linspace(-2.5, 2.5, 10).to(device)

        # Image layout parameters
        img_size = 96  # Base image size
        padding = 1  # Reduced padding between images
        label_margin = 1  # Margin for labels inside images
        font_size = 7  # Smaller font size for better fit

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        frames = []
        for t_idx in range(len(traverse_range)):
            current_images = []

            # Generate images for each ladder and dimension
            for ladder_idx in range(len(latent_vars) - 1, -1, -1):
                for dim in range(latent_vars[ladder_idx].shape[1]):
                    z_mod = [v.clone() for v in latent_vars]
                    z_mod[ladder_idx][0, dim] = traverse_range[t_idx]

                    with torch.amp.autocast(device_type="cuda", enabled=False):
                        gen_img = model.generate(z_mod)
                    img = gen_img[0].cpu().float()
                    img = torch.clamp(img, 0, 1)

                    # Resize image if needed
                    if img.shape[-1] != img_size:
                        img = F.interpolate(
                            img.unsqueeze(0),
                            size=img_size,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)

                    # Handle both single-channel and multi-channel images
                    if img.shape[0] == 1:
                        # Single channel (grayscale) - repeat to create RGB
                        img = img.repeat(3, 1, 1)

                    # Convert to PIL Image for adding text
                    img_array = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array)
                    draw = ImageDraw.Draw(img_pil)

                    # Add label inside the image
                    label = f"L{len(latent_vars)-ladder_idx} z{dim+1}"
                    # draw black and white text to create border effect
                    for offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        draw.text(
                            (label_margin + offset[0], label_margin + offset[1]),
                            label,
                            (0, 0, 0),
                            font=font,
                        )
                    draw.text((label_margin, label_margin), label, (255, 255, 255), font=font)

                    # Add value label to bottom-left image
                    if ladder_idx == 0 and dim == 0:
                        value_label = f"v = {traverse_range[t_idx].item():.2f}"
                        for offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            draw.text(
                                (
                                    label_margin + offset[0],
                                    img_size - font_size - label_margin + offset[1],
                                ),
                                value_label,
                                (0, 0, 0),
                                font=font,
                            )
                        draw.text(
                            (label_margin, img_size - font_size - label_margin),
                            value_label,
                            (255, 255, 255),
                            font=font,
                        )

                    # Convert back to tensor
                    img_tensor = torch.tensor(np.array(img_pil)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    current_images.append(img_tensor)

            # Convert to tensor and create grid
            current_images = torch.stack(current_images)

            # Create grid with minimal padding
            grid = torchvision.utils.make_grid(current_images, nrow=params.z_dim, padding=padding, normalize=True)

            grid_array = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames.append(grid_array)

        # Save GIF with infinite loop
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # duration=200 means 5 FPS, loop=0 means infinite loop
        imageio.imwrite(save_path, frames, duration=200, loop=0, format="GIF", optimize=False)

        return input_path


@exec_time
def train_model(model, data_loader, optimizer, params, device, logger, scaler=None, autocast_dtype=torch.float16):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if hasattr(model, "module"):
        model.module.to(device)
        model.module.num_epochs = params.num_epochs
    else:
        model.to(device)

    model.train()
    global_step = 0

    logger.info(f"Start training [progress {params.train_seq}]")
    for epoch in range(params.num_epochs):
        if hasattr(model, "module"):
            model.module.current_epoch = epoch
        else:
            model.current_epoch = epoch

        if params.distributed:
            data_loader.sampler.set_epoch(epoch)

        with tqdm(
            enumerate(data_loader),
            desc=f"Current epoch [{epoch + 1}/{params.num_epochs}]",
            leave=False,
            total=len(data_loader),
            disable=params.distributed and params.local_rank != 0,
        ) as pbar:
            for batch_idx, (inputs, _) in pbar:
                inputs = inputs.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    x_recon, loss, latent_loss, recon_loss, kl_weight = model(inputs, step=global_step)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if params.local_rank == 0 or not params.distributed:  # Only show progress on main process
                    pbar.set_postfix(
                        total_loss=f"{loss.item():.5f}",
                        latent_loss=f"{latent_loss.item():.5f}",
                        recon_loss=f"{recon_loss.item():.5f}",
                    )

                if params.use_wandb and params.distributed:
                    metrics = {
                        f"ELBO/rank_{params.local_rank}": loss.item(),
                        f"KL Term/rank_{params.local_rank}": latent_loss.item(),
                        f"Reconstruction Error/rank_{params.local_rank}": recon_loss.item(),
                        f"KL Weight/rank_{params.local_rank}": kl_weight.item(),
                    }

                    all_metrics = [None] * params.world_size
                    dist.all_gather_object(all_metrics, metrics)

                    if params.local_rank == 0:
                        combined_metrics = {}
                        for rank_metrics in all_metrics:
                            combined_metrics.update(rank_metrics)
                        wandb.log(combined_metrics, step=global_step)
                elif params.use_wandb:
                    metrics = {
                        "ELBO": loss.item(),
                        "KL Term": latent_loss.item(),
                        "Reconstruction Error": recon_loss.item(),
                        "KL Weight": kl_weight.item(),
                    }
                    wandb.log(metrics, step=global_step)

                global_step += 1

        # Save checkpoints and visualizations only on main process
        checkpoint_path = os.path.join(params.output_dir, params.checkpoint_dir, f"model_seq{params.train_seq}.pt")
        if params.local_rank == 0 or not params.distributed:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
                recon_path = os.path.join(params.output_dir, params.recon_dir, f"recon_seq{params.train_seq}.png")
                traverse_path = os.path.join(
                    params.output_dir, params.traverse_dir, f"traverse_seq{params.train_seq}.gif"
                )

                save_reconstruction(inputs, x_recon, recon_path)
                input_path = create_latent_traversal(model, data_loader, traverse_path, device, params)

            # reconstruction and traversal images (Media)
            if params.use_wandb and (params.local_rank == 0 or not params.distributed):
                wandb.log(
                    {
                        "Reconstruction": wandb.Image(recon_path),
                        "Traversal": wandb.Image(traverse_path),
                        "Input": wandb.Image(input_path),
                    },
                    step=global_step,
                )

            model.train()

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict() if params.distributed else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "loss": loss.item(),
            }
            torch.save(checkpoint, checkpoint_path)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch: [{epoch+1}/{params.num_epochs}], Loss: {loss.item():.2f}")


def main():
    params = parse_arguments()  # hyperparameter and training config

    """TODO: fix random seed"""

    try:
        is_distributed = setup_distributed(params)  # Setup distributed training
        rank = params.local_rank if is_distributed else 0
        world_size = params.world_size if is_distributed else 1
        device = torch.device(f"cuda:{params.local_rank}" if is_distributed else "cuda")
        logger = setup_logger(rank, world_size)  # ddp logger

        torch.set_float32_matmul_precision("high")
        if params.on_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # Mixed precision setup
        if torch.cuda.get_device_capability()[0] >= 8:
            if not torch.cuda.is_bf16_supported():
                logger.warning("BF16 is not supported, falling back to FP16")
                autocast_dtype = torch.float16
                scaler = torch.amp.GradScaler()
            else:
                autocast_dtype = torch.bfloat16
                scaler = None
                logger.debug("Using BF16 mixed precision")
        else:
            autocast_dtype = torch.float16
            scaler = torch.amp.GradScaler()
            logger.debug("Using FP16 mixed precision with gradient scaling")

        # Create output directory
        if rank == 0:
            os.makedirs(params.output_dir, exist_ok=True)

        if is_distributed:
            dist.barrier()

        # Get dataset and model
        train_loader, test_loader = get_dataset(params, logger)
        model = ProVLAE(
            z_dim=params.z_dim,
            beta=params.beta,
            learning_rate=params.learning_rate,
            fade_in_duration=params.fade_in_duration,
            chn_num=params.chn_num,
            train_seq=params.train_seq,
            image_size=params.image_size,
            num_ladders=params.num_ladders,
            hidden_dim=params.hidden_dim,
            coff=params.coff,
            pre_kl=params.pre_kl,
            use_kl_annealing=params.use_kl_annealing,
            kl_annealing_mode=params.kl_annealing_mode,
            cycle_period=params.cycle_period,
            max_kl_weight=params.max_kl_weight,
            min_kl_weight=params.min_kl_weight,
            ratio=params.ratio,
            use_capacity_increase=params.use_capacity_increase,
            gamma=params.gamma,
            max_capacity=params.max_capacity,
            capacity_max_iter=params.capacity_max_iter,
        ).to(device)

        if is_distributed:
            model = DDP(
                model,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=True,
            )
            torch.cuda.synchronize()
            dist.barrier()

        optimizer = get_optimizer(model, params)
        if not is_distributed:
            model = torch.compile(model, mode=params.compile_mode)
            logger.debug("model compiled")

        # Training mode selection
        if params.mode == "seq_train":
            if rank == 0:
                logger.opt(colors=True).info(f"✅ Mode: sequential execution [progress 1 >> {params.num_ladders}]")

            for i in range(1, params.num_ladders + 1):
                if is_distributed:
                    torch.cuda.synchronize()
                    dist.barrier()

                # Update sequence number
                params.train_seq = i
                if is_distributed:
                    model.module.train_seq = i
                else:
                    model.train_seq = i

                if params.use_wandb:
                    hash_str = os.urandom(8).hex().upper()
                    init_wandb(params, hash_str)

                # Load checkpoint if needed
                if params.train_seq >= 2:
                    prev_checkpoint = os.path.join(
                        params.output_dir, params.checkpoint_dir, f"model_seq{params.train_seq-1}.pt"
                    )
                    if os.path.exists(prev_checkpoint):
                        model, optimizer, scaler = load_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            checkpoint_path=prev_checkpoint,
                            device=device,
                            logger=logger,
                        )

                        if is_distributed:
                            torch.cuda.synchronize()
                            dist.barrier()

                # Training
                train_model(
                    model=model,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    params=params,
                    device=device,
                    logger=logger,
                    scaler=scaler,
                    autocast_dtype=autocast_dtype,
                )

                if params.use_wandb:
                    wandb.finish()
                if is_distributed:
                    torch.cuda.synchronize()
                    dist.barrier()

        elif params.mode == "indep_train":
            logger.info(f"Current trainig progress >> {params.train_seq}")
            if rank == 0:
                logger.opt(colors=True).info(f"✅ Mode: independent execution [progress {params.train_seq}]")

            if is_distributed:
                torch.cuda.synchronize()
                dist.barrier()

            if params.use_wandb:
                hash_str = os.urandom(8).hex().upper()
                init_wandb(params, hash_str)

            # Load checkpoint if needed
            if params.train_seq >= 2:
                prev_checkpoint = os.path.join(
                    params.output_dir, params.checkpoint_dir, f"model_seq{params.train_seq-1}.pt"
                )
                if os.path.exists(prev_checkpoint):
                    model, optimizer, scaler = load_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        checkpoint_path=prev_checkpoint,
                        device=device,
                        logger=logger,
                    )

                    if is_distributed:
                        torch.cuda.synchronize()
                        dist.barrier()

            # Training
            train_model(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                params=params,
                device=device,
                logger=logger,
                scaler=scaler,
                autocast_dtype=autocast_dtype,
            )

            if is_distributed:
                torch.cuda.synchronize()
                dist.barrier()

        elif params.mode == "traverse":
            logger.opt(colors=True).info(f"✅ Mode: traverse execution [progress 1 {params.num_ladders}]")
            try:
                model, optimizer, scaler = load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    checkpoint_path=os.path.join(
                        params.output_dir, params.checkpoint_dir, f"model_seq{params.train_seq}.pt"
                    ),
                    device=device,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"Load checkpoint failed: {str(e)}")

            traverse_path = os.path.join(params.output_dir, params.traverse_dir, f"traverse_seq{params.train_seq}.gif")
            create_latent_traversal(model, test_loader, traverse_path, device, params)
            logger.success("Traverse compelted")
        else:
            logger.error(f"Unsupported mode: {params.mode}, use 'seq_train' or 'indep_train'")
            return

    except KeyboardInterrupt as e:
        logger.opt(colors=True).error("<red>Keyboard interupt</red>")

    except Exception as e:
        logger.opt(colors=True).exception(f"<red>Training failed: {str(e)}</red>")

    finally:
        if not is_distributed:
            logger.info("no resources clean up (is_distributed=False)")
        else:
            try:
                if is_distributed:
                    torch.cuda.synchronize()
                    cleanup_distributed()
                    logger.info("Distributed resources cleaned up successfully")
            except Exception as e:
                logger.opt(colors=True).exception(f"<red>Error during cleanup</red>: {e}")


if __name__ == "__main__":
    main()
