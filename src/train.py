import argparse
import os
import sys
from dataclasses import dataclass, field

import imageio.v3 as imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as jettify_optim
import torchvision
import wandb
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import DTD, MNIST, MPI3D, CelebA, DSprites, FashionMNIST, Flowers102, Ident3D, ImageNet, Shapes3D
from ddp_utils import cleanup_distributed, setup_distributed, setup_logger
from provlae import ProVLAE
from utils import add_dataclass_args, exec_time


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


def get_dataset(params, logger):
    """Load dataset with distributed support"""
    dataset_classes = {
        "mnist": MNIST,
        "fashionmnist": FashionMNIST,
        "shapes3d": Shapes3D,
        "dsprites": DSprites,
        "celeba": CelebA,
        "flowers102": Flowers102,
        "dtd": DTD,
        "imagenet": ImageNet,
        "mpi3d": MPI3D,
        "ident3d": Ident3D,
    }

    if params.dataset not in dataset_classes:
        raise ValueError(f"Unknown dataset: {params.dataset}")

    dataset_class = dataset_classes[params.dataset]

    try:
        if params.dataset == "mpi3d":
            variant = getattr(params, "mpi3d_variant", "toy")
            dataset = dataset_class(root=params.data_path, batch_size=params.batch_size, num_workers=4, variant=variant)
        else:
            dataset = dataset_class(root=params.data_path, batch_size=params.batch_size, num_workers=4)

        config = dataset.get_config()
        params.chn_num = config.chn_num
        params.image_size = config.image_size

        train_loader, test_loader = dataset.get_data_loader()
        if params.distributed:
            train_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=params.world_size,
                rank=params.local_rank,
                shuffle=True,
                drop_last=True,
            )

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=params.batch_size,
                sampler=train_sampler,
                num_workers=params.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )

            if params.local_rank == 0:
                logger.info(f"Dataset {params.dataset} loaded with distributed sampler")
        else:
            logger.info(f"Dataset {params.dataset} loaded")

        return train_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


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


def create_latent_traversal(model, data_loader, save_path, device, params):
    """Create and save organized latent traversal GIF with optimized layout"""
    model.eval()

    if hasattr(model, "module"):
        model = model.module

    model.fade_in = 1.0
    with torch.no_grad():
        inputs, _ = next(iter(data_loader))  # Get a single batch of images
        inputs = inputs[0:1].to(device)

        input_path = save_input_image(inputs.cpu(), os.path.join(params.output_dir, params.input_dir), params.train_seq)

        # Get latent representations
        with torch.amp.autocast(device_type="cuda", enabled=False):
            latent_vars = [z[0] for z in model.inference(inputs)]

        traverse_range = torch.linspace(-1.5, 1.5, 15).to(device)

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
    else:
        model.to(device)

    model.train()
    global_step = 0

    logger.info(f"Start training [progress {params.train_seq}]")
    for epoch in range(params.num_epochs):
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
                    x_recon, loss, latent_loss, recon_loss = model(inputs, step=global_step)

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
                        total_loss=f"{loss.item():.2f}",
                        latent_loss=f"{latent_loss:.2f}",
                        recon_loss=f"{recon_loss:.2f}",
                    )

                if params.use_wandb and params.distributed:
                    metrics = {
                        f"loss/rank_{params.local_rank}": loss.item(),
                        f"latent_loss/rank_{params.local_rank}": latent_loss.item(),
                        f"recon_loss/rank_{params.local_rank}": recon_loss.item(),
                    }

                    all_metrics = [None] * params.world_size
                    dist.all_gather_object(all_metrics, metrics)

                    if params.local_rank == 0:
                        combined_metrics = {}
                        for rank_metrics in all_metrics:
                            combined_metrics.update(rank_metrics)
                        wandb.log(combined_metrics, step=global_step)
                elif params.use_wandb:
                    metrics = {"loss": loss.item(), "latent_loss": latent_loss.item(), "recon_loss": recon_loss.item()}
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

            # reconstruction and traversal images
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


def init_wandb(params, hash):
    if params.use_wandb:
        if wandb.run is not None:
            wandb.finish()

        run_id = None
        if params.local_rank == 0:
            logger.debug(f"Current run ID: {hash}")
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


def main():
    params = parse_arguments()

    try:
        # Setup distributed training
        is_distributed = setup_distributed(params)
        rank = params.local_rank if is_distributed else 0
        world_size = params.world_size if is_distributed else 1

        # Setup device and logger
        device = torch.device(f"cuda:{params.local_rank}" if is_distributed else "cuda")
        logger = setup_logger(rank, world_size)

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
                logger.opt(colors=True).info(
                    f"<blue>✅ Mode: sequential execution [progress 1 >> {params.num_ladders}]</blue>"
                )

            for i in range(1, params.num_ladders + 1):
                if is_distributed:
                    torch.cuda.synchronize()
                    dist.barrier()

                # Update sequence number
                if is_distributed:
                    params.train_seq = i
                    model.module.train_seq = i
                else:
                    params.train_seq = i
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
                logger.opt(colors=True).info(
                    f"<blue>✅ Mode: independent execution [progress {params.train_seq}]</blue>"
                )

            if is_distributed:
                torch.cuda.synchronize()
                dist.barrier()

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
            logger.opt(colors=True).info(f"<blue>✅ Mode: traverse execution [progress 1 {params.num_ladders}]</blue>")
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
