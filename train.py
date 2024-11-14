import argparse
import os
import time
from dataclasses import dataclass, field

import imageio.v3 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as jettify_optim
import torchvision
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import DTD, MNIST, MPI3D, CelebA, DSprites, FashionMNIST, Flowers102, Ident3D, ImageNet, Shapes3D
from models import ProVLAE


@dataclass
class HyperParameters:
    dataset: str = field(default="shapes3d")
    data_path: str = field(default="./data")
    z_dim: int = field(default=3)
    num_ladders: int = field(default=3)
    beta: float = field(default=8.0)
    learning_rate: float = field(default=5e-4)
    fade_in_duration: int = field(default=5000)
    image_size: int = field(default=64)
    chn_num: int = field(default=3)
    train_seq: int = field(default=1)
    batch_size: int = field(default=100)
    num_epochs: int = field(default=1)
    mode: str = field(default="seq_train")
    hidden_dim: int = field(default=32)
    coff: float = field(default=0.5)
    output_dir: str = field(default="outputs")

    # pytorch optimization
    compile_mode: str = field(default="default")  # or max-autotune-no-cudagraphs
    on_cudnn_benchmark: bool = field(default=True)
    optim: str = field(default="adam")

    @property
    def checkpoint_path(self):
        return os.path.join(self.output_dir, f"checkpoints/model_seq{self.train_seq}.pt")

    @property
    def recon_path(self):
        return os.path.join(self.output_dir, f"reconstructions/recon_seq{self.train_seq}.png")

    @property
    def traverse_path(self):
        return os.path.join(self.output_dir, f"traversals/traverse_seq{self.train_seq}.gif")

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        for field_info in cls.__dataclass_fields__.values():
            parser.add_argument(f"--{field_info.name}", type=field_info.type, default=field_info.default)
        return cls(**vars(parser.parse_args()))


def get_dataset(params):
    """Load the dataset and return the data loader"""
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
        raise ValueError(f"Unknown dataset: {params.dataset}. " f"Available datasets: {list(dataset_classes.keys())}")

    dataset_class = dataset_classes[params.dataset]
    if params.dataset == "mpi3d":
        # mpi3d variants: toy, real
        variant = getattr(params, "mpi3d_variant", "toy")
        dataset = dataset_class(
            root=params.data_path,
            variant=variant,
            batch_size=params.batch_size,
            num_workers=4,
        )
    else:  # other dataset
        dataset = dataset_class(root=params.data_path, batch_size=params.batch_size, num_workers=4)

    config = dataset.get_config()
    params.chn_num = config.chn_num
    params.image_size = config.image_size

    logger.success("Dataset loaded.")
    return dataset.get_data_loader()


def exec_time(func):
    """Decorates a function to measure its execution time in hours and minutes."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        logger.success(f"Training completed ({int(execution_time // 3600)}h {int((execution_time % 3600) // 60)}min)")
        return result

    return wrapper


def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    """
    Load a model checkpoint to resume training or run further inference.
    """
    torch.serialization.add_safe_globals([set])
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        f"Loaded checkpoint from '{checkpoint_path}' (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})"
    )

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


def create_latent_traversal(model, data_loader, save_path, device, params):
    """Create and save organized latent traversal GIF with optimized layout"""
    model.eval()
    model.fade_in = 1.0
    with torch.no_grad():
        # Get a single batch of images
        inputs, _ = next(iter(data_loader))
        inputs = inputs[0:1].to(device)

        # Get latent representations
        with torch.amp.autocast(device_type="cuda", enabled=False):
            latent_vars = [z[0] for z in model.inference(inputs)]

        # Traverse values
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


@exec_time
def train_model(model, data_loader, optimizer, params, device, scaler=None, autocast_dtype=torch.float16):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    global_step = 0

    logger.info("Start training.")
    for epoch in range(params.num_epochs):
        with tqdm(
            enumerate(data_loader),
            desc=f"Current epoch [{epoch + 1}/{params.num_epochs}]",
            leave=False,
            total=len(data_loader),
        ) as pbar:
            for batch_idx, (inputs, _) in pbar:
                inputs = inputs.to(device, non_blocking=True)

                # Forward pass with autocast
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    x_recon, loss, latent_loss, recon_loss = model(inputs, step=global_step)

                # Backward pass with appropriate scaling
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                pbar.set_postfix(
                    total_loss=f"{loss.item():.2f}",
                    latent_loss=f"{latent_loss:.2f}",
                    recon_loss=f"{recon_loss:.2f}",
                )
                global_step += 1

        # Save checkpoints and visualizations
        os.makedirs(os.path.dirname(params.checkpoint_path), exist_ok=True)

        # Model evaluation for visualizations
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype):
            save_reconstruction(inputs, x_recon, params.recon_path)
            create_latent_traversal(model, data_loader, params.traverse_path, device, params)
        model.train()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "loss": loss.item(),
        }
        torch.save(checkpoint, params.checkpoint_path)

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch: [{epoch+1}/{params.num_epochs}], Loss: {loss.item():.2f}")


def get_optimizer(model, params):
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "lamb": jettify_optim.Lamb,
        "diffgrad": jettify_optim.DiffGrad,
        "madgrad": jettify_optim.MADGRAD,
    }

    optimizer = optimizers.get(params.optim.lower())

    if optimizer is None:
        optimizer = optimizers.get("adam")
        logger.warning(f"unsupported optimizer {params.optim}, use Adam optimizer.")

    return optimizer(model.parameters(), lr=params.learning_rate)


def main():
    # Setup
    params = HyperParameters.from_args()

    # gpu config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if params.on_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if torch.cuda.get_device_capability()[0] >= 8:
        if not torch.cuda.is_bf16_supported():
            logger.warning("BF16 is not supported, falling back to FP16")
            autocast_dtype = torch.float16
            scaler = torch.amp.GradScaler()
        else:  # BF16
            autocast_dtype = torch.bfloat16
            scaler = None
            logger.debug("Using BF16 mixed precision")
    else:  # FP16
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler()
        logger.debug("Using FP16 mixed precision with gradient scaling")

    os.makedirs(params.output_dir, exist_ok=True)
    train_loader, test_loader = get_dataset(params)

    # Initialize model
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
    ).to(device)

    optimizer = get_optimizer(model, params)
    model = torch.compile(model, mode=params.compile_mode)

    # Train model or visualize traverse
    if params.mode == "seq_train":
        logger.opt(colors=True).info(f"<blue>✓ Mode: sequential execution [progress 1 >> {params.num_ladders}]</blue>")
        for i in range(1, params.num_ladders + 1):
            params.train_seq, model.train_seq = i, i
            if params.train_seq >= 2:
                prev_checkpoint = os.path.join(params.output_dir, f"checkpoints/model_seq{params.train_seq-1}.pt")
                if os.path.exists(prev_checkpoint):
                    model, optimizer, scaler = load_checkpoint(model, optimizer, scaler, prev_checkpoint)
            train_model(model, train_loader, optimizer, params, device, scaler, autocast_dtype)

    elif params.mode == "indep_train":
        logger.opt(colors=True).info(f"<blue>✓ Mode: independent execution [progress {params.train_seq}]</blue>")
        if params.train_seq >= 2:
            prev_checkpoint = os.path.join(params.output_dir, f"checkpoints/model_seq{params.train_seq-1}.pt")
            if os.path.exists(prev_checkpoint):
                model, optimizer, scaler = load_checkpoint(model, optimizer, scaler, prev_checkpoint)

        train_model(model, train_loader, optimizer, params, device, scaler, autocast_dtype)

    elif params.mode == "visualize":
        logger.opt(colors=True).info(f"<blue>✓ Mode: visualize latent traversing [progress {params.train_seq}]</blue>")
        current_checkpoint = os.path.join(params.output_dir, f"checkpoints/model_seq{params.train_seq}.pt")
        if os.path.exists(current_checkpoint):
            model, _, _ = load_checkpoint(model, optimizer, scaler, current_checkpoint)
        create_latent_traversal(model, test_loader, params.traverse_path, device, params)
        logger.success("Latent traversal visualization saved.")

    else:
        logger.error(f"unsupported mode: {params.mode}, use 'train' or 'visualize'.")


if __name__ == "__main__":
    main()
