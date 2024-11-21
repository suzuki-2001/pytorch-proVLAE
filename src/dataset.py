import os
import tarfile
from dataclasses import dataclass

import h5py
import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Dataset configuration class"""

    name: str
    image_size: int
    chn_num: int
    default_path: str

    @classmethod
    def get_config(cls, dataset_name: str) -> "DatasetConfig":
        """Get dataset configuration by name"""
        configs = {
            "mnist": cls("mnist", 28, 1, "./data"),
            "fashionmnist": cls("fashionmnist", 28, 1, "./data"),
            "dsprites": cls("dsprites", 64, 1, "./data"),
            "shapes3d": cls("shapes3d", 64, 3, "./data"),
            "celeba": cls("celeba", 128, 3, "./data"),
            "flowers102": cls("flowers102", 128, 3, "./data"),
            "dtd": cls("dtd", 128, 3, "./data"),
            "imagenet": cls("imagenet", 128, 3, "./data/imagenet"),
            "mpi3d": cls("mpi3d", 64, 3, "./data"),
            "ident3d": cls("ident3d", 128, 3, "./data"),
        }

        if dataset_name not in configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return configs[dataset_name]


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


def download_file(url, filename):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filename, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        logger.success("Dowload complete")
    except HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
    except ConnectionError as e:
        logger.error(f"Connection error occurred: {e}")
    except Timeout as e:
        logger.error(f"Timeout error occurred: {e}")
    except RequestException as e:
        logger.error(f"An error occurred during the request: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(filename) and os.path.getsize(filename) == 0:
            os.remove(filename)
            logger.warning("Incomplete file removed due to download failure")


def dsprites_download(root="./data"):
    """Download dsprites dataset if not present"""
    dsprites_dir = os.path.join(root, "dsprites")
    npz_path = os.path.join(dsprites_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    if not os.path.exists(npz_path):
        os.makedirs(dsprites_dir, exist_ok=True)
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        download_file(url, npz_path)

    return npz_path


def shapes3d_download(root="./data"):
    """Download 3dshapes dataset if not present"""
    shapes_dir = os.path.join(root, "shapes3d")
    h5_path = os.path.join(shapes_dir, "3dshapes.h5")

    # GCS URL for 3dshapes dataset
    url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"

    if not os.path.exists(h5_path):
        os.makedirs(shapes_dir, exist_ok=True)
        logger.info("Downloading 3dshapes dataset...")
        download_file(url, h5_path)
    else:
        logger.info("3dshapes dataset already exists.")

    return h5_path


def mpi3d_download(root="./data", variant="toy"):
    """Download MPI3D dataset if not present"""
    mpi3d_dir = os.path.join(root, "mpi3d")
    os.makedirs(mpi3d_dir, exist_ok=True)

    variants = {
        "toy": {
            "url": "https://storage.googleapis.com/mpi3d_disentanglement_dataset/data/mpi3d_toy.npz",
            "filename": "mpi3d_toy.npz",
        },
        "realistic": {
            "url": "https://storage.googleapis.com/mpi3d_disentanglement_dataset/data/mpi3d_realistic.npz",
            "filename": "mpi3d_realistic.npz",
        },
        "real": {
            "url": "https://storage.googleapis.com/mpi3d_disentanglement_dataset/data/real.npz",
            "filename": "mpi3d_real.npz",
        },
    }

    if variant not in variants:
        raise ValueError(f"Unknown MPI3D variant: {variant}. Choose from {list(variants.keys())}")

    npz_path = os.path.join(mpi3d_dir, variants[variant]["filename"])

    if not os.path.exists(npz_path):
        logger.info(f"Downloading MPI3D {variant} dataset...")
        download_file(variants[variant]["url"], npz_path)

    return npz_path


def ident3d_download(root="./data"):
    """Download 3DIdent dataset if not present"""
    os.makedirs(root, exist_ok=True)

    train_dir = os.path.join(root, "ident3d/train")
    test_dir = os.path.join(root, "ident3d/test")
    train_tar = os.path.join(root, "3dident_train.tar")
    test_tar = os.path.join(root, "3dident_test.tar")

    # Download and extract training data
    if not os.path.exists(train_dir):
        try:
            logger.info("Downloading 3DIdent training dataset...")
            download_file(
                "https://zenodo.org/records/4502485/files/3dident_train.tar?download=1",
                train_tar,
            )
            logger.info("Extracting training dataset...")
            with tarfile.open(train_tar, mode="r") as tar:
                tar.extractall(root)
        finally:
            if os.path.exists(train_tar):
                os.remove(train_tar)

    # Download and extract test data
    if not os.path.exists(test_dir):
        try:
            logger.info("Downloading 3DIdent test dataset...")
            download_file(
                "https://zenodo.org/records/4502485/files/3dident_test.tar?download=1",
                test_tar,
            )
            logger.info("Extracting test dataset...")
            with tarfile.open(test_tar, mode="r") as tar:
                tar.extractall(root)
        finally:
            if os.path.exists(test_tar):
                os.remove(test_tar)

    downloaded_path = os.path.join(root, "3dident")
    if os.path.exists(downloaded_path):
        os.rename(downloaded_path, os.path.join(root, "ident3d"))
    return train_dir, test_dir


class DSpritesDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["imgs"] * 255

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, 0


class DSprites:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("dsprites")
        npz_path = dsprites_download(root)

        dataset = DSpritesDataset(npz_path)

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class MPI3DDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data["images"]

        # dataset variation
        n_images = len(self.images)
        if n_images == 1036800:
            self.dataset_type = "regular"
        elif n_images == 460800:
            self.dataset_type = "complex"
        else:
            raise ValueError(f"Unexpected number of images: {n_images}")

        self.images = self.images.astype(np.float32) / 255.0

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()

        if self.transform:
            if isinstance(image, np.ndarray):
                image = self.transform(image)

        return image, 0


class MPI3D:
    def __init__(self, root="./data", variant="toy", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("mpi3d")
        npz_path = mpi3d_download(root, variant)

        dataset = MPI3DDataset(npz_path)

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class Ident3DDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Root directory
            split (str): 'train' or 'test'
            transform (callable, optional): Transform to apply on the images
        """
        self.root_dir = os.path.join(root_dir, "ident3d", split)
        self.images_dir = os.path.join(self.root_dir, "images")

        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0


class Ident3D:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        train_dir, test_dir = ident3d_download(root=root)
        self.config = DatasetConfig.get_config("ident3d")

        train_dataset = Ident3DDataset(root, split="train")
        test_dataset = Ident3DDataset(root, split="test")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class MNIST:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("mnist")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class FashionMNIST:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("fashionmnist")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class Shapes3DDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, "r") as f:
            self.data = f["images"][:]  # [N, H, W, C]

        self.data = self.data.astype(np.float32) / 255.0
        self.transform = transform
        self.labels = torch.zeros(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # [H, W, C]

        if self.transform:
            image = self.transform(image)  # [C, H, W]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # [C, H, W]

        return image, self.labels[idx]


class Shapes3D:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        h5_path = shapes3d_download(root=root)
        self.config = DatasetConfig.get_config("shapes3d")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        dataset = Shapes3DDataset(h5_path, transform=transform)

        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class CelebA:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("celeba")
        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )

        dataset = torchvision.datasets.CelebA(root=root, split="train", download=True, transform=transform)
        test_dataset = torchvision.datasets.CelebA(root=root, split="test", download=True, transform=transform)

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class Flowers102:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("flowers102")
        transform = transforms.Compose(
            [
                transforms.Resize(146),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.Flowers102(root=root, split="train", download=True, transform=transform)
        testset = torchvision.datasets.Flowers102(root=root, split="test", download=True, transform=transform)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class DTD:
    def __init__(self, root="./data", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("dtd")
        transform = transforms.Compose(
            [
                transforms.Resize(146),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.DTD(root=root, split="train", download=True, transform=transform)
        testset = torchvision.datasets.DTD(root=root, split="test", download=True, transform=transform)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader

    def get_config(self):
        return self.config


class ImageNet:
    def __init__(self, root="./data/imagenet", batch_size=32, num_workers=4):
        self.config = DatasetConfig.get_config("imagenet")
        transform = transforms.Compose(
            [
                transforms.Resize(146),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]
        )

        trainset = torchvision.datasets.ImageNet(root=root, split="train", transform=transform)
        valset = torchvision.datasets.ImageNet(root=root, split="val", transform=transform)

        self.train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_data_loader(self):
        return self.train_loader, self.val_loader

    def get_config(self):
        return self.config
