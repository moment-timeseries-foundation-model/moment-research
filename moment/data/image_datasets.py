# Code taken from:
# [1] https://github.com/kzl/universal-computation/blob/master/universal_computation/datasets/cifar10_gray.py
# [2] https://github.com/kzl/universal-computation/blob/master/universal_computation/datasets/cifar10.py

import os

import torch
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data import DataLoader

from moment.common import PATHS

from .base import FPTDataset


class CIFAR10GrayDataset(FPTDataset):
    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int = 4,
        data_aug: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size  # grid of (patch_size x patch_size)

        if data_aug:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.d_train = DataLoader(
            torchvision.datasets.CIFAR10(
                os.path.join(PATHS.DATA_DIR, "FPT_datasets/CIFAR"),
                download=True,
                train=True,
                transform=transform,
            ),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        self.d_test = DataLoader(
            torchvision.datasets.CIFAR10(
                os.path.join(PATHS.DATA_DIR, "FPT_datasets/CIFAR"),
                download=True,
                train=False,
                transform=val_transform,
            ),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.test_enum = enumerate(self.d_test)
                _, (x, y) = next(self.test_enum)

        if self.patch_size is not None:
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            )

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y


class CIFAR10Dataset(FPTDataset):
    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int = 4,
        data_aug: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size  # grid of (patch_size x patch_size)

        if data_aug:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.d_train = DataLoader(
            torchvision.datasets.CIFAR10(
                os.path.join(PATHS.DATA_DIR, "FPT_datasets/CIFAR"),
                download=True,
                train=True,
                transform=transform,
            ),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        self.d_test = DataLoader(
            torchvision.datasets.CIFAR10(
                os.path.join(PATHS.DATA_DIR, "FPT_datasets/CIFAR"),
                download=True,
                train=False,
                transform=val_transform,
            ),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

        self.train_size = len(self.d_train)
        self.test_size = len(self.d_test)

    def reset_test(self):
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.test_enum = enumerate(self.d_test)
                _, (x, y) = next(self.test_enum)

        if self.patch_size is not None:
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            )

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y


class MNISTDataset(FPTDataset):
    def __init__(self, batch_size, patch_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size  # we fix it so we can use dataloader
        self.patch_size = patch_size  # grid of (patch_size x patch_size)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.0, std=1.0),
                transforms.Pad(padding=2, fill=0, padding_mode="constant"),
                # MNIST dataset is 28 x 28 but it is frequently padded to 32 x 32
            ]
        )

        self.d_train = DataLoader(
            torchvision.datasets.MNIST(
                os.path.join(PATHS.DATA_DIR, "FPT_datasets/MNIST"),
                download=True,
                train=True,
                transform=transform,
            ),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        self.d_test = DataLoader(
            torchvision.datasets.MNIST(
                os.path.join(PATHS.DATA_DIR, "FPT_datasets/MNIST"),
                download=True,
                train=False,
                transform=transform,
            ),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        self.train_enum = enumerate(self.d_train)
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.test_enum = enumerate(self.d_test)
                _, (x, y) = next(self.test_enum)

        if self.patch_size is not None:
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            )

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
