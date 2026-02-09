import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=config["data"]["root"],
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=config["data"]["root"],
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False
    )

    return train_loader, test_loader