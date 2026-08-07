"""
utility functions for the repo
"""

import torch
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    """
    switch between mps, cpu and cuda
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def plot_training_metrics(l: list, a: list) -> None:
    """
    plot training loss ans validation acc plots
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(l, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(a, label="Validation Accuracy")
    ax2.set_ylabel("Accuracy")

    plt.title("Training Loss and Validation Accuracy")
    plt.show()
