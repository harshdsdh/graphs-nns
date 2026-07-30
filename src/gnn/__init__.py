"""
run graph neural network code on gpu
"""

import random
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .gcn_model import Net


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


def train(model, optimizer, data) -> list:
    """train a basic GCN model"""
    losses = []
    accuracies = []

    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        model.eval()
        _, pred = model(data).max(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
        accuracies.append(acc)
    return [losses, accuracies]


@torch.no_grad()
def test(model, data) -> list:
    """
    test the model on test masked labels
    """
    accuracies = []
    _, pred = model(data).max(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    accuracies.append(acc)
    return accuracies


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


def main() -> None:
    """init GNN code on graph dataset"""
    print("Hello from graphs-nn!")

    parser = argparse.ArgumentParser(description="run a gnn on a graph dataset")
    parser.add_argument("graph_name", help="name of a graph dataset, such as cora")
    args = parser.parse_args()
    graph_name = args.graph_name

    print(f"graph name: {graph_name}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    print(f"using device: {device}")

    ### load data
    path = f"./src/gnn/grasp_data/{graph_name}/processed_data.pt"
    print(f"path:{path}")
    data = torch.load(path, weights_only=False)
    data = data.to(device)
    print(data.label_texts)

    model = Net(data.num_node_features, data.num_classes).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-3)

    losses, acc = train(model, optimizer, data)

    plot_training_metrics(losses, acc)

    temp = [test(model, data) for _ in range(10)]
    print(f"mean test results: {np.mean(temp)}, std dev: {np.std(temp)}")
