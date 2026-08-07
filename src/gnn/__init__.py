"""
run graph neural network code on gpu
"""

import random
import argparse
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F

from model import Net
from utilities import get_device
from load_emb import generate_node_text_llm_embs, model_dict

HIDDEN_DIM = 16


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
    path = Path(f"./src/gnn/grasp_data/{graph_name}/processed_data.pt")
    if path.is_file():
        print(f"{graph_name} dataset available")

    print(f"path:{path}")
    data = torch.load(path, weights_only=False)
    data = data.to(device)
    print(data.label_texts)
    original_text = data.raw_texts
    for i, m in enumerate(model_dict.keys()):
        text = original_text
        print(f"case {i}: {m} LLM model using {model_dict[m]} embedding")
        emb = generate_node_text_llm_embs(text, graph_name, model_dict[m])

        print(f"Nodes: {data.num_nodes}")
        print(f"Embedding dimension: {data.x.shape[1]}")
        print(f"Feature matrix: {data.x.shape}")

        data = data.to(device)
        data.x = emb

        model = Net(data.num_node_features, HIDDEN_DIM, data.num_classes).to(device)
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=0.01, weight_decay=5e-3
        )
        train(model, optimizer, data)

        temp = [test(model, data) for _ in range(10)]
        print(f"mean test results: {np.mean(temp)}, std dev: {np.std(temp)}")

    print("process complete")
