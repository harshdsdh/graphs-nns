"""
load stored embeddings if present else load from HF
"""

from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from .utilities import get_device

model_dict = {
    "qwen_small_language_model": "Qwen/Qwen3-Embedding-0.6B",
    "qwen_large_language_model": "Qwen/Qwen3-Embedding-8B",
}


def load_embs_from_hf(cleaned_texts: list, model_type: str, batch_size=16) -> any:
    """
    load embs from hf
    """
    device = get_device()
    encoder = SentenceTransformer(model_dict[model_type], device=device)
    embeddings = encoder.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    return embeddings.cpu().float()


def generate_node_text_llm_embs(texts: list, graph_name: str, model_type: str):
    """load embeddings for text data present in graph nodes"""
    path = Path(f"./src/gnn/grasp_data/{graph_name}/{model_type}.pt")

    if path.is_file():
        print("saved embedding found. loading saved embedding")
        print(f"model type: {model_type}")
        emb = torch.load(path, weights_only=False)
    else:
        cleaned_texts = ["" if text is None else str(text) for text in texts]
        emb = load_embs_from_hf(cleaned_texts, model_type)
    return emb
