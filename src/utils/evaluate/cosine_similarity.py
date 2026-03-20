import torch
import torch.nn.functional as F
from torch import Tensor


def compute_cosine_similarity(vectors: tuple[Tensor], dimension=1) -> float:
    """compute cosine similarity between two tensors along the given dimension."""
    return float(F.cosine_similarity(vectors[0], vectors[1], dim=dimension))


if __name__ == "__main__":
    vector_1 = torch.randn(1, 128)
    vector_2 = torch.randn(1, 128)
    print(compute_cosine_similarity((vector_1, vector_2)))
