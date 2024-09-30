from typing import Sequence

import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim."""
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim."""
    data = np.random.rand(dim)
    row = np.arange(dim)
    col = np.zeros(dim, dtype=int)
    return sparse.coo_matrix((data, (row, col)), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition."""
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar."""
    return a * x


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors."""
    return sum(c * v for c, v in zip(coeffs, vectors))


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product."""
    return np.dot(x.T, y).item()


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max."""
    return np.linalg.norm(x, ord=order)


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors."""
    return np.linalg.norm(x - y)


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between vectors in degrees."""
    cos_theta = dot_product(x, y) / (norm(x, 2) * norm(y, 2))
    return np.degrees(np.arccos(cos_theta))


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check if vectors are orthogonal."""
    return np.isclose(dot_product(x, y), 0.0)


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations."""
    return np.linalg.solve(a, b)