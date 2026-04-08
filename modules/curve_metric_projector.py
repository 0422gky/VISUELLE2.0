"""
Frozen GTM embeddings -> small MLP -> L2-normalized space for curve-similarity alignment.
Optional PCA on train embeddings only (fit on train, transform train/test).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class CurveMetricProjector(nn.Module):
    """
    Linear(D -> 128) -> ReLU -> Linear(128 -> out_dim), then L2 normalize per row.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1, eps=1e-8)


def fit_pca_on_train(
    embeddings_train: np.ndarray,
    n_components: int | None,
    random_state: int = 0,
) -> PCA | None:
    """
    Fit PCA on train embeddings only. Returns None if n_components is None or <= 0.
    """
    if n_components is None or n_components <= 0:
        return None
    n_samples, n_features = embeddings_train.shape
    k = min(n_components, n_samples, n_features)
    if k < 1:
        return None
    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(embeddings_train.astype(np.float64))
    return pca


def transform_with_pca(pca: PCA | None, x: np.ndarray) -> np.ndarray:
    if pca is None:
        return np.asarray(x, dtype=np.float32)
    return pca.transform(np.asarray(x, dtype=np.float64)).astype(np.float32)


def pearson_r_12d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays; nan if either is constant."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("a and b must have same length")
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pearson_r_batch(curves: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    """
    curves: (N, 12), idx_i, idx_j: (B,) -> (B,) Pearson values, nan where undefined.
    """
    ci = curves[idx_i]
    cj = curves[idx_j]
    B = idx_i.shape[0]
    out = np.empty(B, dtype=np.float64)
    for k in range(B):
        out[k] = pearson_r_12d(ci[k], cj[k])
    return out
