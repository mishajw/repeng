"""
Replication of linear discriminant analysis (LDA) probes.

See LDA probes in <https://arxiv.org/abs/2312.01037v1> and MMP-IID described in
<https://arxiv.org/abs/2310.06824>.
"""

import numpy as np
from jaxtyping import Bool, Float

from repeng.probes.base import DotProductProbe


# TODO: Finding the inverse of the covariance matrix is really slow on big datasets and
# and big hidden dimensions. Speed this up!
def train_lda_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> DotProductProbe:
    _, hidden_dim = activations.shape
    mean_true = activations[labels].mean(axis=0)
    mean_false = activations[~labels].mean(axis=0)
    direction = mean_true - mean_false
    centered_activations = np.concatenate(
        [
            activations[labels] - mean_true,
            activations[~labels] - mean_false,
        ]
    )
    cov = np.cov(centered_activations, rowvar=False)
    assert cov.shape == (hidden_dim, hidden_dim), cov.shape
    inv_cov = np.linalg.inv(cov)
    return DotProductProbe(inv_cov.T @ direction)
