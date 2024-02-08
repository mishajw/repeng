"""
Baseline random probe.
"""

import numpy as np
from jaxtyping import Float

from repeng.probes.base import DotProductProbe


def train_random_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
) -> DotProductProbe:
    _, hidden_dim = activations.shape
    probe = np.random.uniform(-1, 1, size=hidden_dim)
    probe /= np.linalg.norm(probe)
    return DotProductProbe(probe=probe)
