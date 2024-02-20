"""
Replication of difference-in-means probes.

See DIM probes in <https://arxiv.org/abs/2312.01037v1> and MMP described in
<https://arxiv.org/abs/2310.06824>.

Methodology:
1. Given a set of activations, and whether they respond to true or false statements.
2. Compute the difference in means between the true and false activations. This is the
probe.
"""

import numpy as np
from jaxtyping import Bool, Float

from repeng.probes.base import DotProductProbe


def train_dim_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> DotProductProbe:
    return DotProductProbe(
        activations[labels].mean(axis=0) - activations[~labels].mean(axis=0)
    )
