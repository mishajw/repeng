"""
Replication of LAT probes described in <https://arxiv.org/abs/2310.01405>.
See appendix C.1
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from sklearn.decomposition import PCA

from repeng.probes.types import Activations


@dataclass
class LatTrainingConfig:
    num_random_pairs: int = 100


@dataclass
class LatProbe:
    probe: Float[np.ndarray, "d"]  # noqa: F821


def train_lat_probe(
    activations: Activations,
    config: LatTrainingConfig,
) -> LatProbe:
    num_samples, _ = activations.activations.shape
    indices_1 = np.random.choice(num_samples, size=config.num_random_pairs)
    indices_2 = np.random.choice(num_samples, size=config.num_random_pairs)
    activation_diffs = (
        activations.activations[indices_1] - activations.activations[indices_2]
    )
    pca = PCA(n_components=1)
    pca.fit_transform(activation_diffs)
    return LatProbe(pca.components_.squeeze(0))
