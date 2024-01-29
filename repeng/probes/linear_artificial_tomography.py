"""
Replication of LAT probes described in <https://arxiv.org/abs/2310.01405>.
See appendix C.1
"""

import random
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from sklearn.decomposition import PCA
from typing_extensions import override

from repeng.activations.probe_preparations import ActivationArray
from repeng.probes.base import DotProductProbe, PredictResult


@dataclass
class CentredDotProductProbe(DotProductProbe):
    center: Float[np.ndarray, "d"]  # noqa: F821

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        return super().predict(activations - self.center)


def train_lat_probe(
    activations: ActivationArray,
) -> DotProductProbe:
    indices = list(range(len(activations.activations)))
    random.shuffle(indices)  # TODO: Double check if shuffling breaks things.
    indices = np.array(indices)[: len(indices) // 2 * 2]
    indices_1, indices_2 = indices.reshape(2, -1)

    activation_diffs = (
        activations.activations[indices_1] - activations.activations[indices_2]
    )
    activations_center = np.mean(activation_diffs, axis=0)
    activation_diffs_norm = activation_diffs - activations_center
    pca = PCA(n_components=1)
    pca.fit_transform(activation_diffs_norm)
    probe = pca.components_.squeeze(0)
    probe = probe / np.linalg.norm(probe)
    return CentredDotProductProbe(probe=probe, center=activations_center)
