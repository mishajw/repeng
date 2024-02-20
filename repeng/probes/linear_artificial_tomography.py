"""
Replication of LAT probes described in <https://arxiv.org/abs/2310.01405>. See appendix
C.1.

Methodology:
1. Given a set of activations, randomly sample pairs without replacement.
2. Compute the difference between each pair.
3. Normalize the differences by subtracting the mean difference.
4. Take the first principle component of the normalized differences. This results in the
probe.
"""

import random
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int64
from sklearn.decomposition import PCA
from typing_extensions import override

from repeng.probes.base import DotProductProbe, PredictResult
from repeng.probes.normalization import normalize_by_group


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
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    answer_types: Int64[np.ndarray, "n d"] | None,  # noqa: F722
) -> DotProductProbe:
    if answer_types is not None:
        activations = normalize_by_group(activations, answer_types)
    indices = list(range(len(activations)))
    random.shuffle(indices)  # TODO: Double check if shuffling breaks things.
    indices = np.array(indices)[: len(indices) // 2 * 2]
    indices_1, indices_2 = indices.reshape(2, -1)

    activation_diffs = activations[indices_1] - activations[indices_2]
    activations_center = np.mean(activation_diffs, axis=0)
    activation_diffs_norm = activation_diffs - activations_center
    pca = PCA(n_components=1)
    pca.fit_transform(activation_diffs_norm)
    probe = pca.components_.squeeze(0)
    probe = probe / np.linalg.norm(probe)
    return CentredDotProductProbe(probe=probe, center=activations_center)
