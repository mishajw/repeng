"""
Replication of mean-mass probes (MMP) described in <https://arxiv.org/abs/2310.06824>.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from typing_extensions import override

from repeng.activations.probe_preparations import LabeledActivationArray
from repeng.probes.base import DotProductProbe, PredictResult


@dataclass
class IidDotProductProbe(DotProductProbe):
    inv_covariance_matrix: Float[np.ndarray, "d d"]  # noqa: F722

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        activations_skewed = (self.inv_covariance_matrix @ activations.T).T
        return super().predict(activations_skewed)


def train_mmp_probe(
    activations: LabeledActivationArray,
    *,
    use_iid: bool,
) -> DotProductProbe | IidDotProductProbe:
    mean_true = activations.activations[activations.labels].mean(axis=0)
    mean_false = activations.activations[~activations.labels].mean(axis=0)
    if use_iid:
        centered_activations = activations.activations
        centered_activations[activations.labels] -= mean_true
        centered_activations[~activations.labels] -= mean_false
        cov = np.cov(centered_activations, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        return IidDotProductProbe(mean_true - mean_false, inv_cov)
    else:
        return DotProductProbe(mean_true - mean_false)
