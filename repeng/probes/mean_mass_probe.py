"""
Replication of mean-mass probes (MMP) described in <https://arxiv.org/abs/2310.06824>.
"""
import numpy as np

from repeng.activations.probe_preparations import LabeledActivationArray
from repeng.probes.base import DotProductProbe


def train_mmp_probe(
    activations: LabeledActivationArray,
    *,
    use_iid: bool,
) -> DotProductProbe:
    _, hidden_dim = activations.activations.shape
    mean_true = activations.activations[activations.labels].mean(axis=0)
    mean_false = activations.activations[~activations.labels].mean(axis=0)
    direction = mean_true - mean_false
    if not use_iid:
        return DotProductProbe(direction)

    centered_activations = np.concatenate(
        [
            activations.activations[activations.labels] - mean_true,
            activations.activations[~activations.labels] - mean_false,
        ]
    )
    cov = np.cov(centered_activations, rowvar=False)
    assert cov.shape == (hidden_dim, hidden_dim), cov.shape
    inv_cov = np.linalg.inv(cov)
    return DotProductProbe(inv_cov.T @ direction)
