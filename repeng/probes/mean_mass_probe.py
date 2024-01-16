"""
Replication of mean-mass probes (MMP) described in <https://arxiv.org/abs/2310.06824>.
"""

from repeng.activations.probe_preparations import LabeledActivationArray
from repeng.probes.base import DotProductProbe


def train_mmp_probe(activations: LabeledActivationArray) -> DotProductProbe:
    # TODO: Implement the IID version of this probe.
    mean_true = activations.activations[activations.labels].mean(axis=0)
    mean_false = activations.activations[~activations.labels].mean(axis=0)
    return DotProductProbe(mean_true - mean_false)
