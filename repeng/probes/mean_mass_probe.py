"""
Replication of mean-mass probes (MMP) described in <https://arxiv.org/abs/2310.06824>.
"""

from repeng.probes.types import LabeledActivations


def train_mmp_probe(activations: LabeledActivations):
    # TODO: Implement the IID version of this probe.
    mean_true = activations.activations[activations.labels].mean(axis=0)
    mean_false = activations.activations[~activations.labels].mean(axis=0)
    return mean_true - mean_false
