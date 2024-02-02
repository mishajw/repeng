from typing import Literal

from repeng.activations.probe_preparations import ActivationArrays
from repeng.probes.base import BaseProbe
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import train_lat_probe
from repeng.probes.logistic_regression import train_grouped_lr_probe, train_lr_probe
from repeng.probes.mean_mass_probe import train_mmp_probe

ProbeMethod = Literal["ccs", "lat", "mmp", "mmp-iid", "lr", "lr-grouped"]


def train_probe(
    probe_method: ProbeMethod, arrays: ActivationArrays
) -> BaseProbe | None:
    if probe_method == "ccs":
        if arrays.groups is None:
            return None
        return train_ccs_probe(
            CcsTrainingConfig(),
            activations=arrays.activations,
            groups=arrays.groups,
            answer_types=arrays.answer_types,
            # N.B.: Technically unsupervised!
            labels=arrays.labels,
        )
    elif probe_method == "lat":
        return train_lat_probe(
            activations=arrays.activations,
        )
    elif probe_method == "mmp":
        return train_mmp_probe(
            activations=arrays.activations,
            labels=arrays.labels,
            use_iid=False,
        )
    elif probe_method == "mmp-iid":
        return train_mmp_probe(
            activations=arrays.activations,
            labels=arrays.labels,
            use_iid=True,
        )
    elif probe_method == "lr":
        return train_lr_probe(
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif probe_method == "lr-grouped":
        if arrays.groups is None:
            return None
        return train_grouped_lr_probe(
            activations=arrays.activations,
            groups=arrays.groups,
            labels=arrays.labels,
        )
    else:
        raise ValueError(f"Unknown probe_method: {probe_method}")
