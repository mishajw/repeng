from typing import Literal

from repeng.activations.probe_preparations import ProbeArrays
from repeng.probes.base import BaseProbe
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import train_lat_probe
from repeng.probes.logistic_regression import train_grouped_lr_probe, train_lr_probe
from repeng.probes.mean_mass_probe import train_mmp_probe

ProbeMethod = Literal["ccs", "lat", "mmp", "mmp-iid", "lr", "lr-grouped"]


def train_probe(
    probe_method: ProbeMethod, probe_arrays: ProbeArrays
) -> BaseProbe | None:
    if probe_method == "ccs":
        if probe_arrays.labeled_grouped is None:
            return None
        return train_ccs_probe(
            # N.B.: Technically unsupervised!
            probe_arrays.labeled_grouped,
            CcsTrainingConfig(),
        )
    elif probe_method == "lat":
        return train_lat_probe(
            probe_arrays.activations,
        )
    elif probe_method == "mmp":
        return train_mmp_probe(
            probe_arrays.labeled,
            use_iid=False,
        )
    elif probe_method == "mmp-iid":
        return train_mmp_probe(
            probe_arrays.labeled,
            use_iid=True,
        )
    elif probe_method == "lr":
        return train_lr_probe(
            probe_arrays.labeled,
        )
    elif probe_method == "lr-grouped":
        if probe_arrays.labeled_grouped is None:
            return None
        return train_grouped_lr_probe(
            probe_arrays.labeled_grouped,
        )
    else:
        raise ValueError(f"Unknown probe_method: {probe_method}")
