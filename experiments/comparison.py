# %%

from typing import cast

import pandas as pd
from dotenv import load_dotenv

from repeng.activations.probe_preparations import (
    Activation,
    prepare_activations_for_probes,
)
from repeng.datasets.activations.types import ActivationResultRow
from repeng.evals.probes import evaluate_probe
from repeng.models.llms import LlmId
from repeng.probes.base import BaseProbe
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import (
    LatTrainingConfig,
    train_lat_probe,
)
from repeng.probes.mean_mass_probe import train_mmp_probe

assert load_dotenv()

# %%
llm_ids: list[LlmId] = ["gpt2", "pythia-70m"]

# %%
# TODO
activations_dataset: list[ActivationResultRow] = cast(list[ActivationResultRow], None)

# %%
probe_arrays = prepare_activations_for_probes(
    [
        Activation(
            dataset_id=row.dataset_id,
            pair_id=row.pair_id,
            activations=row.activations,
            label=row.label,
        )
        for row in activations_dataset
        if row.split == "train" and row.llm_id == "pythia-70m"
    ]
)
probe_arrays_validation = prepare_activations_for_probes(
    [
        Activation(
            dataset_id=row.dataset_id,
            pair_id=row.pair_id,
            activations=row.activations,
            label=row.label,
        )
        for row in activations_dataset
        if row.split == "validation" and row.llm_id == "pythia-70m"
    ]
)

# %%
probes: dict[str, BaseProbe] = dict(
    ccs=train_ccs_probe(
        probe_arrays.paired,
        CcsTrainingConfig(),
    ),
    lat=train_lat_probe(
        probe_arrays.activations,
        LatTrainingConfig(),
    ),
    mmp=train_mmp_probe(
        probe_arrays.labeled,
    ),
)

# %%
pd.DataFrame(
    [
        dict(
            name=name,
            **evaluate_probe(
                probe,
                probe_arrays_validation.labeled,
            ).model_dump(),
        )
        for name, probe in probes.items()
    ]
)
