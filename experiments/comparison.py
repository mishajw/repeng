# %%
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from mppr import mppr

from repeng import models
from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import (
    Activation,
    prepare_activations_for_probes,
)
from repeng.datasets.collections import PAIRED_DATASET_IDS, get_datasets
from repeng.datasets.filters import limit_dataset_and_split_fn
from repeng.datasets.types import BinaryRow
from repeng.evals.probes import evaluate_probe
from repeng.probes.base import BaseProbe
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import (
    LatTrainingConfig,
    train_lat_probe,
)
from repeng.probes.mean_mass_probe import train_mmp_probe

assert load_dotenv()

# %%
model, tokenizer, points = models.gpt2()

# %%
inputs = mppr.init(
    "init",
    Path("../output/comparison"),
    init_fn=lambda: get_datasets(PAIRED_DATASET_IDS),
    to=BinaryRow,
).filter(
    limit_dataset_and_split_fn(100),
)
print(len(inputs.get()))


# %%
@dataclass
class ActivationAndInputRow(Activation):
    split: str


activations_and_inputs = (
    inputs.map(
        "activations",
        fn=lambda _, value: get_model_activations(
            model,
            tokenizer,
            points,
            value.text,
        ),
        to="pickle",
    )
    .join(
        inputs,
        lambda _, activations, input: ActivationAndInputRow(
            dataset_id=input.dataset_id,
            split=input.split,
            label=input.is_true,
            activations=activations.activations[points[-1].name],
            pair_id=input.pair_id,
        ),
    )
    .upload("s3://repeng/comparison/activations/gpt2", to="pickle")
)

# %%
probe_arrays = prepare_activations_for_probes(
    activations_and_inputs.filter(lambda _, row: row.split == "train").get()
)
probe_arrays_validation = prepare_activations_for_probes(
    activations_and_inputs.filter(lambda _, row: row.split == "validation").get()
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
            **evaluate_probe(probe, probe_arrays.labeled).model_dump(),
        )
        for name, probe in probes.items()
    ]
)
