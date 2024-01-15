# %%
from pathlib import Path

import numpy as np
from mppr import mppr

from repeng import models
from repeng.activations import get_activations
from repeng.datasets.collections import PAIRED_DATASET_IDS, get_all_datasets
from repeng.datasets.types import BinaryRow
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import (
    LatTrainingConfig,
    train_lat_probe,
)
from repeng.probes.types import Activations, PairedActivations

# %%
model, tokenizer, points = models.gpt2()

# %%
inputs = mppr.init(
    "init-limit-100",
    Path("../output/comparison"),
    init_fn=lambda: get_all_datasets(limit_per_dataset=100),
    to=BinaryRow,
)
print(len(inputs.get()))

# %%
df = (
    inputs.map(
        "activations",
        fn=lambda _, value: get_activations(
            model,
            tokenizer,
            points,
            value.text,
        ),
        to="pickle",
    )
    .join(
        inputs,
        lambda _, activations, input: dict(
            dataset_id=input.dataset_id,
            is_true=input.is_true,
            activations=activations.activations[points[-1].name],
            pair_id=input.pair_id,
        ),
    )
    .to_dataframe(lambda d: d)
)
df

# %%
df_subset = df[df["dataset_id"].isin(PAIRED_DATASET_IDS)].copy()
activations = Activations(
    activations=np.stack(df_subset["activations"].to_list()),
)

df_subset = df_subset.groupby(["dataset_id", "pair_id", "is_true"]).first()
df_subset = df_subset.reset_index()
df_subset = df_subset.pivot(index="pair_id", columns="is_true", values="activations")
paired_activations = PairedActivations(
    activations_1=np.stack(df_subset[False].to_list()),
    activations_2=np.stack(df_subset[False].to_list()),
)

# %%
ccs_probe = train_ccs_probe(
    paired_activations,
    CcsTrainingConfig(),
)
lat_probe = train_lat_probe(
    activations,
    LatTrainingConfig(),
)
