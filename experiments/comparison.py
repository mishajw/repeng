# %%
from pathlib import Path

import numpy as np
from mppr import mppr

from repeng import models
from repeng.activations import get_activations
from repeng.datasets.collections import PAIRED_DATASET_IDS, get_datasets
from repeng.datasets.filters import limit_dataset_and_split_fn
from repeng.datasets.types import BinaryRow
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import (
    LatTrainingConfig,
    train_lat_probe,
)
from repeng.probes.mean_mass_probe import train_mmp_probe
from repeng.probes.types import Activations, LabeledActivations, PairedActivations

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
            split=input.split,
            is_true=input.is_true,
            activations=activations.activations[points[-1].name],
            pair_id=input.pair_id,
        ),
    )
    .to_dataframe(lambda d: d)
)
df

# %%
df_train = df[df["split"] == "train"]
activations = Activations(
    activations=np.stack(df_train["activations"].to_list()),
)
print(
    "activations",
    activations.activations.shape,
)

labeled_activations = LabeledActivations(
    activations=activations.activations,
    labels=df_train["is_true"].to_numpy(),
)
print(
    "labeled_activations",
    labeled_activations.activations.shape,
    labeled_activations.labels.shape,
)

df_paired = df_train.groupby(["dataset_id", "pair_id", "is_true"]).first()
df_paired = df_paired.reset_index()
df_paired = df_paired.pivot(index="pair_id", columns="is_true", values="activations")
df_paired = df_paired.dropna()
paired_activations = PairedActivations(
    activations_1=np.stack(df_paired[True].to_list()),
    activations_2=np.stack(df_paired[False].to_list()),
)
print(
    "paired_activations",
    paired_activations.activations_1.shape,
    paired_activations.activations_2.shape,
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
mmp_probe = train_mmp_probe(
    labeled_activations,
)
