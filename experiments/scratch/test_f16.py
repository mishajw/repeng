# %%
from pathlib import Path

import numpy as np
import plotly.express as px
from mppr import MContext

from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.evals.logits import eval_logits_by_question
from repeng.evals.probes import eval_probe_by_question
from repeng.probes.logistic_regression import train_grouped_lr_probe

# %%
mcontext = MContext(Path("../output/comparison"))
activations_dataset: list[ActivationResultRow] = mcontext.download_cached(
    "activations_dataset",
    path=(
        "s3://repeng/datasets/activations/"
        # "datasets_2024-02-02_tokensandlayers_v1.pickle"
        "datasets_2024-02-03_v1.pickle"
    ),
    to="pickle",
).get()
print(set(row.llm_id for row in activations_dataset))
print(set(row.dataset_id for row in activations_dataset))
print(set(row.split for row in activations_dataset))
dataset = ActivationArrayDataset(activations_dataset)

# %%
arrays = dataset.get(
    llm_id="pythia-12b",
    dataset_filter_id="arc_easy",
    split="train",
    point_name="h34",
    token_idx=-1,
    limit=None,
)
assert arrays.groups is not None
probe = train_grouped_lr_probe(
    activations=arrays.activations,
    labels=arrays.labels,
    groups=arrays.groups,
)

arrays_val = dataset.get(
    llm_id="pythia-12b",
    dataset_filter_id="arc_easy",
    split="validation",
    point_name="h34",
    token_idx=-1,
    limit=None,
)
assert arrays_val.groups is not None
print(
    eval_probe_by_question(
        probe,
        activations=arrays_val.activations,
        labels=arrays_val.labels,
        groups=arrays_val.groups,
    )
)

# %%
arrays = dataset.get(
    llm_id="pythia-12b",
    dataset_filter_id="arc_easy",
    split="train",
    point_name="logprobs",
    token_idx=-1,
    limit=None,
)
assert arrays.groups is not None
print(
    eval_logits_by_question(
        logits=arrays.activations,
        labels=arrays.labels,
        groups=arrays.groups,
    )
)

# %%
activations = arrays.activations.copy()
for group in np.unique(arrays.groups):
    activations[arrays.groups == group] -= activations[arrays.groups == group].mean(
        axis=0
    )

(idxs0,) = np.where(arrays.labels == 0)
(idxs1,) = np.where(arrays.labels == 1)
idxs = np.concatenate([idxs0[::3], idxs1])
px.histogram(
    x=activations[idxs],
    color=arrays.labels[idxs],
    nbins=50,
    opacity=0.5,
    barmode="overlay",
)

# %%
activations_dataset[0].activations["h0"].dtype
