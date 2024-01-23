# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from mppr import MContext
from sklearn.decomposition import PCA

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import (
    Activation,
    prepare_activations_for_probes,
)
from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.elk.utils.collections import get_datasets
from repeng.evals.probes import evaluate_probe
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.collections import ProbeId, train_probe

# %%
mcontext = MContext(Path("../output/probe_test"))
dataset = mcontext.create_cached(
    "dataset",
    lambda: get_datasets(["geometry_of_truth-cities"]),
    to=BinaryRow,
).limit(500)


# %%
FewShotStyle = Literal[
    "none",
    "examples",
    "assistant",
]
FEW_SHOT_PREFIX_EXAMPLES = (
    "The city of London is in England.\n"
    "The city of Paris is in France.\n"
    "The city of Berlin is in Germany.\n"
)
FEW_SHOT_PREFIX_ASSISTANT = (
    "You are a helpful assistant who always tells the truth. "
    "The user will ask you for information, "
    "and your answers should *always* be truthful.\n"
    "USER: Could you tell me the location of a city?\n"
    "ASSISTANT: Sure: "
)


def add_few_shot_prefix(text: str, style: FewShotStyle) -> str:
    if style == "none":
        return text
    elif style == "examples":
        return FEW_SHOT_PREFIX_EXAMPLES + text
    elif style == "assistant":
        return FEW_SHOT_PREFIX_ASSISTANT + text
    else:
        raise ValueError(f"Unknown few-shot style: {style}")


# %%
@dataclass
class InputRow:
    row: BinaryRow
    llm_id: LlmId
    few_shot_style: FewShotStyle
    text: str


llm_ids: list[LlmId] = [
    "pythia-1b",
    "pythia-dpo-1b",
    # "pythia-sft-1b",
    # "pythia-1.4b",
    # "pythia-dpo-1.4b",
    # "pythia-sft-1.4b",
]
few_shot_styles: list[FewShotStyle] = [
    "none",
    # "examples",
    # "assistant",
]
inputs = dataset.flat_map(
    lambda key, row: {
        f"{key}-{llm_id}-{few_shot_style}": InputRow(
            row=row,
            llm_id=llm_id,
            few_shot_style=few_shot_style,
            text=add_few_shot_prefix(row.text, few_shot_style),
        )
        for llm_id in llm_ids
        for few_shot_style in few_shot_styles
    }
).sort(lambda _, row: row.llm_id)

# %%
activations = inputs.map_cached(
    "activations",
    lambda _, row: get_model_activations(
        load_llm_oioo(row.llm_id, device=torch.device("cpu"), dtype=torch.float32),
        text=row.text,
        last_n_tokens=1,
    ),
    to="pickle",
)

# %%
df = activations.join(
    inputs,
    lambda _, activations_row, input_row: dict(
        activations=activations_row.activations,
        label=input_row.row.is_true,
        split=input_row.row.split,
        logprobs=activations_row.token_logprobs,
        model=input_row.llm_id,
        few_shot_style=input_row.few_shot_style,
    ),
).to_dataframe(lambda d: d)
df

# %% plot top 2 PCA components per-model
for llm_id in llm_ids:
    pca = PCA(n_components=2)
    df_model = df[df["model"] == llm_id].copy()
    point_name = get_points(llm_id)[-2].name  # second-to-last layer
    a = np.stack(df_model["activations"].apply(lambda a: a[point_name][-1]).to_list())
    pca_fit = pca.fit_transform(a)
    df_model["pca_0"] = pca_fit[:, 0]
    df_model["pca_1"] = pca_fit[:, 1]
    sns.scatterplot(data=df_model, x="pca_0", y="pca_1", hue="label")
    plt.title(llm_id)
    plt.show()

# %% plot logprobs by model and few-shot style
df["logprob"] = df["logprobs"].apply(lambda x: x.sum())
g = sns.FacetGrid(
    df,
    col="model",
    row="few_shot_style",
    hue="label",
    margin_titles=True,
)
g.map(sns.histplot, "logprob", edgecolor="w").add_legend()
plt.show()

# %% train and evaluate probes
probe_arrays = prepare_activations_for_probes(
    [
        Activation(
            dataset_id="geometry_of_truth-cities",
            pair_id=None,
            activations=row.activations[get_points(row.model)[-2].name][-1],
            label=row.label,
        )
        for row in df.itertuples()
        if row.split == "train" and row.model == "pythia-1b"
    ]
)
probe_arrays_val = prepare_activations_for_probes(
    [
        Activation(
            dataset_id="geometry_of_truth-cities",
            pair_id=None,
            activations=row.activations[get_points(row.model)[-2].name][-1],
            label=row.label,
        )
        for row in df.itertuples()
        if row.split == "validation" and row.model == "pythia-1b"
    ]
)

probe_ids: list[ProbeId] = ["lat", "mmp", "lr"]
df_eval = pd.DataFrame(
    [
        dict(
            probe_id=probe_id,
            **evaluate_probe(
                train_probe(probe_id, probe_arrays), probe_arrays_val.labeled
            ).model_dump(),
        )
        for probe_id in probe_ids
    ],
)
df_eval[["probe_id", "f1_score", "precision", "recall", "roc_auc_score"]]

# %% plot ROC curves
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, row in zip(axs, df_eval.itertuples()):
    ax.plot(
        row.fprs,
        row.tprs,
    )
    ax.set_title(row.probe_id)
