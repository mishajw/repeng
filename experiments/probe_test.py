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
from sklearn.metrics import roc_auc_score

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import (
    Activation,
    ActivationArray,
    LabeledActivationArray,
    prepare_activations_for_probes,
)
from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.elk.utils.collections import get_datasets
from repeng.datasets.elk.utils.filters import limit_dataset_and_split_fn
from repeng.evals.probes import evaluate_probe
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.collections import ProbeId, train_probe
from repeng.probes.logistic_regression import train_lr_probe

# %%
mcontext = MContext(Path("../output/probe_test"))
dataset = (
    mcontext.create_cached(
        "dataset-csq6",
        lambda: get_datasets(
            [
                "common_sense_qa/repe",
                "common_sense_qa/qna",
                "common_sense_qa/options-numbers",
                "common_sense_qa/options-letters",
                "open_book_qa",
            ]
        ),
        to=BinaryRow,
    )
    .filter(
        limit_dataset_and_split_fn(train_limit=2000, validation_limit=500),
    )
    .filter(
        lambda _, row: row.dataset_id
        in [
            "common_sense_qa/repe",
            "open_book_qa",
            # "common_sense_qa/qna",
        ]
    )
)


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
    "pythia-6.9b",
    # "pythia-1b",
    # "pythia-dpo-1b",
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
        load_llm_oioo(row.llm_id, device=torch.device("cuda"), dtype=torch.bfloat16),
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
        dataset_id=input_row.row.dataset_id,
        answer_tag=input_row.row.answer_tag,
        pair_id=input_row.row.pair_id,
    ),
).to_dataframe(lambda d: d)
df

# %% plot top 2 PCA components per-model
for llm_id in llm_ids:
    # for dataset_id in df["dataset_id"].unique():
    df2 = df.copy()
    df2 = df2[df2["model"] == llm_id]
    # df2 = df2[df2["dataset_id"] == dataset_id]
    point_name = get_points(llm_id)[-8].name  # second-to-last layer
    a = np.stack(df2["activations"].apply(lambda a: a[point_name][-1]).to_list())
    for answer_tag in df2["answer_tag"].unique():
        if answer_tag is None:
            continue
        mask = df2["answer_tag"] == answer_tag
        a[mask] -= a[mask].mean(axis=0)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(a)
    df2["pca_0"] = pca_fit[:, 0]
    df2["pca_1"] = pca_fit[:, 1]
    sns.scatterplot(data=df2, x="pca_0", y="pca_1", hue="label")
    plt.title(llm_id)
    plt.show()

# %%
df2 = df.copy()
df2 = df2[df2["dataset_id"] == "common_sense_qa/repe"]
# df2 = df2[df2["dataset_id"] == "open_book_qa"]
df2["activation"] = df2["activations"].apply(lambda a: a["h14"][-1])

# tag_means = (
#     df2.groupby(["answer_tag"])["activation"]
#     .apply(lambda a: np.mean(a, axis=0))
#     .reset_index()
#     .rename(columns={"activation": "tag_mean"})
#     .set_index("answer_tag")
# )
# df2 = df2.join(tag_means, on="answer_tag")
# df2["activation"] -= df2["tag_mean"]

# all_mean = df2["activation"].mean(axis=0)
# df2["activation"] = df2["activation"].apply(lambda a: a - all_mean)

pair_means = (
    df2.groupby(["pair_id"])["activation"]
    .apply(lambda a: np.mean(a, axis=0))
    .rename("pair_mean")
)
df2 = df2.join(pair_means, on="pair_id")
df2["activation"] -= df2["pair_mean"]

# pair_std = (
#     df2.groupby(["pair_id"])["activation"]
#     .apply(lambda a: np.std(np.stack(a), axis=0) + 1e-6)
#     .rename("pair_std")
# )
# df2 = df2.join(pair_std, on="pair_id")
# df2["activation"] /= df2["pair_std"]

# pca = PCA(n_components=2)
# pca_fit = pca.fit_transform(np.stack(df2["activation"].to_list()))
# df2["pca_0"] = pca_fit[:, 0]
# df2["pca_1"] = pca_fit[:, 1]
# sns.scatterplot(data=df2, x="pca_0", y="pca_1", hue="label")
# plt.show()

df2_train = df2[df2["split"] == "train"]
df2_val = df2[df2["split"] == "validation"]
train_limit = 2000
print(train_limit, len(df2_train))
labelled_activation_array = LabeledActivationArray(
    activations=np.array(df2_train["activation"].to_list()[:train_limit]),
    labels=np.array(df2_train["label"].to_list()[:train_limit]),
)
activation_array = ActivationArray(
    activations=np.array(df2_train["activation"].to_list()[:train_limit]),
)
activation_array_val = LabeledActivationArray(
    activations=np.array(df2_val["activation"].to_list()),
    labels=np.array(df2_val["label"].to_list()),
)

probe = train_lr_probe(labelled_activation_array)
# probe = train_mmp_probe(labelled_activation_array, use_iid=False)
# probe = train_lat_probe(
#     activation_array,
#     LatTrainingConfig(num_random_pairs=5000),
# )
eval = evaluate_probe(probe, activation_array_val)
plt.plot(eval.fprs, eval.tprs)
print(eval.roc_auc_score)
# print(eval.f1_score)

# %%
df2 = df.copy()
df2["logprob"] = df2["logprobs"].apply(lambda x: x.squeeze(1).sum())
df2["prob"] = df2["logprob"].apply(lambda x: np.exp(x))
pair_prob_denom = (
    df2.groupby(["pair_id", "dataset_id"])["prob"]
    .apply(lambda x: sum(x))
    .rename("sum_prob")
)
df2 = df2.join(pair_prob_denom, on=["pair_id", "dataset_id"])
df2["prob_norm"] = df2["prob"] / df2["sum_prob"]
df2
sns.barplot(data=df2, x="dataset_id", y="prob_norm", hue="label")
plt.xticks(rotation=45)

# %%
df2 = df.copy()
df2["logprob"] = df2["logprobs"].apply(
    lambda x: np.exp(x.squeeze(1).sum().astype("float64"))
)
# df2 = df2.dropna()
for dataset_id in df2["dataset_id"].unique():
    df3 = df2[df2["dataset_id"] == dataset_id]
    logprobs_and_labels = df3.groupby("pair_id").apply(
        lambda a: (np.array(a.logprob), np.array(a.label))
    )
    logprobs = logprobs_and_labels.apply(lambda a: a[0] / a[0].sum())
    labels = logprobs_and_labels.apply(lambda a: a[1])
    assert all(logprobs.index == labels.index)
    logprobs = np.stack(logprobs.to_numpy())
    labels = np.stack(labels.to_numpy())
    print(dataset_id, roc_auc_score(labels, logprobs))

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
    ax.plot(row.fprs, row.tprs)
    ax.set_title(row.probe_id)

# %%
np.std([np.array([1, 2, 3]), np.array([1, 2, 3])], axis=0)
