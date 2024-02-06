# %%
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from mppr import MContext
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.collections import get_datasets
from repeng.datasets.elk.utils.limits import limit_dataset_and_split_fn
from repeng.evals.logits import eval_logits_by_question
from repeng.evals.probes import eval_probe_by_question, eval_probe_by_row
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.collections import ProbeMethod, train_probe
from repeng.probes.logistic_regression import train_lr_probe

# %%
mcontext = MContext(Path("../output/probe_test"))
dataset = (
    mcontext.create_cached(
        "dataset-v4",
        lambda: get_datasets(
            [
                "arc_easy",
                "arc_easy/qna",
                # "geometry_of_truth/cities",
                # "common_sense_qa/qna",
                # "common_sense_qa/options-numbers",
                # "common_sense_qa/options-letters",
                # "open_book_qa",
            ]
        ),
        to=BinaryRow,
    )
    # .filter(lambda _, row: row.dataset_id == "open_book_qa")
    .filter(
        limit_dataset_and_split_fn(train_limit=2000, validation_limit=500),
    )
)


# %%
pprint(dataset.get()[0].model_dump())
pprint(dataset.get()[-1].model_dump())
print(set(d.dataset_id for d in dataset.get()))


# %%
@dataclass
class InputRow:
    row: BinaryRow
    llm_id: LlmId
    text: str


llm_ids: list[LlmId] = [
    # "pythia-12b",
    # "Llama-2-13b-chat-hf",
    # "pythia-6.9b",
    "Llama-2-7b-hf",
    # "Llama-2-7b-chat-hf",
]
inputs = dataset.flat_map(
    lambda key, row: {
        f"{key}-{llm_id}": InputRow(
            row=row,
            llm_id=llm_id,
            text=row.text,
        )
        for llm_id in llm_ids
    }
).sort(lambda _, row: llm_ids.index(row.llm_id))

# %%
activations = inputs.map_cached(
    "activations-v4",
    lambda _, row: get_model_activations(
        load_llm_oioo(
            row.llm_id,
            device=torch.device("cuda"),
            use_half_precision=True,
        ),
        text=row.text,
        last_n_tokens=1,
    ),
    to="pickle",
)

# %%
arrays_dataset = activations.join(
    inputs,
    lambda _, activations_row, input_row: ActivationResultRow(
        dataset_id=input_row.row.dataset_id,
        group_id=input_row.row.group_id,
        template_name=input_row.row.template_name,
        answer_type=input_row.row.answer_type,
        activations=activations_row.activations,
        prompt_logprobs=activations_row.token_logprobs.sum(),
        label=input_row.row.is_true,
        split=input_row.row.split,
        llm_id=input_row.llm_id,
    ),
)
arrays_dataset = ActivationArrayDataset(arrays_dataset.get())


# %%
def train_and_eval(
    llm_id: LlmId,
    point: str,
    probe_method: ProbeMethod,
    dataset_id: DatasetId,
):
    arrays = arrays_dataset.get(
        llm_id=llm_id,
        dataset_filter_id=dataset_id,
        split="train",
        point_name=point,
        token_idx=-1,
        limit=None,
    )
    probe = train_probe(probe_method, arrays)
    assert probe is not None
    arrays_val = arrays_dataset.get(
        llm_id=llm_id,
        dataset_filter_id=dataset_id,
        split="validation",
        point_name=point,
        token_idx=-1,
        limit=None,
    )
    assert arrays_val.groups is not None
    results = eval_probe_by_question(
        probe,
        activations=arrays_val.activations,
        labels=arrays_val.labels,
        groups=arrays_val.groups,
    )
    return dict(
        llm_id=llm_id,
        point=point,
        probe_method=probe_method,
        dataset_id=dataset_id,
        acc=results.accuracy,
    )


def eval_logprobs(llm_id: LlmId, dataset_id: DatasetId):
    arrays_logits = arrays_dataset.get(
        llm_id=llm_id,
        dataset_filter_id=dataset_id,
        split="validation",
        point_name="logprobs",
        token_idx=-1,
        limit=None,
    )
    assert arrays_logits.groups is not None
    results = eval_logits_by_question(
        logits=arrays_logits.activations,
        labels=arrays_logits.labels,
        groups=arrays_logits.groups,
    )
    return dict(
        llm_id=llm_id,
        point="logprobs",
        probe_method="logprobs",
        dataset_id=dataset_id,
        acc=results.accuracy,
    )


probe_methods: list[ProbeMethod] = ["mmp", "lr"]
dataset_ids: list[DatasetId] = ["arc_easy", "arc_easy/qna"]
df = (
    mcontext.create(
        {
            f"{llm_id}-{point.name}-{probe_method}-{dataset_id}": (
                llm_id,
                point.name,
                probe_method,
                dataset_id,
            )
            for llm_id in llm_ids
            for point in get_points(llm_id)[-10:]
            for probe_method in probe_methods
            for dataset_id in dataset_ids
        }
    )
    .map_cached(
        "train-and-eval",
        lambda _, args: train_and_eval(
            llm_id=args[0], point=args[1], probe_method=args[2], dataset_id=args[3]
        ),
        to="pickle",
    )
    .to_dataframe(lambda d: d)
)
df_logprobs = (
    mcontext.create(
        {
            f"{llm_id}-{dataset_id}": (llm_id, dataset_id)
            for llm_id in llm_ids
            for dataset_id in dataset_ids
        }
    )
    .map_cached(
        "eval-logprobs",
        lambda _, args: eval_logprobs(llm_id=args[0], dataset_id=args[1]),
        to="pickle",
    )
    .to_dataframe(lambda d: d)
)
df = pd.concat([df, df_logprobs])


# %%
fig = px.line(
    pd.DataFrame(df).sort_values("point"),
    x="point",
    y="acc",
    color="probe_method",
    facet_col="llm_id",
    facet_row="dataset_id",
    markers=True,
)
fig.update_layout(height=600)
fig.show()

# %%
acts = activations.join(
    inputs,
    lambda _, activations_row, input_row: dict(
        dataset_id=input_row.row.dataset_id,
        label=input_row.row.is_true,
        group_id=input_row.row.group_id,
        activations=activations_row.token_logprobs.sum(),
        llm_id=input_row.llm_id,
    ),
).to_dataframe(lambda d: d)
mean_activations = (
    acts.groupby("group_id")["activations"].mean().rename("mean_activations")
)
acts = acts.join(mean_activations, on="group_id")
acts["activations"] -= acts["mean_activations"]
acts = acts.groupby("label").sample(100)

# %%
fig = px.histogram(
    acts,
    x="activations",
    color="label",
    facet_col="llm_id",
    nbins=50,
    opacity=0.5,
    barmode="overlay",
)
fig.show()

# %%
for model in llm_ids:
    acts = activations.join(
        inputs.filter(lambda _, row: row.llm_id == model),
        lambda _, activations_row, input_row: Activation(
            dataset_id=input_row.row.dataset_id,
            label=input_row.row.is_true,
            group_id=input_row.row.group_id,
            activations=activations_row.token_logprobs.sum(),
        ),
    ).get()
    arrays = prepare_activations_for_probes(acts)
    assert arrays.labeled_grouped is not None
    print(model)
    pprint(
        eval_logits_by_question(
            LabeledGroupedLogits(
                logits=arrays.labeled_grouped.activations,
                labels=arrays.labeled_grouped.labels,
                groups=arrays.labeled_grouped.groups,
            )
        ).model_dump()
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
        # answer_tag=input_row.row.answer_tag,
        pair_id=input_row.row.group_id,
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
# df2 = df2[df2["dataset_id"] == "common_sense_qa"]
df2 = df2[df2["dataset_id"] == "open_book_qa"]
df2["activation"] = df2["activations"].apply(lambda a: a["h13"][-1])

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

# pair_means = (
#     df2.groupby(["pair_id"])["activation"]
#     .apply(lambda a: np.mean(a, axis=0))
#     .rename("pair_mean")
# )
# df2 = df2.join(pair_means, on="pair_id")
# df2["activation"] -= df2["pair_mean"]

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

df2_train = df2[df2["split"] == "train"].copy()
# pair_means = (
#     df2_train.groupby(["pair_id"])["activation"]
#     .apply(lambda a: np.mean(a, axis=0))
#     .rename("pair_mean")
# )
# df2_train = df2_train.join(pair_means, on="pair_id")
# df2_train["activation"] -= df2_train["pair_mean"]

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
labeled_grouped_activation_array = LabeledGroupedActivationArray(
    activations=np.array(df2_train["activation"].to_list()),
    labels=np.array(df2_train["label"].to_list()),
    groups=np.array(df2_train["pair_id"].to_list()),
)
activation_array_val = LabeledActivationArray(
    activations=np.array(df2_val["activation"].to_list()),
    labels=np.array(df2_val["label"].to_list()),
)

probe = train_lr_probe(labelled_activation_array)
# probe = train_grouped_lr_probe(labeled_grouped_activation_array)
# probe = train_mmp_probe(labelled_activation_array, use_iid=False)
# probe = train_lat_probe(
#     activation_array,
#     LatTrainingConfig(num_random_pairs=5000),
# )
eval = eval_probe_by_row(probe, activation_array_val)
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
            group_id=None,
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
            group_id=None,
            activations=row.activations[get_points(row.model)[-2].name][-1],
            label=row.label,
        )
        for row in df.itertuples()
        if row.split == "validation" and row.model == "pythia-1b"
    ]
)

probe_methods: list[ProbeMethod] = ["lat", "mmp", "lr"]
df_eval = pd.DataFrame(
    [
        dict(
            probe_method=probe_method,
            **eval_probe_by_row(
                train_probe(probe_method, probe_arrays), probe_arrays_val.labeled
            ).model_dump(),
        )
        for probe_method in probe_methods
    ],
)
df_eval[["probe_method", "f1_score", "precision", "recall", "roc_auc_score"]]

# %% plot ROC curves
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, row in zip(axs, df_eval.itertuples()):
    ax.plot(row.fprs, row.tprs)
    ax.set_title(row.probe_method)

# %%
np.std([np.array([1, 2, 3]), np.array([1, 2, 3])], axis=0)

# %%
df["activations"].apply(lambda a: a["h14"][-1]).mean()
# sorted(df["pair_id"].unique())
acts = np.stack(df["activations"].apply(lambda a: a["h14"][-1]))
print(acts.shape)
print(np.cov(acts.T).shape)
print(acts.mean(axis=0)[:5])
print(np.cov(acts.T).flatten()[:5])
