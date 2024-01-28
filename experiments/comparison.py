# %%
import itertools
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from mppr import MContext

from repeng.activations.probe_preparations import (
    Activation,
    ProbeArrays,
    prepare_activations_for_probes,
)
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import DatasetId, Split, is_dataset_paired
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from repeng.evals.probes import ProbeEvalResult, evaluate_paired_probe, evaluate_probe
from repeng.models.llms import LlmId
from repeng.models.points import get_points
from repeng.probes.base import BasePairedProbe, BaseProbe
from repeng.probes.collections import ProbeId, train_probe
from repeng.probes.logistic_regression import train_lr_probe, train_paired_lr_probe

assert load_dotenv()

# %%
mcontext = MContext(Path("../output/comparison"))
activations_dataset: list[ActivationResultRow] = mcontext.download_cached(
    "activations_dataset",
    path="s3://repeng/datasets/activations/pythia_2024-01-26_v1.pickle",
    to="pickle",
).get()
print(set(row.llm_id for row in activations_dataset))
print(set(row.dataset_id for row in activations_dataset))
print(set(row.split for row in activations_dataset))


# %%
@dataclass
class ProbeTrainSpec:
    llm_id: LlmId
    dataset_collection_id: DatasetId | DatasetCollectionId
    probe_id: ProbeId
    point_id: str
    token_idx_from_back: int


llm_ids: list[LlmId] = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
]
llm_points = {llm_id: get_points(llm_id) for llm_id in llm_ids}
point_ids_by_llm = {
    llm_id: {
        f"p{int(i*100):02d}": llm_points[llm_id][
            int((len(llm_points[llm_id]) - 1) * i)
        ].name
        for i in np.arange(0, 1.1, 0.1)
    }
    for llm_id in llm_ids
}
dataset_collection_ids: list[DatasetId | DatasetCollectionId] = [
    "representation-engineering",
    "geometry-of-truth",
    "geometry_of_truth-cities",
    "geometry_of_truth-neg_cities",
    "geometry-of-truth-cities-with-neg",
    "open_book_qa",
    "common_sense_qa",
    "race",
    "arc_challenge",
    "arc_easy",
]
probe_ids: list[ProbeId] = [
    "lat",
    "mmp",
    "mmp-iid",
    "lr",
    "lr-paired",
]
probe_train_specs = mcontext.create(
    {
        f"{llm_id}-{dataset_collection_id}-{probe_id}-{point_id}": ProbeTrainSpec(
            llm_id=llm_id,
            dataset_collection_id=dataset_collection_id,
            probe_id=probe_id,
            point_id=point_id,
            token_idx_from_back=1,
        )
        for llm_id, dataset_collection_id, probe_id in itertools.product(
            llm_ids,
            dataset_collection_ids,
            probe_ids,
        )
        for point_id in point_ids_by_llm[llm_id].keys()
    }
)


# %%
def prepare_probe_arrays(
    llm_id: LlmId,
    dataset_ids: list[DatasetId],
    split: Split,
    point_name: str,
    token_idx_from_back: int,
) -> ProbeArrays:
    return prepare_activations_for_probes(
        [
            Activation(
                dataset_id=row.dataset_id,
                pair_id=row.pair_id,
                activations=row.activations[point_name][token_idx_from_back],
                label=row.label,
            )
            for row in activations_dataset
            if row.llm_id == llm_id
            and row.dataset_id in dataset_ids
            and row.split == split
        ]
    )


probes = probe_train_specs.map_cached(
    "probe_train",
    lambda _, spec: train_probe(
        spec.probe_id,
        prepare_probe_arrays(
            spec.llm_id,
            resolve_dataset_ids(spec.dataset_collection_id),
            split="train",
            point_name=point_ids_by_llm[spec.llm_id][spec.point_id],
            token_idx_from_back=spec.token_idx_from_back,
        ),
    ),
    to="pickle",
)


# %%
@dataclass
class ProbeEvalSpec:
    train_spec: ProbeTrainSpec
    probe: BaseProbe
    dataset_id: DatasetId
    is_paired: bool


evaluation_dataset_ids: list[DatasetId] = [
    "geometry_of_truth-cities",
    "geometry_of_truth-neg_cities",
    "open_book_qa",
    "common_sense_qa",
    "race",
    "arc_challenge",
    "arc_easy",
    "truthful_qa",
]


def evaluate(_: str, spec: ProbeEvalSpec) -> ProbeEvalResult:
    probe_arrays = prepare_probe_arrays(
        spec.train_spec.llm_id,
        [spec.dataset_id],
        split="validation",
        point_name=point_ids_by_llm[spec.train_spec.llm_id][spec.train_spec.point_id],
        token_idx_from_back=spec.train_spec.token_idx_from_back,
    )
    if spec.is_paired:
        assert probe_arrays.labeled_paired is not None
        assert isinstance(spec.probe, BasePairedProbe)
        return evaluate_paired_probe(spec.probe, probe_arrays.labeled_paired)
    else:
        return evaluate_probe(spec.probe, activations=probe_arrays.labeled)


probe_eval_specs = (
    probes.filter(lambda _, probe: probe is not None)
    .join(
        probe_train_specs,
        lambda _, probe, spec: (probe, spec),
    )
    .flat_map(
        lambda key, probe_and_spec: {
            f"{key}-{evaluation_dataset_id}": ProbeEvalSpec(
                probe_and_spec[1],
                cast(BaseProbe, probe_and_spec[0]),
                evaluation_dataset_id,
                is_paired=False,
            )
            for evaluation_dataset_id in evaluation_dataset_ids
        }
    )
    .flat_map(
        lambda key, spec: {
            f"{key}-paired={is_paired}": replace(spec, is_paired=is_paired)
            for is_paired in [True, False]
            if not is_paired
            or (
                is_dataset_paired(spec.dataset_id)
                and isinstance(spec.probe, BasePairedProbe)
            )
        }
    )
)
df = pd.DataFrame(
    [
        dict(
            probe_id=row.train_spec.probe_id,
            is_paired=row.is_paired,
            dataset_id=row.dataset_id,
            is_dataset_paired=is_dataset_paired(row.dataset_id),
            is_probe_paired=isinstance(row.probe, BasePairedProbe),
        )
        for row in probe_eval_specs.get()
    ]
)
df
probe_evaluations = probe_eval_specs.map_cached(
    "probe_evaluate", evaluate, to=ProbeEvalResult
)

# %%
df = probe_evaluations.join(
    probe_eval_specs,
    lambda _, evaluation, spec: dict(
        **asdict(spec.train_spec),
        **evaluation.model_dump(),
        eval_dataset_id=spec.dataset_id,
        is_eval_paired=spec.is_paired,
    ),
).to_dataframe(lambda d: d)
df["llm_id"] = pd.Categorical(df["llm_id"], llm_ids)
df["eval_dataset_id"] = pd.Categorical(df["eval_dataset_id"], list(get_args(DatasetId)))
df["point_id"] = pd.Categorical(
    df["point_id"],
    sorted(df["point_id"].unique().tolist(), key=lambda n: int(n.lstrip("p"))),
)
df = df.sort_values("llm_id")
# dims = llm_id, dataset_collection_id, probe_id, point_id, eval_dataset_id
df  # type: ignore

# %%
df2 = df.copy()
df2 = df2[df2["point_id"] == "p90"]
df2 = df2[df2["llm_id"] == "pythia-1b"]
# df2 = df2[df2["eval_dataset_id"].str.startswith("arc")]
# df2 = df2[df2["dataset_collection_id"].str.startswith("arc")]

g = sns.FacetGrid(
    df2,
    col="eval_dataset_id",
    row="probe_id",
    # margin_titles=True,
)
g.map(sns.barplot, "dataset_collection_id", "roc_auc_score")
plt.tight_layout()

# %%
df2 = df.copy()
df2 = df2[df2["point_id"] == "p90"]
df2 = df2[df2["llm_id"] == "pythia-6.9b"]
df2 = df2[df2["eval_dataset_id"] == "truthful_qa"]
df2 = df2[
    df2["dataset_collection_id"].isin(
        [
            "geometry-of-truth",
            "geometry-of-truth-cities-with-neg",
            "representation-engineering",
            "geometry_of_truth-cities",
            "common_sense_qa",
            "open_book_qa",
        ]
    )
]
df2["probe"] = df2["probe_id"] + "+" + df2["is_eval_paired"].astype(str)
df2 = df2.sort_values("is_eval_paired")

sns.heatmap(
    df2.pivot(
        index="dataset_collection_id",
        columns="probe",
        values="roc_auc_score",
    ),
    annot=True,
    fmt=".2f",
    cmap="Blues",
)


# %% plot ROC curves
df_subset = df.copy()
df_subset = df_subset[
    df_subset["dataset_collection_id"] == "common_sense_qa"
]  # TODO: remove
df_subset = df_subset[df_subset["point_id"] == "p90"]
df_subset = df_subset.drop(columns=["point_id"])
df_subset = df_subset[df_subset["llm_id"] == "pythia-6.9b"]
df_subset = df_subset.drop(columns=["llm_id"])  # type: ignore
df_subset["probe_method"] = (
    df_subset["probe_id"]
    + "+"
    + df_subset["dataset_collection_id"]
    + "+"
    + df_subset["is_eval_paired"].astype(str)
)
df_subset = df_subset.drop(columns=["probe_id", "dataset_collection_id"])


fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(4 * 5, 4 * 5))
df_subset = df_subset.sort_values("eval_dataset_id")  # type: ignore
for eval_dataset_id, ax in zip(df_subset["eval_dataset_id"].unique(), axs.flatten()):
    ax.set_title(eval_dataset_id)
    for probe_method in (
        df_subset[df_subset["eval_dataset_id"] == eval_dataset_id]["probe_method"]
        .unique()
        .tolist()
    ):
        df_row = df_subset[
            (df_subset["eval_dataset_id"] == eval_dataset_id)
            & (df_subset["probe_method"] == probe_method)
        ].iloc[0]
        ax.plot(
            df_row["fprs"],
            df_row["tprs"],
            label=probe_method,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

handles, labels = [], []
for ax in axs.flatten():
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
fig.legend(handles, labels)
fig.tight_layout()
df_subset[["probe_method", "eval_dataset_id", "roc_auc_score", "is_eval_paired"]]

# %% plot probe performance by model size
df_subset = df.copy()
df_subset = df_subset[df_subset["point_id"] == "p90"]
df_subset = df_subset[df_subset["probe_id"] == "mmp"]
df_subset = df_subset.drop(columns=["point_id", "probe_id"])
g = (
    sns.FacetGrid(
        df_subset,
        col="eval_dataset_id",
        hue="dataset_collection_id",
        margin_titles=True,
    )
    .map(sns.lineplot, "llm_id", "roc_auc_score")
    .add_legend()
)
# sns.lineplot(data=df_subset, x="llm_id", y="roc_auc_score")
# plt.xticks(rotation=90)
# g.fig.tight_layout()
plt.show()

# %%
df_subset = df.copy()
df_subset = df_subset[df_subset["llm_id"] == "pythia-6.9b"]
df_subset = df_subset[df_subset["dataset_collection_id"] == "geometry-of-truth-cities"]
df_subset = df_subset[df_subset["point_id"] == "p100"]
sns.barplot(
    data=df_subset,
    x="probe_id",
    y="roc_auc_score",
    hue="eval_dataset_id",
    legend=False,
)

# %% generalization matrix
df_subset = df.copy()
df_subset = df_subset[df_subset["llm_id"] == "pythia-1b"]
df_subset = df_subset[df_subset["point_id"] == "p90"]
df_subset = df_subset[df_subset["probe_id"] == "lr"]
df_subset = df_subset.pivot(
    index="dataset_collection_id",
    columns="eval_dataset_id",
    values="roc_auc_score",
)
df_subset = df_subset.sort_index(level=0)
df_subset = df_subset.sort_values("dataset_collection_id")
sns.heatmap(df_subset, annot=True, fmt=".2f", cmap="Blues")

# %%
probe_arrays = prepare_activations_for_probes(
    [
        Activation(
            dataset_id=row.dataset_id,
            pair_id=row.pair_id,
            activations=row.activations["h14"][-1],
            label=row.label,
        )
        for row in activations_dataset
        if row.dataset_id == "common_sense_qa"
        and row.split == "train"
        and row.llm_id == "pythia-6.9b"
    ]
)
probe_arrays_val = prepare_activations_for_probes(
    [
        Activation(
            dataset_id=row.dataset_id,
            pair_id=row.pair_id,
            activations=row.activations["h31"][-1],
            label=row.label,
        )
        for row in activations_dataset
        if row.dataset_id == "open_book_qa"
        and row.split == "validation"
        and row.llm_id == "pythia-6.9b"
    ]
)
probe = train_lr_probe(probe_arrays.labeled)
print(evaluate_probe(probe, probe_arrays_val.labeled).roc_auc_score)
assert probe_arrays.labeled_paired is not None
probe = train_paired_lr_probe(probe_arrays.labeled_paired)
print(evaluate_probe(probe, probe_arrays_val.labeled).roc_auc_score)
print(probe_arrays.activations.activations.shape)

# %%
activations = [
    Activation(
        dataset_id=row.dataset_id,
        pair_id=row.pair_id,
        activations=row.activations["h13"][-1],
        label=row.label,
    )
    for row in activations_dataset
    if row.dataset_id == "open_book_qa"
    and row.split == "train"
    and row.llm_id == "pythia-1b"
]
activations_val = [
    Activation(
        dataset_id=row.dataset_id,
        pair_id=row.pair_id,
        activations=row.activations["h14"][-1],
        label=row.label,
    )
    for row in activations_dataset
    if row.dataset_id == "open_book_qa"
    and row.split == "validation"
    and row.llm_id == "pythia-1b"
]
df = pd.DataFrame([asdict(activation) for activation in activations])
acts = np.stack(df["activations"])
print(acts.shape)
print(np.cov(acts.T).shape)
print(acts.mean(axis=0)[:5])
print(np.cov(acts.T).flatten()[:5])

probe_arrays = prepare_activations_for_probes(activations)
probe_arrays_val = prepare_activations_for_probes(activations_val)
probe = train_paired_lr_probe(probe_arrays.labeled_paired)
evaluate_probe(probe, probe_arrays_val.labeled).roc_auc_score

# %%
point_ids_by_llm
