# %%
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args

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
from repeng.datasets.elk.types import DatasetId, Split
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from repeng.evals.probes import ProbeEvalResult, evaluate_probe
from repeng.models.llms import LlmId
from repeng.models.points import get_points
from repeng.probes.base import BaseProbe
from repeng.probes.collections import ProbeId, train_probe

assert load_dotenv()

# %%
mcontext = MContext(Path("../output/comparison"))
activations_dataset_p1 = mcontext.download_cached(
    "activations_dataset",
    path="s3://repeng/datasets/activations/pythia_2024-01-23_v1.pickle",
    to="pickle",
).get()
activations_dataset_p2 = mcontext.download_cached(
    "activations_dataset_p2",
    path="s3://repeng/datasets/activations/pythia_2024-01-23_v1_pythia-6.9b.pickle",
    to="pickle",
).get()
activations_dataset = activations_dataset_p1 + activations_dataset_p2
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


# %%
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
    # "all",
    # "representation-engineering",
    # "geometry-of-truth",
    "geometry-of-truth-cities-with-neg",
    "geometry-of-truth-cities-with-neg",
    "arc_challenge",
    "arc_easy",
    "geometry_of_truth-cities",
    "geometry_of_truth-sp_en_trans",
    "geometry_of_truth-neg_sp_en_trans",
    "geometry_of_truth-larger_than",
    "geometry_of_truth-smaller_than",
    "geometry_of_truth-cities_cities_conj",
    "geometry_of_truth-cities_cities_disj",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
    "truthful_model_written",
    "true_false",
]
probe_ids: list[ProbeId] = [
    "lat",
    "mmp",
    "mmp-iid",
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


evaluation_dataset_ids = sorted(
    set(row.dataset_id for row in activations_dataset if row.split == "validation")
)

probe_eval_specs = probes.join(
    probe_train_specs,
    lambda _, probe, spec: (probe, spec),
).flat_map(
    lambda key, probe_and_spec: {
        f"{key}-{evaluation_dataset_id}": ProbeEvalSpec(
            probe_and_spec[1],
            probe_and_spec[0],
            evaluation_dataset_id,
        )
        for evaluation_dataset_id in evaluation_dataset_ids
    }
)
probe_evaluations = probe_eval_specs.map_cached(
    "probe_evaluate",
    lambda _, eval_spec: evaluate_probe(
        eval_spec.probe,
        prepare_probe_arrays(
            eval_spec.train_spec.llm_id,
            [eval_spec.dataset_id],
            split="validation",
            point_name=point_ids_by_llm[eval_spec.train_spec.llm_id][
                eval_spec.train_spec.point_id
            ],
            token_idx_from_back=eval_spec.train_spec.token_idx_from_back,
        ).labeled,
    ),
    to=ProbeEvalResult,
)

# %%
df = probe_evaluations.join(
    probe_eval_specs,
    lambda _, evaluation, spec: dict(
        **asdict(spec.train_spec),
        **evaluation.model_dump(),
        eval_dataset_id=spec.dataset_id,
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

# %% plot ROC curves
df_subset = df.copy()
df_subset = df_subset[df_subset["point_id"] == "p100"]
df_subset = df_subset.drop(columns=["point_id"])
df_subset = df_subset[df_subset["llm_id"] == "pythia-6.9b"]
df_subset = df_subset.drop(columns=["llm_id"])  # type: ignore
df_subset["probe_method"] = (
    df_subset["probe_id"] + "+" + df_subset["dataset_collection_id"]
)
df_subset = df_subset.drop(columns=["probe_id", "dataset_collection_id"])

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(4 * 5, 4 * 5))
df_subset = df_subset.sort_values("eval_dataset_id")  # type: ignore
probe_methods = df_subset["probe_method"].unique().tolist()
for eval_dataset_id, ax in zip(df_subset["eval_dataset_id"].unique(), axs.flatten()):
    ax.set_title(eval_dataset_id)
    for probe_method in probe_methods:
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


# %% plot probe performance by model size
df_subset = df.copy()
df_subset = df_subset[df_subset["point_id"] == "p90"]
df_subset = df_subset[df_subset["probe_id"] == "lat"]
df_subset = df_subset.drop(columns=["point_id", "probe_id"])
g = (
    sns.FacetGrid(
        df_subset,
        col="eval_dataset_id",
        col_wrap=4,
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

# %% bin
# # %%
# df_subset = df.copy()
# df_subset = df_subset[df_subset["probe_id"] == "lat"]
# df_subset = df_subset[df_subset["dataset_collection_id"] == "all"]
# sns.lineplot(data=df_subset, x="point_id", y="f1_score", hue="llm_id", errorbar=None)
# plt.xticks(rotation=90)
# plt.show()

# # %%
# df_subset = df.copy()
# df_subset = df_subset[df_subset["llm_id"] == "pythia-6.9b"]
# df_subset = df_subset[df_subset["point_id"] == "h21"]
# df_subset = df_subset[df_subset["probe_id"] == "mmp"]
# # df_subset = df_subset[df_subset["dataset_collection_id"] == "geometry-of-truth"]
# sns.barplot(
#     data=df_subset, x="eval_dataset_id", y="f1_score", hue="dataset_collection_id"
# )
# plt.xticks(rotation=90)
# plt.show()

# # %%
# df_subset = df.copy()
# df_subset = df_subset[df_subset["eval_dataset_id"] == "geometry_of_truth-cities"]
# # df_subset = df_subset[df_subset["point_id"] == "h21"]
# df_subset = df_subset[df_subset["dataset_collection_id"] == "all"]
# df_subset = df_subset[df_subset["probe_id"] == "mmp"]
# # df_subset = df_subset[df_subset["dataset_collection_id"] == "geometry-of-truth"]
# sns.lineplot(data=df_subset, x="point_id", y="f1_score", hue="llm_id")
# plt.xticks(rotation=90)
# plt.show()
