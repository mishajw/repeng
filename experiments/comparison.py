# %%
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from mppr import MContext
from pydantic import BaseModel

from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import DatasetId, Split
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    DatasetCollectionIdFilter,
    resolve_dataset_ids,
)
from repeng.datasets.elk.utils.filters import (
    DatasetCollectionFilter,
    DatasetFilter,
    DatasetIdFilter,
)
from repeng.evals.logits import eval_logits_by_question, eval_logits_by_row
from repeng.evals.probes import eval_probe_by_question, eval_probe_by_row
from repeng.models.llms import LlmId
from repeng.models.points import get_points
from repeng.probes.base import BaseProbe
from repeng.probes.collections import ProbeMethod, train_probe

assert load_dotenv("../.env")

# %%
"""
Gets a pre-calculated dataset of activations.

See experiments/comparison_2024-01-30.sh for the script that produced the dataset.
"""

path = Path("../output/comparison")
mcontext = MContext(path)
activation_results: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results",
    path="s3://repeng/datasets/activations/datasets_2024-02-07_v1.pickle",
    to="pickle",
).get()
print(set(row.llm_id for row in activation_results))
print(set(row.dataset_id for row in activation_results))
print(set(row.split for row in activation_results))
dataset = ActivationArrayDataset(activation_results)


# %%
"""
Pipeline for training & evaluating probes.
"""


@dataclass
class TrainSpec:
    llm_id: LlmId
    dataset: DatasetFilter
    probe_method: ProbeMethod
    point_name: str
    token_idx: int


@dataclass
class EvalSpec:
    train_spec: TrainSpec
    probe: BaseProbe
    dataset: DatasetFilter


class PipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    train_dataset: str
    eval_dataset: str
    probe_method: ProbeMethod
    point_name: str
    token_idx: int
    accuracy: float
    accuracy_hparams: float


token_idxs: list[int] = [-1]


def run_pipeline(
    llm_ids: list[LlmId],
    train_datasets: Sequence[DatasetFilter],
    eval_datasets: Sequence[DatasetFilter],
    probe_methods: list[ProbeMethod],
    point_skip: int | None,
) -> list[PipelineResultRow]:
    train_specs = mcontext.create(
        {
            "-".join(
                [llm_id, str(train_dataset), probe_method, point.name, str(token_idx)]
            ): TrainSpec(
                llm_id=llm_id,
                dataset=train_dataset,
                probe_method=probe_method,
                point_name=point.name,
                token_idx=token_idx,
            )
            for llm_id in llm_ids
            for train_dataset in train_datasets
            for probe_method in probe_methods
            for point in get_points(llm_id)[::point_skip]
            for token_idx in token_idxs
        }
    )
    probes = train_specs.map_cached(
        "probe_train",
        lambda _, spec: train_probe(
            spec.probe_method,
            dataset.get(
                llm_id=spec.llm_id,
                dataset_filter=spec.dataset,
                split="train",
                point_name=spec.point_name,
                token_idx=spec.token_idx,
                limit=None,
            ),
        ),
        to="pickle",
    ).filter(lambda _, probe: probe is not None)
    return (
        probes.join(
            train_specs,
            lambda _, probe, spec: (probe, spec),
        )
        .flat_map(
            lambda key, probe_and_spec: {
                f"{key}-{eval_dataset}": EvalSpec(
                    train_spec=probe_and_spec[1],
                    probe=cast(BaseProbe, probe_and_spec[0]),
                    dataset=eval_dataset,
                )
                for eval_dataset in eval_datasets
            }
        )
        .map_cached(
            "probe_evaluate",
            _eval_probe,
            to=PipelineResultRow,
        )
        .get()
    )


def _eval_probe(_: str, spec: EvalSpec) -> PipelineResultRow:
    return PipelineResultRow(
        llm_id=spec.train_spec.llm_id,
        train_dataset=spec.train_spec.dataset.get_name(),
        eval_dataset=spec.dataset.get_name(),
        probe_method=spec.train_spec.probe_method,
        point_name=spec.train_spec.point_name,
        token_idx=spec.train_spec.token_idx,
        accuracy=_eval_probe_on_split(
            spec.probe, spec.train_spec, spec.dataset, "validation"
        ),
        accuracy_hparams=_eval_probe_on_split(
            spec.probe, spec.train_spec, spec.dataset, "train"
        ),
    )


def _eval_probe_on_split(
    probe: BaseProbe,
    train_spec: TrainSpec,
    eval_dataset: DatasetFilter,
    split: Split,
) -> float:
    arrays = dataset.get(
        llm_id=train_spec.llm_id,
        dataset_filter=eval_dataset,
        split=split,
        point_name=train_spec.point_name,
        token_idx=train_spec.token_idx,
        limit=100,
    )
    question_result = None
    if arrays.groups is not None:
        question_result = eval_probe_by_question(
            probe,
            activations=arrays.activations,
            labels=arrays.labels,
            groups=arrays.groups,
        )
        return question_result.accuracy
    else:
        row_result = eval_probe_by_row(
            probe, activations=arrays.activations, labels=arrays.labels
        )
        return row_result.accuracy


# %%
"""
Pipeline for evaluating LLM performance based on logprobs.
"""


@dataclass
class LogprobEvalSpec:
    llm_id: LlmId
    dataset: DatasetFilter


class LogprobsPipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    eval_dataset: str
    accuracy: float
    row_accuracy: float
    row_roc_auc: float
    question_accuracy: float | None


def run_logprobs_pipeline(
    llm_ids: list[LlmId],
    eval_datasets: Sequence[DatasetFilter],
) -> list[LogprobsPipelineResultRow]:
    return (
        mcontext.create(
            {
                f"{llm_id}-{eval_dataset}": LogprobEvalSpec(llm_id, eval_dataset)
                for llm_id in llm_ids
                for eval_dataset in eval_datasets
            }
        )
        .map_cached(
            "logprob_evaluate",
            lambda _, spec: _eval_logprobs(spec),
            to="pickle",
        )
        .get()
    )


def _eval_logprobs(spec: LogprobEvalSpec) -> LogprobsPipelineResultRow:
    arrays = dataset.get(
        llm_id=spec.llm_id,
        dataset_filter=spec.dataset,
        split="validation",
        point_name="logprobs",
        token_idx=-1,
        limit=None,
    )
    row_result = eval_logits_by_row(
        logits=arrays.activations,
        labels=arrays.labels,
    )
    question_result = None
    if arrays.groups is not None:
        question_result = eval_logits_by_question(
            logits=arrays.activations,
            labels=arrays.labels,
            groups=arrays.groups,
        )
    return LogprobsPipelineResultRow(
        llm_id=spec.llm_id,
        eval_dataset=spec.dataset.get_name(),
        accuracy=question_result.accuracy if question_result else row_result.accuracy,
        row_accuracy=row_result.accuracy,
        question_accuracy=question_result.accuracy if question_result else None,
        row_roc_auc=row_result.roc_auc_score,
    )


# %%
"""
Utilities for visualizing the results.
"""

DIMS = {
    "llm_id",
    "train_dataset",
    "eval_dataset",
    "probe_method",
    "point_name",
    "token_idx",
}


def to_dataframe(
    results: Sequence[PipelineResultRow | LogprobsPipelineResultRow],
) -> pd.DataFrame:
    df = pd.DataFrame([row.model_dump() for row in results])
    df["is_supervised"] = df["probe_method"].isin(["lr", "lr-grouped", "mmp"])
    return df


def select_best(
    df: pd.DataFrame, column: str, metric: str, extra_dims: list[str] | None = None
) -> pd.DataFrame:
    extra_dims = extra_dims or []
    return (
        df.sort_values(metric, ascending=False)
        .groupby(list(DIMS - {column} | set(extra_dims)))
        .first()
        .reset_index()
    )


# %%
"""
Q0: Simple test. Do we get 80% accuracy on arc_easy?
"""
results = run_pipeline(
    llm_ids=["Llama-2-7b-chat-hf"],
    train_datasets=[DatasetIdFilter("arc_easy")],
    eval_datasets=[DatasetIdFilter("arc_easy")],
    probe_methods=["lr", "lat"],
    point_skip=4,
)
df = to_dataframe(results)
px.line(
    df,
    x="point_name",
    y="accuracy",
    color="probe_method",
    line_dash="is_supervised",
)

# %%
"""
Q1: Does adding more datasets improve generalization?
"""


def sample(
    dataset_collection_id: DatasetCollectionId, *, seed: int, k: int | None
) -> list[DatasetId]:
    dataset_ids = resolve_dataset_ids(dataset_collection_id)
    if k is None:
        return dataset_ids
    random.seed(seed)
    return random.sample(dataset_ids, k=k)


datasets_multi = [
    DatasetCollectionFilter(
        f"{dataset}-{size_name}-{i}", sample(dataset, seed=i, k=size)
    )
    for dataset, sizes in [
        ("dlk", [3, 5, None]),
        ("repe", [2, 3, None]),
        ("got", [2, 3, None]),
    ]
    for size, n_iters, size_name in zip(
        sizes,
        [5, 5, 1],
        ["small", "medium", "large"],
    )
    for i in range(n_iters)
]
results_multi = run_pipeline(
    llm_ids=["Llama-2-7b-chat-hf"],
    train_datasets=datasets_multi,
    eval_datasets=[
        DatasetCollectionIdFilter("dlk-val"),
        DatasetCollectionIdFilter("repe-val"),
        DatasetCollectionIdFilter("got-val"),
        DatasetIdFilter("truthful_qa"),
    ],
    probe_methods=["lr"],
    point_skip=4,
)
results_single = run_pipeline(
    llm_ids=["Llama-2-7b-chat-hf"],
    train_datasets=[
        DatasetIdFilter(dataset)
        for collection in ["dlk", "repe", "got"]
        for dataset in resolve_dataset_ids(cast(DatasetCollectionId, collection))
    ],
    eval_datasets=[
        DatasetCollectionIdFilter("dlk-val"),
        DatasetCollectionIdFilter("repe-val"),
        DatasetCollectionIdFilter("got-val"),
        DatasetIdFilter("truthful_qa"),
    ],
    probe_methods=["lr"],
    point_skip=4,
)

# %%
df_single = to_dataframe(results_single)
dlk_datasets = resolve_dataset_ids("dlk")
repe_datasets = resolve_dataset_ids("repe")
got_datasets = resolve_dataset_ids("got")
df_single["train_dataset"] = df_single["train_dataset"].apply(
    lambda d: "dlk" if d in dlk_datasets else "repe" if d in repe_datasets else "got"
)
best_train_dataset_idxs = df_single.groupby(list(DIMS - {"point_name"}))[
    "accuracy_hparams"
].idxmax()
df_single = df_single.loc[best_train_dataset_idxs]
df_single["train_subset"] = "single-best"
df_single["single_best"] = True

df_multi = to_dataframe(results_multi)
df_multi["train_subset"] = df_multi["train_dataset"].apply(lambda d: d.split("-")[1])
df_multi["train_dataset"] = df_multi["train_dataset"].apply(lambda d: d.split("-")[0])
df_multi = (
    df_multi.groupby(list(DIMS | {"train_subset"}))["accuracy"]
    .agg(accuracy="mean", accuracy_min="min", accuracy_max="max")  # type: ignore
    .reset_index()
)
df_multi["accuracy_min"] = df_multi["accuracy"] - df_multi["accuracy_min"]
df_multi["accuracy_max"] = df_multi["accuracy_max"] - df_multi["accuracy"]
df_multi["single_best"] = False

df = pd.concat([df_multi, df_single])
df = select_best(df, "point_name", "accuracy_hparams", extra_dims=["train_subset"])
fig = px.bar(
    df,
    title="Q1: Does adding more datasets improve generalization?",
    x="train_subset",
    y="accuracy",
    error_y="accuracy_max",
    error_y_minus="accuracy_min",
    color="single_best",
    facet_col="eval_dataset",
    facet_row="train_dataset",
    category_orders={
        "train_dataset": ["dlk", "repe", "got"],
        "train_subset": ["small", "medium", "large", "single-best"],
        "eval_dataset": ["dlk-val", "repe-val", "got-val", "truthful_qa"],
    },
)
fig.update_layout(height=600, width=800, showlegend=False)
fig.write_image(path / "q1_dataset_generalization.png")
fig.show()
df[["train_dataset", "train_subset"]].value_counts()
