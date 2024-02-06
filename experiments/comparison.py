# %%
import random
import re
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
from repeng.datasets.elk.types import DatasetId
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

mcontext = MContext(Path("../output/comparison"))
activation_results_nonchat: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results_nonchat",
    path="s3://repeng/datasets/activations/datasets_2024-02-05_v3.pickle",
    to="pickle",
).get()
activation_results_chat: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results_chat",
    path="s3://repeng/datasets/activations/datasets_2024-02-05_v3_chat.pickle",
    to="pickle",
).get()
activation_results = activation_results_nonchat + activation_results_chat
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
    row_accuracy: float
    row_roc_auc: float
    question_accuracy: float | None


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
    arrays = dataset.get(
        llm_id=spec.train_spec.llm_id,
        dataset_filter=spec.dataset,
        split="validation",
        point_name=spec.train_spec.point_name,
        token_idx=spec.train_spec.token_idx,
        limit=100,
    )
    row_result = eval_probe_by_row(
        spec.probe, activations=arrays.activations, labels=arrays.labels
    )
    question_result = None
    if arrays.groups is not None:
        question_result = eval_probe_by_question(
            spec.probe,
            activations=arrays.activations,
            labels=arrays.labels,
            groups=arrays.groups,
        )
    return PipelineResultRow(
        llm_id=spec.train_spec.llm_id,
        train_dataset=spec.train_spec.dataset.get_name(),
        eval_dataset=spec.dataset.get_name(),
        probe_method=spec.train_spec.probe_method,
        point_name=spec.train_spec.point_name,
        token_idx=spec.train_spec.token_idx,
        accuracy=question_result.accuracy if question_result else row_result.accuracy,
        row_accuracy=row_result.accuracy,
        question_accuracy=question_result.accuracy if question_result else None,
        row_roc_auc=row_result.roc_auc_score,
    )


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


def select_best(df: pd.DataFrame, column: str, metric: str) -> pd.DataFrame:
    return (
        df.sort_values(metric, ascending=False)
        .groupby(list(DIMS - {column}))
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
    y="question_accuracy",
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


train_datasets = [
    DatasetCollectionFilter(
        f"{dataset}-{size_name}-{i}", sample(dataset, seed=i, k=size)
    )
    for dataset, sizes in [
        ("dlk", [3, 5, None]),
        ("repe", [2, 3, None]),
        ("got", [2, 3, None]),
        ("multis", [4, 8, None]),
    ]
    for size, n_iters, size_name in zip(
        sizes,
        [3, 3, 1],
        ["small", "medium", "large"],
    )
    for i in range(n_iters)
]
results = run_pipeline(
    llm_ids=["Llama-2-7b-chat-hf"],
    train_datasets=[
        *train_datasets,
        *[
            DatasetIdFilter(dataset)
            for collection in ["dlk", "repe", "got"]
            for dataset in resolve_dataset_ids(cast(DatasetCollectionId, collection))
        ],
    ],
    eval_datasets=[
        DatasetCollectionIdFilter("dlk-val"),
        DatasetCollectionIdFilter("repe-val"),
        DatasetCollectionIdFilter("got-val"),
        DatasetIdFilter("truthful_qa"),
    ],
    probe_methods=["lr", "lat"],
    point_skip=4,
)

# %%
df = to_dataframe(results)
df["train_dataset"] = df["train_dataset"].apply(lambda d: re.sub(r"-\d+$", "", d))
df = (
    df.groupby(list(DIMS))["accuracy"]
    .agg(accuracy="mean", accuracy_std="std", count="count")  # type: ignore
    .reset_index()
)
dlk_datasets = ["dlk-small", "dlk-medium", "dlk-large", *resolve_dataset_ids("dlk")]
repe_datasets = [
    "repe-small",
    "repe-medium",
    "repe-large",
    *resolve_dataset_ids("repe"),
]
got_datasets = ["got-small", "got-medium", "got-large", *resolve_dataset_ids("got")]
df["train_group"] = df["train_dataset"].apply(
    lambda d: "dlk" if d in dlk_datasets else "repe" if d in repe_datasets else "got"
)
df = select_best(df, "point_name", "accuracy")
fig = px.bar(
    df,
    x="train_dataset",
    y="accuracy",
    error_y="accuracy_std",
    color="train_group",
    facet_col="eval_dataset",
    facet_row="probe_method",
    category_orders={
        "train_dataset": dlk_datasets + repe_datasets + got_datasets,
    },
)
fig.update_layout(
    height=800,
    width=2000,
)
fig.show()
