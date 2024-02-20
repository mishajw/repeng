# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats
from dotenv import load_dotenv
from jaxtyping import Bool, Float, Int32
from mppr import MContext
from pydantic import BaseModel
from tqdm import tqdm

from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import DatasetId, Split
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from repeng.datasets.elk.utils.filters import DatasetFilter, DatasetIdFilter
from repeng.evals.logits import eval_logits_by_question, eval_logits_by_row
from repeng.evals.probes import eval_probe_by_question, eval_probe_by_row
from repeng.models.llms import LlmId
from repeng.models.points import get_points
from repeng.probes.base import BaseProbe
from repeng.probes.collections import (
    ALL_PROBES,
    GROUPED_PROBES,
    SUPERVISED_PROBES,
    ProbeMethod,
    train_probe,
)

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
    path="s3://repeng/datasets/activations/datasets_2024-02-14_v1.pickle",
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


@dataclass
class EvalResult:
    accuracy: float
    n: int


class PipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    train_dataset: str
    eval_dataset: str
    probe_method: ProbeMethod
    point_name: str
    token_idx: int
    accuracy: float
    accuracy_n: int
    accuracy_hparams: float
    accuracy_hparams_n: float


token_idxs: list[int] = [-1]
points = get_points("Llama-2-13b-chat-hf")[1::4]


def run_pipeline(
    llm_ids: list[LlmId],
    train_datasets: Sequence[DatasetFilter],
    eval_datasets: Sequence[DatasetFilter],
    probe_methods: list[ProbeMethod],
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
            for point in points
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
    result_validation = _eval_probe_on_split(
        spec.probe, spec.train_spec, spec.dataset, "validation"
    )
    result_hparams = _eval_probe_on_split(
        spec.probe, spec.train_spec, spec.dataset, "train-hparams"
    )
    return PipelineResultRow(
        llm_id=spec.train_spec.llm_id,
        train_dataset=spec.train_spec.dataset.get_name(),
        eval_dataset=spec.dataset.get_name(),
        probe_method=spec.train_spec.probe_method,
        point_name=spec.train_spec.point_name,
        token_idx=spec.train_spec.token_idx,
        accuracy=result_validation.accuracy,
        accuracy_n=result_validation.n,
        accuracy_hparams=result_hparams.accuracy,
        accuracy_hparams_n=result_hparams.n,
    )


def _eval_probe_on_split(
    probe: BaseProbe,
    train_spec: TrainSpec,
    eval_dataset: DatasetFilter,
    split: Split,
) -> EvalResult:
    arrays = dataset.get(
        llm_id=train_spec.llm_id,
        dataset_filter=eval_dataset,
        split=split,
        point_name=train_spec.point_name,
        token_idx=train_spec.token_idx,
        limit=None,
    )
    question_result = None
    if arrays.groups is not None:
        question_result = eval_probe_by_question(
            probe,
            activations=arrays.activations,
            labels=arrays.labels,
            groups=arrays.groups,
        )
        return EvalResult(accuracy=question_result.accuracy, n=question_result.n)
    else:
        row_result = eval_probe_by_row(
            probe, activations=arrays.activations, labels=arrays.labels
        )
        return EvalResult(accuracy=row_result.accuracy, n=row_result.n)


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
    "train",
    "eval",
    "algorithm",
    "layer",
}
DLK_DATASETS = resolve_dataset_ids("dlk")
REPE_DATASETS = resolve_dataset_ids("repe")
GOT_DATASETS = resolve_dataset_ids("got")
BASE_COLORS = px.colors.sequential.Plotly3
COLORS = [
    [0.0, "#222"],
    [1e-5, BASE_COLORS[0]],
    *[
        [(i + 1) / (len(BASE_COLORS) - 1), color]
        for i, color in enumerate(BASE_COLORS[1:])
    ],
]


def to_dataframe(
    results: Sequence[PipelineResultRow | LogprobsPipelineResultRow],
) -> pd.DataFrame:
    df = pd.DataFrame([row.model_dump() for row in results])
    df = df.rename(
        columns={
            "train_dataset": "train",
            "eval_dataset": "eval",
            "probe_method": "algorithm",
        }
    )
    df["eval"] = df["eval"].replace(
        {"dlk-val": "dlk", "repe-qa-val": "repe", "got-val": "got"}
    )
    df["supervised"] = np.where(df["algorithm"].isin(SUPERVISED_PROBES), "sup", "unsup")
    df["grouped"] = np.where(
        df["algorithm"].isin(GROUPED_PROBES), "grouped", "ungrouped"
    )
    df["layer"] = df["point_name"].apply(lambda p: int(p.lstrip("h")))
    df["train_group"] = df["train"].apply(
        lambda d: (
            "dlk" if d in DLK_DATASETS else "repe" if d in REPE_DATASETS else "got"
        )
    )
    df["algorithm"] = df["algorithm"].str.upper()
    df = df.drop(columns=["point_name", "token_idx"])
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
datasets = [
    DatasetIdFilter(dataset)
    for collection in ["dlk", "repe", "got"]
    for dataset in resolve_dataset_ids(cast(DatasetCollectionId, collection))
]
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=datasets,
    eval_datasets=datasets,
    probe_methods=ALL_PROBES,
)

# %%
df = to_dataframe(results_truthful_qa)
thresholds_idx = df.query("train == eval").groupby("eval")["accuracy_hparams"].idxmax()
thresholds = (
    df.loc[thresholds_idx][["eval", "accuracy", "accuracy_n"]]
    .rename(
        columns={
            "accuracy": "threshold",
            "accuracy_n": "threshold_n",
        }
    )
    .set_index("eval")
)
df = df.join(thresholds, on="eval")
df  # type: ignore


# %%
"""
Add pass/fail generalization metrics.
"""
P_VALUE = 0.95


def generalizes(
    accuracy: Float[np.ndarray, "n"],  # noqa: F821
    accuracy_n: Int32[np.ndarray, "n"],  # noqa: F821
    threshold: Float[np.ndarray, "n"],  # noqa: F821
    threshold_n: Int32[np.ndarray, "n"],  # noqa: F821
) -> Bool[np.ndarray, "n"]:  # noqa: F821
    # See variance in https://en.wikipedia.org/wiki/Binomial_distribution
    accuracy_std = (accuracy * (1 - accuracy)) ** 0.5
    threshold_std = (threshold * (1 - threshold)) ** 0.5
    accuracy_stderr = accuracy_std / (accuracy_n**0.5)
    threshold_stderr = threshold_std / (threshold_n**0.5)
    diff_mean = accuracy - threshold
    diff_stderr = ((accuracy_stderr**2) + (threshold_stderr**2)) ** 0.5
    z_value = scipy.stats.norm.ppf(P_VALUE)
    diff_ci_upper = diff_mean + z_value * diff_stderr
    return diff_ci_upper >= 0


df["generalizes"] = generalizes(
    df["accuracy"].to_numpy(),
    df["accuracy_n"].to_numpy(),
    df["threshold"].to_numpy(),
    df["threshold_n"].to_numpy(),
)
df["generalizes_hparams"] = generalizes(
    df["accuracy_hparams"].to_numpy(),
    df["accuracy_hparams_n"].to_numpy(),
    df["threshold"].to_numpy(),
    df["threshold_n"].to_numpy(),
)
df  # type: ignore

# %%
"""
Add recovered accuracy metrics.
"""
df["recovered_accuracy"] = df["accuracy"] / df["threshold"].clip(0, 1)
df["recovered_accuracy_hparams"] = df["accuracy_hparams"] / df["threshold"].clip(0, 1)
df  # type: ignore

# %%
"""
Rank probes by generalization performance.

Use a pair-wise comparison score, where a probe gets a point if it generalizes better
on all datasets that *neither* probe has been trained on.
"""
probes = (
    df.sort_values("eval")
    .groupby(["train", "algorithm", "layer"])[["eval", "generalizes_hparams"]]
    .agg(list)
    .reset_index()
)
results = []
probe1: Any
probe2: Any
for probe1 in tqdm(probes.itertuples()):
    for probe2 in probes.itertuples():
        generalizes_sum1 = 0
        generalizes_sum2 = 0
        for eval1, generalizes1, eval2, generalizes2 in zip(
            probe1.eval,
            probe1.generalizes_hparams,
            probe2.eval,
            probe2.generalizes_hparams,
        ):
            assert eval1 == eval2, (eval1, eval2)
            if eval1 == probe1.train or eval1 == probe2.train:
                continue
            generalizes_sum1 += generalizes1
            generalizes_sum2 += generalizes2
        results.append(
            dict(
                train=probe1.train,
                algorithm=probe1.algorithm,
                layer=probe1.layer,
                score=1 if generalizes_sum1 > generalizes_sum2 else 0,
            )
        )

(
    df.query("train != eval")
    .groupby(["train", "algorithm", "layer"])[
        [
            "generalizes",
            "generalizes_hparams",
            "recovered_accuracy",
            "recovered_accuracy_hparams",
        ]
    ]
    .mean()
    .reset_index()
    .join(
        pd.DataFrame(results).groupby(["train", "algorithm", "layer"])["score"].sum(),
        on=["train", "algorithm", "layer"],
    )
    .sort_values("score", ascending=False)
    .head(20)
)

# %%
train_order = (
    df.groupby(["train"])["generalizes"]
    .mean()
    .sort_values(ascending=False)  # type: ignore
    .index.to_list()
)
algorithm_order = (
    df.groupby(["algorithm"])["generalizes"]
    .mean()
    .sort_values(ascending=False)  # type: ignore
    .index.to_list()
)

# %%
best_train = "rte"
best_algorithm = "PCA-G"
best_layer = 17

df_best_probes = (
    df.copy()
    .query("train != eval")
    .query(f"train == '{best_train}'")
    .query(f"layer == {best_layer}")
    .groupby(["algorithm", "eval"])["generalizes"]
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_best_probes.pivot(index="algorithm", columns="eval", values="generalizes")
    .reindex(algorithm_order, axis=0)
    .reindex([d for d in train_order if d != best_train], axis=1),
    color_continuous_scale=COLORS,
    range_color=[0, 1],
    width=800,
    height=500,
)
fig.update_layout(coloraxis_showscale=False)
fig.write_image(path / "r1a_by_algorithms.png", scale=3)
fig.show()

df_best_train = (
    df.copy()
    .query(f"algorithm == '{best_algorithm}'")
    .query(f"layer == {best_layer}")
    .groupby(["train", "eval"])["generalizes"]
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_best_train.pivot(index="train", columns="eval", values="generalizes")
    .reindex(train_order, axis=0)
    .reindex([d for d in train_order if d != best_train], axis=1)
    .dropna(),
    color_continuous_scale=COLORS,
    range_color=[0, 1],
    width=800,
    height=800,
)
fig.update_layout(coloraxis_showscale=False)
fig.write_image(path / "r1b_by_train.png", scale=3)
fig.show()

df_best_layer = (
    df.copy()
    .query(f"train == '{best_train}'")
    .query(f"algorithm == '{best_algorithm}'")
    .groupby(["layer", "eval"])["generalizes"]
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_best_layer.pivot(index="layer", columns="eval", values="generalizes").reindex(
        [d for d in train_order if d != best_train], axis=1
    ),
    color_continuous_scale=COLORS,
    range_color=[0, 1],
    width=800,
)
fig.update_layout(coloraxis_showscale=False)
fig.write_image(path / "r1c_by_layer.png", scale=3)
fig.show()

# %%
id_vars = ["algorithm", "supervised", "grouped"]
fig = px.bar(
    df.copy()
    .groupby(id_vars)
    .agg(
        {
            "generalizes": "mean",
            "recovered_accuracy": "mean",
        }
    )
    .reset_index()
    .melt(id_vars=id_vars, var_name="metric", value_name="value"),
    x="algorithm",
    y="value",
    color="supervised",
    pattern_shape="grouped",
    facet_col="metric",
    category_orders={"algorithm": algorithm_order},
    width=800,
    height=400,
)
fig.update_yaxes(matches=None)
fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
fig.write_image(path / "r2_probes.png", scale=3)
fig.show()

# %%
id_vars = ["train", "train_group"]
fig = px.bar(
    df.copy()
    .groupby(id_vars)
    .agg(
        {
            "generalizes": "mean",
            "recovered_accuracy": "mean",
        }
    )
    .reset_index()
    .melt(id_vars=id_vars, var_name="metric", value_name="value"),
    x="train",
    y="value",
    color="train_group",
    facet_col="metric",
    category_orders={"train": train_order},
)
fig.update_yaxes(matches=None)
fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
fig.write_image(path / "r3a_datasets.png", scale=3)
fig.show()


# %%
dataset_ids = DLK_DATASETS + REPE_DATASETS + GOT_DATASETS
df_train = df.copy()
df_train = df_train[df_train["train"].isin(dataset_ids)]  # type: ignore
df_train = (
    df_train.groupby(["train", "eval"])["generalizes"]  # type: ignore
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_train.pivot(index="train", columns="eval", values="generalizes")
    .reindex(train_order, axis=0)
    .reindex(train_order, axis=1),
    text_auto=".0%",  # type: ignore
    color_continuous_scale=COLORS,
    width=800,
    height=800,
)
fig.update_layout(coloraxis_showscale=False)
fig.write_image(path / "r3b_matrix.png", scale=3)
fig.show()

# %%
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=datasets,
    eval_datasets=[DatasetIdFilter("truthful_qa")],
    probe_methods=ALL_PROBES,
)
truthful_qa = (
    to_dataframe(results_truthful_qa)
    .set_index(["train", "probe_method", "layer"])["accuracy"]
    .rename("truthful_qa")  # type: ignore
)
df_truthful_qa = (
    df.query("train != eval")
    .groupby(["train", "probe_method", "layer"])["generalizes"]
    .mean()
    .reset_index()
    .join(
        truthful_qa,
        on=["train", "probe_method", "layer"],
    )
)
fig = px.box(df_truthful_qa, x="generalizes", y="truthful_qa")
fig.layout.xaxis.tickformat = ",.0%"  # type: ignore
fig.layout.yaxis.tickformat = ",.0%"  # type: ignore
# https://arxiv.org/abs/2310.01405, table 1
fig.add_hline(y=0.359, line_dash="dot", line_color="green")
fig.add_hline(y=0.503, line_dash="dot", line_color="gray")
fig.update_layout(width=800)
fig.write_image(path / "r4_truthful_qa.png", scale=3)
fig.show()
df_truthful_qa.sort_values("generalizes", ascending=False).head(20)

# %%
(truthful_qa_eval,) = run_logprobs_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    eval_datasets=[DatasetIdFilter("truthful_qa")],
)
truthful_qa_eval.model_dump()


# %%
"""
Validating results 1: Simple test. Do we get 80% accuracy on arc_easy?
"""
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=[DatasetIdFilter("arc_easy")],
    eval_datasets=[DatasetIdFilter("arc_easy")],
    probe_methods=["ccs", "lat", "dim", "lda", "lr", "lr-g", "pca", "pca-g", "rand"],
)
df = to_dataframe(results_truthful_qa).sort_values(["supervised", "algorithm", "layer"])
px.line(
    df,
    x="layer",
    y="accuracy",
    color="algorithm",
    line_dash="supervised",
)

# %%
"""
Validating results 2: Do we get the same accuracy as the GoT paper?
"""
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=[
        DatasetIdFilter("got_cities"),
        DatasetIdFilter("got_larger_than"),
        DatasetIdFilter("got_sp_en_trans"),
    ],
    eval_datasets=[
        DatasetIdFilter("got_cities"),
        DatasetIdFilter("got_larger_than"),
        DatasetIdFilter("got_sp_en_trans"),
    ],
    probe_methods=["dim", "lda"],
)

df = to_dataframe(results_truthful_qa)
px.line(
    df.query("eval == train").sort_values(["layer", "is_supervised"]),
    x="layer",
    y="accuracy",
    facet_col="algorithm",
    facet_row="train",
    width=800,
    height=600,
)

# %%
"""
Validating results 3: Do we get the same accuracy as the RepE paper?
"""
train_datasets: list[DatasetId] = [
    "race",
    "open_book_qa",
    "arc_easy",
    "arc_challenge",
]
eval_datasets: list[DatasetId] = [
    "race/qa",
    "open_book_qa/qa",
    "arc_easy/qa",
    "arc_challenge/qa",
]
eval_datasets = train_datasets
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=list(map(DatasetIdFilter, train_datasets)),
    eval_datasets=list(map(DatasetIdFilter, eval_datasets)),
    probe_methods=["lat"],
)

df = to_dataframe(results_truthful_qa)
df = df[df["train"] == df["eval"]]  # .str.rstrip("/qa")]
fig = px.line(
    df.sort_values("layer"),  # type: ignore
    x="layer",
    y="accuracy",
    color="train",
    color_discrete_map={
        "race": "red",
        "open_book_qa": "blue",
        "arc_easy": "green",
        "arc_challenge": "orange",
    },
    range_y=[0, 1],
)
fig.add_hline(y=0.459, line_dash="dot", line_color="red")
fig.add_hline(y=0.547, line_dash="dot", line_color="blue")
fig.add_hline(y=0.803, line_dash="dot", line_color="green")
fig.add_hline(y=0.532, line_dash="dot", line_color="orange")
fig.show()
