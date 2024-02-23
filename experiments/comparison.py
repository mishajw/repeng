# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
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
# %%
activation_results_p1: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results",
    path="s3://repeng/datasets/activations/datasets_2024-02-14_v1.pickle",
    to="pickle",
).get()
activation_results_p2: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results_p2",
    path="s3://repeng/datasets/activations/datasets_2024-02-23_truthfulqa_v1.pickle",
    to="pickle",
).get()
activation_results = activation_results_p1 + activation_results_p2
print(set(row.llm_id for row in activation_results))
print(set(row.dataset_id for row in activation_results))
print(set(row.split for row in activation_results))
dataset = ActivationArrayDataset(activation_results)

# %%
dataset = ActivationArrayDataset([])

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
    eval_splits: list[Split],
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
        "probe_train-v2",
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
            "probe_evaluate-v2",
            lambda _, spec: _eval_probe(spec, eval_splits),
            to=PipelineResultRow,
        )
        .get()
    )


def _eval_probe(spec: EvalSpec, splits: list[Split]) -> PipelineResultRow:
    result_hparams = None
    result_validation = None
    if "train-hparams" in splits:
        result_hparams = _eval_probe_on_split(
            spec.probe, spec.train_spec, spec.dataset, "train-hparams"
        )
    elif "validation" in splits:
        result_validation = _eval_probe_on_split(
            spec.probe, spec.train_spec, spec.dataset, "validation"
        )
    assert "train" not in splits, splits
    return PipelineResultRow(
        llm_id=spec.train_spec.llm_id,
        train_dataset=spec.train_spec.dataset.get_name(),
        eval_dataset=spec.dataset.get_name(),
        probe_method=spec.train_spec.probe_method,
        point_name=spec.train_spec.point_name,
        token_idx=spec.train_spec.token_idx,
        accuracy=result_validation.accuracy if result_validation else 0,
        accuracy_n=result_validation.n if result_validation else 0,
        accuracy_hparams=result_hparams.accuracy if result_hparams else 0,
        accuracy_hparams_n=result_hparams.n if result_hparams else 0,
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
BASE_COLORS = list(reversed(px.colors.sequential.YlOrRd))
# COLORS = [
#     [0.0, "#222"],
#     [1e-5, BASE_COLORS[0]],
#     *[
#         [(i + 1) / (len(BASE_COLORS) - 1), color]
#         for i, color in enumerate(BASE_COLORS[1:-1])
#     ],
#     [1 - 1e-5, BASE_COLORS[-1]],
#     [1, px.colors.sequential.Greens[len(px.colors.sequential.Greens) // 2]],
# ]
COLORS = BASE_COLORS


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
results = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=datasets,
    eval_datasets=datasets,
    probe_methods=ALL_PROBES,
    eval_splits=["validation", "train-hparams"],
)

# %%
df = to_dataframe(results)
df["recovered_accuracy"] = (
    df["accuracy"].to_numpy() / df["threshold"].to_numpy()
).clip(0, 1)
df["recovered_accuracy_hparams"] = (
    df["accuracy_hparams"].to_numpy() / df["threshold"].to_numpy()
).clip(0, 1)
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
thresholds  # type: ignore

# %%
"""
Rank probes by generalization performance.

Use a pair-wise comparison score, where a probe gets a point if it generalizes better
on all datasets that *neither* probe has been trained on.
"""
probes = (
    df.sort_values("eval")
    .groupby(["train", "algorithm", "layer"])[["eval", "recovered_accuracy_hparams"]]
    .agg(list)
    .reset_index()
)
results = []
probe1: Any
probe2: Any
for probe1 in tqdm(probes.itertuples()):
    for probe2 in probes.itertuples():
        probe1_recovered_wins = 0
        probe2_recovered_wins = 0
        probe1_perfect_wins = 0
        probe2_perfect_wins = 0
        for eval1, generalizes1, eval2, generalizes2 in zip(
            probe1.eval,
            probe1.recovered_accuracy_hparams,
            probe2.eval,
            probe2.recovered_accuracy_hparams,
        ):
            assert eval1 == eval2, (eval1, eval2)
            if eval1 == probe1.train or eval1 == probe2.train:
                continue
            if generalizes1 > generalizes2:
                probe1_recovered_wins += 1
            if generalizes1 < generalizes2:
                probe2_recovered_wins += 1
            if generalizes1 == 1 and generalizes2 < 1:
                probe1_perfect_wins += 1
            if generalizes1 < 1 and generalizes2 == 1:
                probe2_perfect_wins += 1
        results.append(
            dict(
                train=probe1.train,
                algorithm=probe1.algorithm,
                layer=probe1.layer,
                recovered_score=probe1_recovered_wins > probe2_recovered_wins,
                perfect_score=probe1_perfect_wins > probe2_perfect_wins,
            )
        )

# %%
(
    df.groupby(["train", "algorithm", "layer"])[
        ["recovered_accuracy", "recovered_accuracy_hparams"]
    ]
    .mean()
    .reset_index()
    .join(
        pd.DataFrame(results)
        .groupby(["train", "algorithm", "layer"])[["recovered_score"]]
        .mean(),
        on=["train", "algorithm", "layer"],
    )
    .sort_values("recovered_score", ascending=False)
    .head(20)
)

# %%
"""
Fix the order of train datasets & algorithms in plots.
"""
train_order = (
    df.groupby(["train"])["recovered_accuracy"]
    .mean()
    .sort_values(ascending=False)  # type: ignore
    .index.to_list()
)
algorithm_order = (
    df.groupby(["algorithm"])["recovered_accuracy"]
    .mean()
    .sort_values(ascending=False)  # type: ignore
    .index.to_list()
)

# %%
fig = px.ecdf(
    df.groupby(["algorithm", "train", "layer", "supervised"])["recovered_accuracy"]
    .mean()
    .reset_index(),
    x="recovered_accuracy",
    color="layer",
    width=800,
    height=400,
    color_discrete_sequence=px.colors.sequential.deep,
)
fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
fig.write_image(path / "r0_acc_by_layer.png", scale=3)
fig.show()

perc_above_80 = np.mean(
    df.query("layer >= 13")
    .groupby(["algorithm", "train", "layer", "supervised"])["recovered_accuracy"]
    .mean()
    > 0.8
)
print(f"Percent of layer 13+ probes with >80% recovered accuracy: {perc_above_80:.1%}")

# %%
"""
1. Examining the best probe
"""
best_train = "dbpedia_14"
best_algorithm = "DIM"
best_layer = 21

df_best_probes = (
    df.copy()
    .query("train != eval")
    .query(f"train == '{best_train}'")
    .query(f"layer == {best_layer}")
    .groupby(["algorithm", "eval"])["recovered_accuracy"]
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_best_probes.pivot(index="algorithm", columns="eval", values="recovered_accuracy")
    .reindex(algorithm_order, axis=0)
    .reindex([d for d in train_order if d != best_train], axis=1)
    .rename(index={best_algorithm: f"<b>{best_algorithm} (best)</b>"}),
    color_continuous_scale=COLORS,
    range_color=[0, 1],
    text_auto=".0%",  # type: ignore
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
    .groupby(["train", "eval"])["recovered_accuracy"]
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_best_train.pivot(index="train", columns="eval", values="recovered_accuracy")
    .reindex(train_order, axis=0)
    .reindex([d for d in train_order if d != best_train], axis=1)
    .rename(index={best_train: f"<b>{best_train} (best)</b>"})
    .dropna(),
    color_continuous_scale=COLORS,
    range_color=[0, 1],
    text_auto=".0%",  # type: ignore
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
    .groupby(["layer", "eval"])["recovered_accuracy"]
    .mean()
    .reset_index()
)
df_best_layer = df_best_layer.pivot(
    index="layer", columns="eval", values="recovered_accuracy"
)
df_best_layer.index = df_best_layer.index.map(str)
fig = px.imshow(
    df_best_layer.reindex([d for d in train_order if d != best_train], axis=1).rename(
        index={str(best_layer): f"<b>{best_layer} (best)</b>"}
    ),
    color_continuous_scale=COLORS,
    range_color=[0, 1],
    text_auto=".0%",  # type: ignore
    width=800,
    height=600,
)
fig.update_layout(coloraxis_showscale=False)
fig.write_image(path / "r1c_by_layer.png", scale=3)
fig.show()

# %%
"""
2. Examining algorithm performance
"""
id_vars = ["algorithm", "supervised", "grouped"]
fig = px.bar(
    df.copy().groupby(id_vars)["recovered_accuracy"].mean().reset_index(),
    x="algorithm",
    y="recovered_accuracy",
    color="supervised",
    pattern_shape="grouped",
    category_orders={"algorithm": algorithm_order},
    width=800,
    height=400,
    text_auto=".0%",  # type: ignore
)
fig.update_layout(yaxis_tickformat=".0%")
fig.write_image(path / "r2_probes.png", scale=3)
fig.show()

# %%

# %%
"""
2. Examining dataset performance
"""
id_vars = ["train", "train_group"]
fig = px.bar(
    df.copy().groupby(id_vars)["recovered_accuracy"].mean().reset_index(),
    x="train",
    y="recovered_accuracy",
    color="train_group",
    category_orders={"train": train_order},
    width=800,
    height=400,
    text_auto=".0%",  # type: ignore
)
fig.update_layout(yaxis_tickformat=".0%")
fig.write_image(path / "r3a_datasets.png", scale=3)
fig.show()

# %%
"""
2. Examining dataset performance: matrix
"""
dataset_ids = DLK_DATASETS + REPE_DATASETS + GOT_DATASETS
df_train = df.copy()
df_train = df_train[df_train["train"].isin(dataset_ids)]  # type: ignore
df_train = (
    df_train.groupby(["train", "eval"])["recovered_accuracy"]  # type: ignore
    .mean()
    .reset_index()
)
fig = px.imshow(
    df_train.pivot(index="train", columns="eval", values="recovered_accuracy")
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
df_sym = (
    pd.concat(
        [
            df.groupby("train")["recovered_accuracy"].mean().rename("generalizes_from"),
            df.groupby("eval")["recovered_accuracy"].mean().rename("generalizes_to"),
        ],
        axis=1,
    )
    .reset_index()
    .rename({"index": "dataset"}, axis=1)
)
df_sym["group"] = df_sym["dataset"].apply(
    lambda d: ("dlk" if d in DLK_DATASETS else "repe" if d in REPE_DATASETS else "got")
)
fig = px.scatter(
    df_sym,
    x="generalizes_from",
    y="generalizes_to",
    color="group",
    text="dataset",
    range_x=[0.66, 0.76],
    height=600,
    width=800,
)
fig.update_traces(textposition="bottom center")
fig.write_image(path / "r3c_to_and_from.png", scale=3)
fig.show()

# %%
"""
4. TruthfulQA
"""
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=datasets,
    eval_datasets=[DatasetIdFilter("truthful_qa")],
    probe_methods=ALL_PROBES,
    eval_splits=["validation"],
)
truthful_qa = (
    to_dataframe(results_truthful_qa)
    .set_index(["train", "algorithm", "layer"])["accuracy"]
    .rename("truthful_qa")  # type: ignore
)
df_truthful_qa = (
    df.query("train != eval")
    .groupby(["train", "algorithm", "layer"])["recovered_accuracy"]
    .mean()
    .reset_index()
    .join(
        truthful_qa,
        on=["train", "algorithm", "layer"],
    )
)
fig = px.scatter(
    df_truthful_qa,
    x="recovered_accuracy",
    y="truthful_qa",
    width=800,
)
fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
# https://arxiv.org/abs/2310.01405, table 1
fig.add_hline(y=0.359, line_dash="dot", line_color="green")
fig.add_hline(y=0.503, line_dash="dot", line_color="gray")
fig.write_image(path / "r4_truthful_qa.png", scale=3)
fig.show()

perc_measuring_truth = (
    (df_truthful_qa["recovered_accuracy"] > 0.8)
    & (df_truthful_qa["truthful_qa"] > 0.359)
).sum() / (df_truthful_qa["recovered_accuracy"] > 0.8).sum()
print(f"Percent of probes with >80% recovered accuracy: {perc_measuring_truth:.1%}")


# %%
"""
Validating results 1: Simple test. Do we get 80% accuracy on arc_easy?
"""
results_truthful_qa = run_pipeline(
    llm_ids=["Llama-2-13b-chat-hf"],
    train_datasets=[DatasetIdFilter("arc_easy")],
    eval_datasets=[DatasetIdFilter("arc_easy")],
    probe_methods=["ccs", "lat", "dim", "lda", "lr", "lr-g", "pca", "pca-g"],
    eval_splits=["validation", "train-hparams"],
)
dfv = to_dataframe(results_truthful_qa).sort_values(
    ["supervised", "algorithm", "layer"]
)
fig = px.line(
    dfv,
    x="layer",
    y="accuracy",
    color="algorithm",
    line_dash="supervised",
    width=800,
)
fig.write_image(path / "v1_arc_easy.png", scale=3)
fig.show()

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
    eval_splits=["validation", "train-hparams"],
)

dfv = to_dataframe(results_truthful_qa)
fig = px.line(
    dfv.query("eval == train").sort_values(["layer", "supervised"]),
    x="layer",
    y="accuracy",
    facet_col="algorithm",
    facet_row="train",
    width=800,
    height=600,
)
fig.write_image(path / "v2_got.png", scale=3)
fig.show()

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
    eval_splits=["validation", "train-hparams"],
)

dfv = to_dataframe(results_truthful_qa)
dfv = dfv[dfv["train"] == dfv["eval"]]  # .str.rstrip("/qa")]
fig = px.line(
    dfv.sort_values("layer"),  # type: ignore
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
    width=800,
)
fig.add_hline(y=0.459, line_dash="dot", line_color="red")
fig.add_hline(y=0.547, line_dash="dot", line_color="blue")
fig.add_hline(y=0.803, line_dash="dot", line_color="green")
fig.add_hline(y=0.532, line_dash="dot", line_color="orange")
fig.write_image(path / "v3_repe.png", scale=3)
fig.show()

# %%
"""
Validating results 4: Investigating LDA performance
"""
fig = px.bar(
    pd.concat(
        [
            df.query("train == 'boolq'").assign(type="train_boolq"),
            df.query("train == eval").assign(type="train_eval"),
        ]
    )
    .query("algorithm == 'LDA' or algorithm == 'DIM'")
    .query("layer == 21"),
    y="accuracy",
    x="eval",
    color="algorithm",
    facet_row="type",
    barmode="group",
    width=800,
)
fig.write_image(path / "v4_lda.png", scale=3)
fig.show()
