# %%
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd
import plotly.express as px
from mppr import MContext
from pydantic import BaseModel
from sklearn.decomposition import PCA

from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.utils.filters import DatasetFilter, DatasetIdFilter
from repeng.evals.logits import eval_logits_by_question, eval_logits_by_row
from repeng.evals.probes import eval_probe_by_question
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.base import BaseProbe
from repeng.probes.collections import SUPERVISED_PROBES, ProbeMethod, train_probe
from repeng.probes.difference_in_means import train_dim_probe

LLM_IDS: list[LlmId] = [
    # "Llama-2-7b-hf",
    # "Llama-2-7b-chat-hf",
    "Llama-2-13b-hf",
    "Llama-2-13b-chat-hf",
    "gemma-2b",
    "gemma-2b-it",
    # "gemma-7b",
    # "gemma-7b-it",
    # "Mistral-7B",
    # "Mistral-7B-Instruct",
]
DATASETS = [
    DatasetIdFilter("boolq/simple"),
    DatasetIdFilter("imdb/simple"),
    DatasetIdFilter("race/simple"),
    DatasetIdFilter("got_cities"),
]
CHAT_MODELS: list[LlmId] = [
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "gemma-2b-it",
    "gemma-7b-it",
    "Mistral-7B-Instruct",
]
MODEL_FAMILIES: dict[LlmId, str] = {
    "Llama-2-7b-hf": "Llama-2-7b",
    "Llama-2-7b-chat-hf": "Llama-2-7b",
    "Llama-2-13b-hf": "Llama-2-13b",
    "Llama-2-13b-chat-hf": "Llama-2-13b",
    "gemma-2b": "gemma-2b",
    "gemma-2b-it": "gemma-2b",
    "gemma-7b": "gemma-7b",
    "gemma-7b-it": "gemma-7b",
    "Mistral-7B": "Mistral-7B",
    "Mistral-7B-Instruct": "Mistral-7B",
}

# %%
output = Path("../output/saliency")
mcontext = MContext(output)
activation_results: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results",
    path="s3://repeng/datasets/activations/saliency_2024-02-26_v1.pickle",
    to="pickle",
).get()
dataset = ActivationArrayDataset(activation_results)


# %%
"""
Pipeline for training and evaluating probes.
"""


@dataclass
class Spec:
    llm_id: LlmId
    dataset: DatasetFilter
    probe_method: ProbeMethod
    point_name: str


@dataclass
class EvalResult:
    accuracy: float
    n: int


class PipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    dataset: str
    probe_method: ProbeMethod
    point_name: str
    accuracy: float
    accuracy_n: int


token_idxs: list[int] = [-1]


def run_probe_pipeline(
    llm_ids: list[LlmId],
    datasets: Sequence[DatasetFilter],
    probe_methods: list[ProbeMethod],
) -> list[PipelineResultRow]:
    train_specs = mcontext.create(
        {
            "-".join(
                [llm_id, str(train_dataset), probe_method, point.name, str(token_idx)]
            ): Spec(
                llm_id=llm_id,
                dataset=train_dataset,
                probe_method=probe_method,
                point_name=point.name,
            )
            for llm_id in llm_ids
            for train_dataset in datasets
            for probe_method in probe_methods
            for point in get_points(llm_id)[1::2]
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
                token_idx=-1,
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
        .map_cached(
            "probe_evaluate",
            lambda _, args: _eval_probe(cast(BaseProbe, args[0]), args[1]),
            to=PipelineResultRow,
        )
        .get()
    )


def _eval_probe(probe: BaseProbe, spec: Spec) -> PipelineResultRow:
    arrays = dataset.get(
        llm_id=spec.llm_id,
        dataset_filter=spec.dataset,
        split="validation",
        point_name=spec.point_name,
        token_idx=-1,
        limit=None,
    )
    assert arrays.groups is not None
    question_result = eval_probe_by_question(
        probe,
        activations=arrays.activations,
        labels=arrays.labels,
        groups=arrays.groups,
    )
    return PipelineResultRow(
        llm_id=spec.llm_id,
        dataset=spec.dataset.get_name(),
        probe_method=spec.probe_method,
        point_name=spec.point_name,
        accuracy=question_result.accuracy,
        accuracy_n=question_result.n,
    )


# %%
"""
Pipeline for evaluating the zero-shot performance of models, based on logprobs.
"""


@dataclass
class LogprobEvalSpec:
    llm_id: LlmId
    dataset: DatasetFilter


class LogprobsPipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    dataset: str
    accuracy: float


def run_logprobs_pipeline(
    llm_ids: list[LlmId],
    datasets: Sequence[DatasetFilter],
) -> list[LogprobsPipelineResultRow]:
    return (
        mcontext.create(
            {
                f"{llm_id}-{eval_dataset}": LogprobEvalSpec(llm_id, eval_dataset)
                for llm_id in llm_ids
                for eval_dataset in datasets
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
        split="train",
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
        dataset=spec.dataset.get_name(),
        accuracy=question_result.accuracy if question_result else row_result.accuracy,
    )


# %%
"""
Pipeline for calculating saliency.
"""
NUM_COMPONENTS: int = 1024


@dataclass
class SaliencySpec:
    llm_id: LlmId
    dataset: DatasetFilter
    point_name: str


@dataclass
class SaliencyResult:
    llm_id: LlmId
    dataset: str
    point_name: str
    saliency: float
    saliency_rank: int
    saliency_ratio: float
    probe_accuracy: float
    probe_accuracy_n: int


def run_saliency_pipeline(
    llm_ids: list[LlmId],
    datasets: Sequence[DatasetFilter],
) -> list[SaliencyResult]:
    return (
        mcontext.create(
            {
                "-".join([llm_id, str(train_dataset), point]): SaliencySpec(
                    llm_id=llm_id,
                    dataset=train_dataset,
                    point_name=point,
                )
                for llm_id in llm_ids
                for train_dataset in datasets
                for point in _get_points(llm_id)
            }
        )
        .map_cached(
            "saliency-v2",
            lambda _, spec: _compare_stds(spec),
            "pickle",
        )
        .get()
    )


def _get_points(llm_id: LlmId) -> list[str]:
    points = [point.name for point in get_points(llm_id)]
    points_skipped = points[1::2]
    if points[-1] not in points_skipped:
        points_skipped.append(points[-1])
    return points_skipped


def _compare_stds(spec: SaliencySpec) -> SaliencyResult:
    arrays = dataset.get(
        llm_id=spec.llm_id,
        dataset_filter=spec.dataset,
        split="train",
        point_name=spec.point_name,
        token_idx=-1,
        limit=None,
    )
    probe = train_dim_probe(
        activations=arrays.activations,
        labels=arrays.labels,
    )
    pca = PCA(n_components=NUM_COMPONENTS).fit(arrays.activations)

    arrays_val = dataset.get(
        llm_id=spec.llm_id,
        dataset_filter=spec.dataset,
        split="validation",
        point_name=spec.point_name,
        token_idx=-1,
        limit=None,
    )
    truth_direction = probe.probe
    if np.linalg.norm(truth_direction) > 0:
        truth_direction /= np.linalg.norm(truth_direction)
    truth_activations = arrays_val.activations @ truth_direction
    truth_variance = np.var(truth_activations)
    total_variance = np.var(arrays_val.activations, axis=0).sum()
    saliency = truth_variance / total_variance

    pca_variances = pca.transform(arrays_val.activations).var(axis=0)
    saliency_rank = np.sum(pca_variances > truth_variance)

    saliency_ratio = truth_variance / pca_variances[0]

    assert arrays_val.groups is not None
    eval_results = eval_probe_by_question(
        probe,
        activations=arrays_val.activations,
        labels=arrays_val.labels,
        groups=arrays_val.groups,
    )

    return SaliencyResult(
        llm_id=spec.llm_id,
        dataset=spec.dataset.get_name(),
        point_name=spec.point_name,
        saliency=saliency,
        saliency_rank=saliency_rank,
        saliency_ratio=saliency_ratio,
        probe_accuracy=eval_results.accuracy,
        probe_accuracy_n=eval_results.n,
    )


# # %%
# """
# Pipeline for LR probes on PCA subsets.
# """
# COMPONENT_INDICES: list[int] = [
#     *range(1, 16),
#     *[2**i for i in range(int(math.log(16, 2)) + 1, int(math.log(512, 2)))],
#     512,
# ]


# @dataclass
# class PcaSubset:
#     components: Float[np.ndarray, "n d"]  # noqa: F722
#     spec: Spec


# class PcaPipelineResultRow(BaseModel, extra="forbid"):
#     llm_id: LlmId
#     dataset: str
#     point_name: str
#     num_components: int
#     accuracy: float
#     accuracy_n: int


# def run_pca_pipeline(
#     llm_ids: list[LlmId],
#     datasets: Sequence[DatasetFilter],
# ) -> list[PcaPipelineResultRow]:
#     train_specs = mcontext.create(
#         {
#             "-".join([llm_id, str(dataset), point.name]): Spec(
#                 llm_id=llm_id,
#                 dataset=dataset,
#                 probe_method="pca",
#                 point_name=point.name,
#             )
#             for llm_id in llm_ids
#             for dataset in datasets
#             for point in get_points(llm_id)[17:18]
#         }
#     )
#     pca_subsets = (
#         train_specs.map_cached(
#             "pca",
#             lambda _, spec: PCA(n_components=max(COMPONENT_INDICES)).fit(
#                 dataset.get(
#                     llm_id=spec.llm_id,
#                     dataset_filter=spec.dataset,
#                     split="train",
#                     point_name=spec.point_name,
#                     token_idx=-1,
#                     limit=None,
#                 ).activations
#             ),
#             to="pickle",
#         )
#         .join(train_specs, lambda _, pca, spec: (pca, spec))
#         .flat_map(
#             lambda key, args: {
#                 f"{key}-{i}": PcaSubset(
#                     components=args[0].components_[:i], spec=args[1]
#                 )
#                 for i in COMPONENT_INDICES
#             }
#         )
#     )
#     return (
#         pca_subsets.map_cached(
#             "pca_train_lr",
#             lambda _, pca_subset: _train_probe_on_pca_subset(pca_subset),
#             to="pickle",
#         )
#         .join(pca_subsets, lambda _, probe, pca_subsets: (probe, pca_subsets))
#         .map_cached(
#             "pca_eval_lr",
#             lambda _, args: _eval_probe_on_pca_subset(args[0], args[1]),
#             to=PcaPipelineResultRow,
#         )
#         .get()
#     )


# def _train_probe_on_pca_subset(pca_subset: PcaSubset) -> BaseProbe:
#     arrays = dataset.get(
#         llm_id=pca_subset.spec.llm_id,
#         dataset_filter=pca_subset.spec.dataset,
#         split="train",
#         point_name=pca_subset.spec.point_name,
#         token_idx=-1,
#         limit=None,
#     )
#     activations = arrays.activations @ pca_subset.components.T
#     return train_lr_probe(
#         LrConfig(),
#         activations=activations,
#         labels=arrays.labels,
#     )


# def _eval_probe_on_pca_subset(
#     probe: BaseProbe, pca_subset: PcaSubset
# ) -> PcaPipelineResultRow:
#     arrays = dataset.get(
#         llm_id=pca_subset.spec.llm_id,
#         dataset_filter=pca_subset.spec.dataset,
#         split="validation",
#         point_name=pca_subset.spec.point_name,
#         token_idx=-1,
#         limit=None,
#     )
#     activations = arrays.activations @ pca_subset.components.T
#     assert arrays.groups is not None
#     question_result = eval_probe_by_question(
#         probe,
#         activations=activations,
#         labels=arrays.labels,
#         groups=arrays.groups,
#     )
#     return PcaPipelineResultRow(
#         llm_id=pca_subset.spec.llm_id,
#         dataset=pca_subset.spec.dataset.get_name(),
#         point_name=pca_subset.spec.point_name,
#         num_components=pca_subset.components.shape[0],
#         accuracy=question_result.accuracy,
#         accuracy_n=question_result.n,
#     )


# results = run_pca_pipeline(
#     llm_ids=[
#         # "Llama-2-13b-hf",
#         # "Llama-2-13b-chat-hf",
#         "Llama-2-7b-hf",
#         "Llama-2-7b-chat-hf",
#     ],
#     datasets=[
#         DatasetIdFilter("boolq/simple"),
#         # DATASET_FILTER_FNS["got_cities/pos"],
#     ],
# )
# df = pd.DataFrame([r.model_dump() for r in results])
# px.line(
#     df,
#     x="num_components",
#     y="accuracy",
#     color="llm_id",
#     facet_row="dataset",
#     log_x=True,
# )

# %%
"""
Show that PCA works on chat models, but not on non-chat models.
"""
results = run_probe_pipeline(
    # llm_ids=["Llama-2-7b-hf", "Llama-2-7b-chat-hf"],
    # llm_ids=["Llama-2-13b-hf", "Llama-2-13b-chat-hf"],
    llm_ids=["gemma-2b", "gemma-2b-it"],
    # llm_ids=["gemma-7b", "gemma-7b-it"],
    datasets=[
        # DatasetIdFilter("got_cities"),
        # DatasetIdFilter("race/simple"),
        DatasetIdFilter("boolq/simple"),
    ],
    probe_methods=["lr", "pca-g"],
)
df = pd.DataFrame([r.model_dump() for r in results])
df["layer"] = df["point_name"].str.extract(r"h(\d+)").astype(int)
df["supervised"] = np.where(
    df["probe_method"].isin(SUPERVISED_PROBES), "supervised", "unsupervised"
)
df["algorithm"] = df["probe_method"].replace(
    {"lr": "Supervised (LogR)", "pca-g": "Unsupervised (PCA-G)"}
)
df["accuracy_stderr"] = np.sqrt(
    df["accuracy"] * (1 - df["accuracy"]) / df["accuracy_n"]
)
df["type"] = np.where(df["llm_id"].isin(CHAT_MODELS), "chat", "base")

fig = px.line(
    df.sort_values(["layer", "supervised", "algorithm"]),
    x="layer",
    y="accuracy",
    error_y="accuracy_stderr",
    color="algorithm",
    facet_row="type",
    width=800,
    height=500,
)
fig.update_layout(yaxis_tickformat=".0%")
fig.write_image(output / "1_lr_v_pca.png", scale=3)
fig.show()

# %%
"""
Show that chat and base models have the same zero-shot accuracy.
"""
results = run_logprobs_pipeline(
    llm_ids=LLM_IDS,
    datasets=DATASETS,
)
df = pd.DataFrame([r.model_dump() for r in results])
df["family"] = df["llm_id"].map(MODEL_FAMILIES)
df["type"] = np.where(df["llm_id"].isin(CHAT_MODELS), "chat", "base")
fig = px.bar(
    df.sort_values(["llm_id"]),
    x="type",
    y="accuracy",
    color="type",
    facet_col="dataset",
    facet_row="family",
    category_orders={
        "family": ["Llama-2-7b", "Llama-2-13b", "gemma-2b", "gemma-7b", "Mistral-7B"],
        "type": ["base", "chat"],
        "dataset": [d.get_name() for d in DATASETS],
    },
    text_auto=".1%",  # type: ignore
    width=800,
    height=800,
)
fig.write_image(output / "2_zero_shot.png", scale=3)
fig.show()

# %%
"""
Plot saliency measures for a range of models.
"""
results = run_saliency_pipeline(
    llm_ids=LLM_IDS,
    datasets=DATASETS,
)
df = pd.DataFrame([asdict(r) for r in results])
df["type"] = np.where(df["llm_id"].isin(CHAT_MODELS), "chat", "base")
df["family"] = df["llm_id"].map(MODEL_FAMILIES)
df["layer"] = df["point_name"].str.extract(r"h(\d+)").astype(int)
df["layer"] = df.apply(
    lambda r: r.layer / len(get_points(r.llm_id)),
    axis=1,
)

fig = px.line(
    df.sort_values("layer"),
    x="layer",
    y="saliency_ratio",
    color="type",
    facet_col="dataset",
    facet_row="family",
    category_orders={
        "family": ["Llama-2-7b", "Llama-2-13b", "gemma-2b", "gemma-7b", "Mistral-7B"],
        "type": ["base", "chat"],
        "dataset": [d.get_name() for d in DATASETS],
    },
    markers=True,
    width=800,
    height=1000,
)
fig.update_yaxes(matches=None)
fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
fig.write_image(output / "3_saliency.png", scale=3)
fig.show()

# %%
"""
Show correlation between saliency measure and unsupervised probe accuracy
"""
results_saliency = run_saliency_pipeline(
    llm_ids=LLM_IDS,
    datasets=DATASETS,
)
results_probe = run_probe_pipeline(
    llm_ids=LLM_IDS,
    datasets=DATASETS,
    probe_methods=["lr", "pca"],
)

# %%
df_probe = pd.DataFrame([r.model_dump() for r in results_probe])
df_probe = df_probe.query("probe_method == 'pca'").join(
    df_probe.query("probe_method == 'lr'")
    .drop(columns=["probe_method"])
    .rename({"accuracy": "threshold", "accuracy_n": "threshold_n"}, axis=1)
    .set_index(["llm_id", "dataset", "point_name"]),
    on=["llm_id", "dataset", "point_name"],
)
df_probe["recovered"] = df_probe["accuracy"] / df_probe["threshold"]

df = pd.DataFrame([asdict(r) for r in results_saliency]).join(
    df_probe.set_index(["llm_id", "dataset", "point_name"]),
    on=["llm_id", "dataset", "point_name"],
)
df["layer"] = df["point_name"].str.extract(r"h(\d+)").astype(int)
px.scatter(
    df.query("layer > 10").query("llm_id == 'Llama-2-13b-chat-hf'"),
    x="saliency_ratio",
    y="recovered",
    color="dataset",
    symbol="llm_id",
    text="layer",
)
