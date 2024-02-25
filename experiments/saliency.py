# %%
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats
from jaxtyping import Float
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
from repeng.probes.logistic_regression import train_lr_probe

CHAT_MODELS = [
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
FIRST_LAYERS: dict[str, int] = {
    "Llama-2-7b": 13,
    "Llama-2-13b": 13,
}
LAST_LAYERS: dict[str, int] = {
    "Llama-2-7b": 32,
    "Llama-2-13b": 40,
}

# %%
output = Path("output/saliency")
mcontext = MContext(output)
activation_results_p1: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results_v3",
    path="s3://repeng/datasets/activations/saliency_2024-02-23_v3.pickle",
    to="pickle",
).get()
activation_results_p2: list[ActivationResultRow] = mcontext.download_cached(
    "activations_results_p2",
    path="s3://repeng/datasets/activations/saliency_2024-02-24_v1_mistralandgemma.pickle",
    to="pickle",
).get()
activation_results = activation_results_p1 + activation_results_p2
dataset = ActivationArrayDataset(activation_results)


# %%
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


def run_pipeline(
    llm_ids: list[LlmId],
    train_datasets: Sequence[DatasetFilter],
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
            for train_dataset in train_datasets
            for probe_method in probe_methods
            for point in get_points(llm_id)[1::4]
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
            "probe_evaluate-v3",
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
        dataset=spec.dataset.get_name(),
        accuracy=question_result.accuracy if question_result else row_result.accuracy,
    )


# %%
"""
Pipeline for calculating saliency.
"""
COMPONENT_INDICES: list[int] = [
    *range(1, 16),
    *[2**i for i in range(int(math.log(16, 2)) + 1, int(math.log(1024, 2)))],
    1024,
]


@dataclass
class PcaSubset:
    components: Float[np.ndarray, "n d"]  # noqa: F722
    spec: Spec


class PcaPipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    dataset: str
    point_name: str
    num_components: int
    accuracy: float
    accuracy_n: int


def run_pca_pipeline(
    llm_ids: list[LlmId],
    train_datasets: Sequence[DatasetFilter],
) -> list[PcaPipelineResultRow]:
    train_specs = mcontext.create(
        {
            "-".join([llm_id, str(train_dataset), point.name]): Spec(
                llm_id=llm_id,
                dataset=train_dataset,
                probe_method="pca",
                point_name=point.name,
            )
            for llm_id in llm_ids
            for train_dataset in train_datasets
            for point in get_points(llm_id)[1::4]
        }
    )
    pca_subsets = (
        train_specs.map_cached(
            "pca-v2",
            lambda _, spec: PCA(n_components=max(COMPONENT_INDICES)).fit(
                dataset.get(
                    llm_id=spec.llm_id,
                    dataset_filter=spec.dataset,
                    split="train",
                    point_name=spec.point_name,
                    token_idx=-1,
                    limit=None,
                ).activations
            ),
            to="pickle",
        )
        .join(train_specs, lambda _, pca, spec: (pca, spec))
        .flat_map(
            lambda llm_id, args: {
                f"{llm_id}-{i}": PcaSubset(
                    components=args[0].components_[:i], spec=args[1]
                )
                for i in COMPONENT_INDICES
            }
        )
    )
    return (
        pca_subsets.map_cached(
            "pca-train",
            lambda _, pca_subset: _train_probe_on_pca_subset(pca_subset),
            to="pickle",
        )
        .join(pca_subsets, lambda _, probe, pca_subsets: (probe, pca_subsets))
        .map_cached(
            "pca-eval",
            lambda _, args: _eval_probe_on_pca_subset(args[0], args[1]),
            to=PcaPipelineResultRow,
        )
        .get()
    )


def _train_probe_on_pca_subset(pca_subset: PcaSubset) -> BaseProbe:
    arrays = dataset.get(
        llm_id=pca_subset.spec.llm_id,
        dataset_filter=pca_subset.spec.dataset,
        split="train",
        point_name=pca_subset.spec.point_name,
        token_idx=-1,
        limit=None,
    )
    activations = arrays.activations @ pca_subset.components.T
    return train_lr_probe(
        activations=activations,
        labels=arrays.labels,
    )


def _eval_probe_on_pca_subset(
    probe: BaseProbe, pca_subset: PcaSubset
) -> PcaPipelineResultRow:
    arrays = dataset.get(
        llm_id=pca_subset.spec.llm_id,
        dataset_filter=pca_subset.spec.dataset,
        split="validation",
        point_name=pca_subset.spec.point_name,
        token_idx=-1,
        limit=None,
    )
    activations = arrays.activations @ pca_subset.components.T
    assert arrays.groups is not None
    question_result = eval_probe_by_question(
        probe,
        activations=activations,
        labels=arrays.labels,
        groups=arrays.groups,
    )
    return PcaPipelineResultRow(
        llm_id=pca_subset.spec.llm_id,
        dataset=pca_subset.spec.dataset.get_name(),
        point_name=pca_subset.spec.point_name,
        num_components=pca_subset.components.shape[0],
        accuracy=question_result.accuracy,
        accuracy_n=question_result.n,
    )


# %%
"""
Show that PCA works on chat models, but not on non-chat models.
"""
results = run_pipeline(
    llm_ids=["Llama-2-13b-hf", "Llama-2-13b-chat-hf"],
    train_datasets=[DatasetIdFilter("boolq")],
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
fig.write_image(output / "1_lr_v_pca.png")
fig.show()


# %%
llm_ids: list[LlmId] = [
    "Llama-2-7b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-hf",
    "Llama-2-13b-chat-hf",
    # "gemma-2b",
    # "gemma-2b-it",
    # "gemma-7b",
    # "gemma-7b-it",
    # "Mistral-7B",
    # "Mistral-7B-Instruct",
]
datasets = [
    DatasetIdFilter("boolq"),
    DatasetIdFilter("got_cities"),
    DatasetIdFilter("imdb"),
    DatasetIdFilter("race"),
]

results = run_pca_pipeline(
    llm_ids=llm_ids,
    train_datasets=datasets,
)
df_pca = pd.DataFrame([r.model_dump() for r in results])

results = run_pipeline(
    llm_ids=llm_ids,
    train_datasets=datasets,
    probe_methods=["lr"],
)
df_lr = (
    pd.DataFrame([r.model_dump() for r in results])
    .set_index(["llm_id", "dataset", "point_name"])
    .rename({"accuracy": "threshold", "accuracy_n": "threshold_n"}, axis=1)
    .drop(columns=["probe_method"])
)

df = df_pca.join(df_lr, on=["llm_id", "dataset", "point_name"])
diff = df["accuracy"] - df["threshold"]
diff_stderr = np.sqrt(
    (df["accuracy"] * (1 - df["accuracy"]) / df["accuracy_n"])
    + (df["threshold"] * (1 - df["threshold"]) / df["threshold_n"])
)
df["diff_upper"] = diff + diff_stderr * scipy.stats.norm.ppf(0.95)

df["type"] = np.where(df["llm_id"].isin(CHAT_MODELS), "chat", "base")
df["family"] = df["llm_id"].map(MODEL_FAMILIES)
df["layer"] = df["point_name"].str.extract(r"h(\d+)").astype(int)
df["layer"] = df.apply(
    lambda r: (
        (r.layer - FIRST_LAYERS.get(r.family, 0))
        / (LAST_LAYERS.get(r.family, 32) - FIRST_LAYERS.get(r.family, 0))
    ),
    axis=1,
)
df = df.query("layer >= 0")

# %%
fig = px.line(
    df.sort_values(["num_components"])
    .query("diff_upper > 0")
    .groupby(["llm_id", "dataset", "point_name"])
    .first()
    .reset_index()
    .sort_values("layer"),
    x="layer",
    y="num_components",
    color="type",
    facet_col="dataset",
    facet_row="family",
    category_orders={
        "family": [
            "Llama-2-7b",
            "Llama-2-13b",
            # "gemma-2b",
            # "gemma-7b",
            # "Mistral-7B",
        ],
        "type": ["base", "chat"],
    },
    markers=True,
    width=800,
    height=500,
)
fig.write_image(output / "2_saliency.png")
fig.show()
