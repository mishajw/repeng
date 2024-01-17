# %%
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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
from repeng.datasets.collections import (
    DatasetCollectionId,
    get_dataset_ids_for_collection,
)
from repeng.datasets.types import DatasetId, Split
from repeng.evals.probes import ProbeEvalResult, evaluate_probe
from repeng.models.llms import LlmId
from repeng.probes.base import BaseProbe
from repeng.probes.collections import ProbeId, train_probe

assert load_dotenv()

# %%
mcontext = MContext(Path("../output/comparison"))
activations_dataset = mcontext.download_cached(
    "activations_dataset",
    # path="s3://repeng/datasets/activations/pythia_2024-01-17_v2.jsonl",
    path="s3://repeng/datasets/activations/test-v4.jsonl",
    to=ActivationResultRow,
).get()
print(set(row.llm_id for row in activations_dataset))
print(set(row.dataset_id for row in activations_dataset))
print(set(row.split for row in activations_dataset))


# %%
@dataclass
class ProbeTrainSpec:
    llm_id: LlmId
    dataset_collection_id: DatasetCollectionId
    probe_id: ProbeId
    point_name: str


llm_ids: list[tuple[LlmId, list[str]]] = [
    ("pythia-70m", []),
    ("pythia-160m", []),
    ("pythia-410m", []),
    ("pythia-1b", []),
    ("pythia-1.4b", []),
    ("pythia-2.8b", []),
    ("pythia-6.9b", []),
]
dataset_collection_ids: list[DatasetCollectionId] = [
    "all",
    "representation-engineering",
    "geometry-of-truth",
    "persona",
]
probe_ids: list[ProbeId] = [
    "lat",
    "mmp",
]
probe_eval_specs = mcontext.create(
    {
        f"{llm_id}-{dataset_collection_id}-{probe_id}": ProbeTrainSpec(
            llm_id=llm_id,
            dataset_collection_id=dataset_collection_id,
            probe_id=probe_id,
            point_name=point_name,
        )
        for (llm_id, point_names), dataset_collection_id, probe_id in itertools.product(
            llm_ids,
            dataset_collection_ids,
            probe_ids,
        )
        for point_name in point_names
    }
)


# %%
def prepare_probe_arrays(
    llm_id: LlmId,
    dataset_ids: list[DatasetId],
    split: Split,
) -> ProbeArrays:
    return prepare_activations_for_probes(
        [
            Activation(
                dataset_id=row.dataset_id,
                pair_id=row.pair_id,
                activations=row.activations,
                label=row.label,
            )
            for row in activations_dataset
            if row.llm_id == llm_id
            and row.dataset_id in dataset_ids
            and row.split == split
        ]
    )


probes = probe_eval_specs.map_cached(
    "probe_train",
    lambda _, spec: train_probe(
        spec.probe_id,
        prepare_probe_arrays(
            spec.llm_id,
            get_dataset_ids_for_collection(spec.dataset_collection_id),
            split="train",
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


evaluation_dataset_ids = set(
    row.dataset_id for row in activations_dataset if row.split == "validation"
)

probe_eval_specs = probes.join(
    probe_eval_specs,
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
df = df.sort_values("llm_id")
df

# %%
sns.barplot(
    data=df[
        (df["eval_dataset_id"] == "arc_easy") & (df["dataset_collection_id"] == "all")
    ],
    x="llm_id",
    y="f1_score",
    hue="probe_id",
)
plt.xticks(rotation=90)

# %%
df[
    (df["eval_dataset_id"] == "arc_easy")
    & (df["dataset_collection_id"] == "geometry-of-truth")
]
