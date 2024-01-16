# %%
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from mppr import mppr

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import (
    Activation,
    prepare_activations_for_probes,
)
from repeng.datasets.collections import PAIRED_DATASET_IDS, get_datasets
from repeng.datasets.filters import limit_dataset_and_split_fn
from repeng.datasets.types import BinaryRow
from repeng.evals.probes import evaluate_probe
from repeng.models.llms import LlmId
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.probes.base import BaseProbe
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import (
    LatTrainingConfig,
    train_lat_probe,
)
from repeng.probes.mean_mass_probe import train_mmp_probe

assert load_dotenv()

# %%
llm_ids: list[LlmId] = ["gpt2", "pythia-70m"]


# %%
class BinaryRowWithLlm(BinaryRow):
    llm_id: LlmId


inputs = (
    mppr.init(
        "init",
        Path("../output/comparison"),
        init_fn=lambda: get_datasets(PAIRED_DATASET_IDS),
        to=BinaryRow,
    )
    .filter(
        limit_dataset_and_split_fn(100),
    )
    .flat_map(
        lambda key, row: {
            f"{key}-{llm_id}": BinaryRowWithLlm(**row.model_dump(), llm_id=llm_id)
            for llm_id in llm_ids
        }
    )
    # avoids constantly reloading models with OIOO
    .sort(lambda _, row: row.llm_id)
)
print(len(inputs.get()))


# %%
@dataclass
class ActivationAndInputRow(Activation):
    split: str
    llm_id: LlmId


activations_and_inputs = (
    inputs.map(
        "activations",
        fn=lambda _, value: get_model_activations(
            load_llm_oioo(value.llm_id),
            value.text,
        ),
        to="pickle",
    )
    .join(
        inputs,
        lambda _, activations, input: ActivationAndInputRow(
            dataset_id=input.dataset_id,
            split=input.split,
            label=input.is_true,
            activations=activations.activations[get_points(input.llm_id)[-1].name],
            pair_id=input.pair_id,
            llm_id=input.llm_id,
        ),
    )
    .upload("s3://repeng/comparison/activations/gpt2", to="pickle")
)

# %%
probe_arrays = prepare_activations_for_probes(
    activations_and_inputs.filter(
        lambda _, row: row.split == "train" and row.llm_id == "gpt2"
    ).get()
)
probe_arrays_validation = prepare_activations_for_probes(
    activations_and_inputs.filter(
        lambda _, row: row.split == "validation" and row.llm_id == "gpt2"
    ).get()
)

# %%
probes: dict[str, BaseProbe] = dict(
    ccs=train_ccs_probe(
        probe_arrays.paired,
        CcsTrainingConfig(),
    ),
    lat=train_lat_probe(
        probe_arrays.activations,
        LatTrainingConfig(),
    ),
    mmp=train_mmp_probe(
        probe_arrays.labeled,
    ),
)

# %%
pd.DataFrame(
    [
        dict(
            name=name,
            **evaluate_probe(
                probe,
                probe_arrays_validation.labeled,
            ).model_dump(),
        )
        for name, probe in probes.items()
    ]
)
