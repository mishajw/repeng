from pathlib import Path

import torch
from dotenv import load_dotenv
from mppr import MContext

from repeng.activations.inference import get_model_activations
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.collections import get_datasets
from repeng.datasets.elk.utils.filters import limit_dataset_and_split_fn
from repeng.models.llms import LlmId
from repeng.models.loading import load_llm_oioo

assert load_dotenv()

_LAST_N_TOKENS = 3


class BinaryRowWithLlm(BinaryRow):
    llm_id: LlmId


def create_activations_dataset(
    tag: str,
    llm_ids: list[LlmId],
    dataset_ids: list[DatasetId],
    num_samples_per_dataset: int,
    num_validation_samples_per_dataset: int,
    device: torch.device,
) -> list[ActivationResultRow]:
    mcontext = MContext(Path("output/create-activations-dataset"))
    inputs = (
        mcontext.create_cached(
            "init",
            init_fn=lambda: get_datasets(dataset_ids),
            to=BinaryRow,
        )
        .filter(
            limit_dataset_and_split_fn(
                train_limit=num_samples_per_dataset,
                validation_limit=num_validation_samples_per_dataset,
            )
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
    return (
        inputs.map_cached(
            "activations",
            fn=lambda _, value: get_model_activations(
                load_llm_oioo(
                    value.llm_id,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                text=value.text,
                last_n_tokens=_LAST_N_TOKENS,
            ),
            to="pickle",
        )
        .join(
            inputs,
            lambda _, activations, input: ActivationResultRow(
                dataset_id=input.dataset_id,
                split=input.split,
                label=input.is_true,
                activations=activations.activations,
                pair_id=input.pair_id,
                llm_id=input.llm_id,
            ),
        )
        .upload(
            f"s3://repeng/datasets/activations/{tag}.pickle",
            to="pickle",
        )
    ).get()
