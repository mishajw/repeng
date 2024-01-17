from pathlib import Path

import torch
from dotenv import load_dotenv
from mppr import mppr

from repeng.activations.inference import ActivationRow, get_model_activations
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.collections import PAIRED_DATASET_IDS, get_datasets
from repeng.datasets.filters import limit_dataset_and_split_fn
from repeng.datasets.types import BinaryRow
from repeng.models.llms import LlmId
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points

assert load_dotenv()


class BinaryRowWithLlm(BinaryRow):
    llm_id: LlmId


def create_activations_dataset(
    tag: str,
    num_samples_per_dataset: int,
    llm_ids: list[LlmId],
    device: torch.device,
) -> list[ActivationResultRow]:
    inputs = (
        mppr.init(
            "init",
            Path("../output/comparison"),
            init_fn=lambda: get_datasets(PAIRED_DATASET_IDS),
            to=BinaryRow,
        )
        .filter(
            limit_dataset_and_split_fn(num_samples_per_dataset),
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
        inputs.map(
            "activations",
            fn=lambda _, value: get_model_activations(
                load_llm_oioo(
                    value.llm_id,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                value.text,
            ),
            to=ActivationRow,
        )
        .join(
            inputs,
            lambda _, activations, input: ActivationResultRow(
                dataset_id=input.dataset_id,
                split=input.split,
                label=input.is_true,
                activations=activations.activations[get_points(input.llm_id)[-1].name],
                pair_id=input.pair_id,
                llm_id=input.llm_id,
            ),
        )
        .upload(
            f"s3://repeng/datasets/activations/{tag}.jsonl",
            to=ActivationResultRow,
        )
    ).get()
