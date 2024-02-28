from pathlib import Path

import torch
from dotenv import load_dotenv
from mppr import MContext, MDict
from pydantic import BaseModel

from repeng.activations.inference import get_model_activations
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.fns import get_dataset
from repeng.datasets.elk.utils.limits import Limits, limit_groups
from repeng.models.llms import LlmId
from repeng.models.loading import load_llm_oioo

assert load_dotenv()


class _BinaryRowWithLlm(BinaryRow):
    llm_id: LlmId


class _Dataset(BaseModel, extra="forbid"):
    rows: dict[str, BinaryRow]


def create_activations_dataset(
    tag: str,
    llm_ids: list[LlmId],
    dataset_ids: list[DatasetId],
    group_limits: Limits,
    device: torch.device,
    num_tokens_from_end: int | None,
    layers_start: int | None,
    layers_end: int | None,
    layers_skip: int | None,
) -> list[ActivationResultRow]:
    mcontext = MContext(Path("output/create-activations-dataset"))
    dataset_ids_mdict: MDict[DatasetId] = mcontext.create(
        {dataset_id: dataset_id for dataset_id in dataset_ids},
    )
    inputs = (
        dataset_ids_mdict.map_cached(
            "datasets",
            lambda _, dataset_id: _Dataset(rows=get_dataset(dataset_id)),
            to=_Dataset,
        )
        .flat_map(lambda _, dataset: {key: row for key, row in dataset.rows.items()})
        .filter(limit_groups(group_limits))
        .flat_map(
            lambda key, row: {
                f"{key}-{llm_id}": _BinaryRowWithLlm(**row.model_dump(), llm_id=llm_id)
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
                    use_half_precision=True,
                ),
                text=value.text,
                last_n_tokens=num_tokens_from_end,
                points_start=layers_start,
                points_end=layers_end,
                points_skip=layers_skip,
            ),
            to="pickle",
        )
        .join(
            inputs,
            lambda _, activations, input: ActivationResultRow(
                dataset_id=input.dataset_id,
                split=input.split,
                answer_type=input.answer_type,
                label=input.label,
                activations=activations.activations,
                prompt_logprobs=activations.token_logprobs.sum().item(),
                group_id=input.group_id,
                llm_id=input.llm_id,
            ),
        )
        .upload(
            f"s3://repeng/datasets/activations/{tag}.pickle",
            to="pickle",
        )
    ).get()
