import fire
import torch

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.types import DatasetId
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from repeng.models.collections import LlmCollectionId, resolve_llm_ids
from repeng.models.types import LlmId


def main(
    tag: str,
    *,
    llms: LlmCollectionId | LlmId,
    datasets: DatasetCollectionId | DatasetId,
    device: str,
    num_samples_per_dataset: int,
    num_validation_samples_per_dataset: int,
    num_tokens_from_end: int | None = None,
    layers_start: int | None = None,
    layers_end: int | None = None,
):
    # 25GB for model weights (+25GB for pythia-12b)
    # 50GB for activations dataset
    # 35GB for pre-upload dataset
    create_activations_dataset(
        tag=tag,
        llm_ids=resolve_llm_ids(llms),
        dataset_ids=resolve_dataset_ids(datasets),
        num_samples_per_dataset=num_samples_per_dataset,
        num_validation_samples_per_dataset=num_validation_samples_per_dataset,
        device=torch.device(device),
        num_tokens_from_end=num_tokens_from_end,
        layers_start=layers_start,
        layers_end=layers_end,
    )


if __name__ == "__main__":
    fire.Fire(main)
