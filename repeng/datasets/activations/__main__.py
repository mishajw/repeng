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
    num_validation_samples_per_dataset: int | None = None,
):
    if num_validation_samples_per_dataset is None:
        num_validation_samples_per_dataset = num_samples_per_dataset // 5
    # 25GB of disk space needed to store these model weights
    # 10GB of disk space needed to store activations dataset
    create_activations_dataset(
        tag=tag,
        llm_ids=resolve_llm_ids(llms),
        dataset_ids=resolve_dataset_ids(datasets),
        num_samples_per_dataset=num_samples_per_dataset,
        num_validation_samples_per_dataset=num_validation_samples_per_dataset,
        device=torch.device(device),
    )


if __name__ == "__main__":
    fire.Fire(main)
