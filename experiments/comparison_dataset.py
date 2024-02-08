import torch

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.types import DatasetId
from repeng.datasets.elk.utils.collections import resolve_dataset_ids
from repeng.datasets.elk.utils.limits import Limits, SplitLimits

validation_only_datasets: list[DatasetId] = [
    *resolve_dataset_ids("dlk-val"),
    *resolve_dataset_ids("repe-val"),
    *resolve_dataset_ids("got-val"),
    "truthful_qa",
]
create_activations_dataset(
    tag="datasets_2024-02-08_v1",
    llm_ids=["Llama-2-7b-chat-hf"],
    dataset_ids=resolve_dataset_ids("all"),
    group_limits=Limits(
        default=SplitLimits(
            train=400,
            train_hparams=100,
            validation=200,
        ),
        by_dataset={
            dataset_id: SplitLimits(
                train=0,
                train_hparams=100,
                validation=1000,
            )
            for dataset_id in validation_only_datasets
        },
    ),
    num_tokens_from_end=1,
    device=torch.device("cuda"),
    layers_start=None,
    layers_end=None,
)
