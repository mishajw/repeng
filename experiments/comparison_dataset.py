import torch

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.utils.collections import resolve_dataset_ids

create_activations_dataset(
    tag="datasets_2024-02-07_v1",
    llm_ids=["Llama-2-7b-hf", "Llama-2-7b-chat-hf"],
    dataset_ids=resolve_dataset_ids("all"),
    num_samples_per_dataset=800,
    num_hparams_samples_per_dataset=200,
    num_validation_samples_per_dataset=200,
    num_tokens_from_end=1,
    device=torch.device("cuda"),
    layers_start=None,
    layers_end=None,
)
