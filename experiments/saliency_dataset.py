import torch

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.datasets.elk.utils.limits import Limits, SplitLimits

"""
4 models
 * 4 datasets
 * 20 layers
 * 1 token
 * 4K questions
 * 3 answers
 * 5120 hidden dim size
 * 2 bytes
= 39GB
"""

create_activations_dataset(
    tag="saliency_2024-02-26_v1",
    llm_ids=[
        "Llama-2-7b-hf",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-hf",
        "Llama-2-13b-chat-hf",
        "gemma-2b",
        "gemma-2b-it",
        "gemma-7b",
        "gemma-7b-it",
        "Mistral-7B",
        "Mistral-7B-Instruct",
    ],
    dataset_ids=[
        "boolq/simple",
        "imdb/simple",
        "race/simple",
        "got_cities",
    ],
    group_limits=Limits(
        default=SplitLimits(
            train=1000,
            train_hparams=0,
            validation=400,
        ),
        by_dataset={},
    ),
    num_tokens_from_end=1,
    device=torch.device("cuda"),
    layers_start=1,
    layers_end=None,
    layers_skip=2,
)
