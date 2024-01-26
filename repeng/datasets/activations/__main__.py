import fire
import torch

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.models.types import LlmId


def main(tag: str, *, num_samples_per_dataset: int, device: str):
    # 25GB of disk space needed to store these model weights
    # 10GB of disk space needed to store activations dataset
    llm_ids: list[LlmId] = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
        "pythia-6.9b",
        # "pythia-12b",
    ]
    create_activations_dataset(
        tag=tag,
        num_samples_per_dataset=num_samples_per_dataset,
        llm_ids=llm_ids,
        device=torch.device(device),
    )


if __name__ == "__main__":
    fire.Fire(main)
