import fire

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.models.types import LlmId


def main(tag: str, *, num_samples_per_dataset: int, llm_ids: list[LlmId]):
    create_activations_dataset(
        tag=tag,
        num_samples_per_dataset=num_samples_per_dataset,
        llm_ids=llm_ids,
    )


if __name__ == "__main__":
    fire.Fire(main)
