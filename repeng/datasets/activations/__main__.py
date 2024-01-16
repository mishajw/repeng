import fire

from repeng.datasets.activations.creation import create_activations_dataset
from repeng.models.types import LlmId


def main(llm_ids: list[LlmId]):
    create_activations_dataset(llm_ids)


if __name__ == "__main__":
    fire.Fire(main)
