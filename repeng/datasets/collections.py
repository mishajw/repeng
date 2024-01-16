from typing import Literal

from repeng.datasets.arc import get_arc
from repeng.datasets.common_sense_qa import get_common_sense_qa
from repeng.datasets.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.open_book_qa import get_open_book_qa
from repeng.datasets.race import get_race
from repeng.datasets.true_false import get_true_false_dataset
from repeng.datasets.truthful_model_written import get_truthful_model_written
from repeng.datasets.truthful_qa import get_truthful_qa
from repeng.datasets.types import BinaryRow

DATASET_NAMES: Literal[
    "arc_challenge",
    "arc_easy",
    "cities",
    "cities_cities_conj",
    "cities_cities_disj",
    "common_sense_qa",
    "larger_than",
]


PAIRED_DATASET_IDS = [
    "arc_easy",
    "arc_challenge",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
]


def get_all_datasets(
    limit_per_dataset: int | None = None,
) -> dict[str, BinaryRow | BinaryRow]:
    if limit_per_dataset is None:
        limit_per_dataset = 1000

    dataset_fns = [
        get_true_false_dataset,
        lambda: get_geometry_of_truth("cities"),
        lambda: get_geometry_of_truth("neg_cities"),
        lambda: get_geometry_of_truth("sp_en_trans"),
        lambda: get_geometry_of_truth("neg_sp_en_trans"),
        lambda: get_geometry_of_truth("larger_than"),
        lambda: get_geometry_of_truth("smaller_than"),
        lambda: get_geometry_of_truth("cities_cities_conj"),
        lambda: get_geometry_of_truth("cities_cities_disj"),
        lambda: get_arc("challenge"),
        lambda: get_arc("easy"),
        lambda: get_common_sense_qa(),
        lambda: get_open_book_qa(),
        lambda: get_race(),
        lambda: get_truthful_qa(),
        lambda: get_truthful_model_written(),
    ]

    result = {}
    for dataset_fn in dataset_fns:
        dataset = dataset_fn()
        dataset_limited = {k: v for k, v in list(dataset.items())[:limit_per_dataset]}
        result.update(dataset_limited)
    return result
