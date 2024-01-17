from typing import Callable

from repeng.datasets.arc import get_arc
from repeng.datasets.common_sense_qa import get_common_sense_qa
from repeng.datasets.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.open_book_qa import get_open_book_qa
from repeng.datasets.race import get_race
from repeng.datasets.true_false import get_true_false_dataset
from repeng.datasets.truthful_model_written import get_truthful_model_written
from repeng.datasets.truthful_qa import get_truthful_qa
from repeng.datasets.types import BinaryRow, DatasetId

ALL_DATASET_IDS: list[DatasetId] = [
    # From Geometry of Truth paper.
    # https://arxiv.org/abs/2310.06824
    "geometry_of_truth-cities",
    "geometry_of_truth-neg_cities",
    "geometry_of_truth-sp_en_trans",
    "geometry_of_truth-neg_sp_en_trans",
    "geometry_of_truth-larger_than",
    "geometry_of_truth-smaller_than",
    "geometry_of_truth-cities_cities_conj",
    "geometry_of_truth-cities_cities_disj",
    # From Representation Engineering paper, extracting "truth".
    # https://arxiv.org/abs/2310.01405
    "open_book_qa",
    "common_sense_qa",
    "race",
    "arc_challenge",
    "arc_easy",
    # From Representation Engineering paper, extracting "honesty".
    # https://arxiv.org/abs/2310.01405
    "true_false",
    # From Representation Engineering paper, for evaluations.
    # https://arxiv.org/abs/2310.01405
    "truthful_qa",
    # Custom dataset mirroring CCA paper.
    # https://arxiv.org/abs/2312.06681
    "truthful_model_written",
]

PAIRED_DATASET_IDS: list[DatasetId] = [
    "arc_easy",
    "arc_challenge",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
]

_DATASET_FNS: dict[DatasetId, Callable[[], dict[str, BinaryRow]]] = {
    "geometry_of_truth-cities": lambda: get_geometry_of_truth("cities"),
    "geometry_of_truth-neg_cities": lambda: get_geometry_of_truth("neg_cities"),
    "geometry_of_truth-sp_en_trans": lambda: get_geometry_of_truth("sp_en_trans"),
    "geometry_of_truth-neg_sp_en_trans": lambda: get_geometry_of_truth(
        "neg_sp_en_trans"
    ),
    "geometry_of_truth-larger_than": lambda: get_geometry_of_truth("larger_than"),
    "geometry_of_truth-smaller_than": lambda: get_geometry_of_truth("smaller_than"),
    "geometry_of_truth-cities_cities_conj": lambda: get_geometry_of_truth(
        "cities_cities_conj"
    ),
    "geometry_of_truth-cities_cities_disj": lambda: get_geometry_of_truth(
        "cities_cities_disj"
    ),
    "arc_challenge": lambda: get_arc("challenge"),
    "arc_easy": lambda: get_arc("easy"),
    "common_sense_qa": lambda: get_common_sense_qa(),
    "open_book_qa": lambda: get_open_book_qa(),
    "race": lambda: get_race(),
    "truthful_qa": lambda: get_truthful_qa(),
    "truthful_model_written": lambda: get_truthful_model_written(),
    "true_false": get_true_false_dataset,
}


def get_datasets(dataset_ids: list[DatasetId]) -> dict[str, BinaryRow]:
    return {
        k: v
        for dataset_id in dataset_ids
        for k, v in _DATASET_FNS[dataset_id]().items()
    }
