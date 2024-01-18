from typing import Callable, Literal, cast, get_args

from tqdm import tqdm

from repeng.datasets.elk.arc import get_arc
from repeng.datasets.elk.common_sense_qa import get_common_sense_qa
from repeng.datasets.elk.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.elk.open_book_qa import get_open_book_qa
from repeng.datasets.elk.race import get_race
from repeng.datasets.elk.true_false import get_true_false_dataset
from repeng.datasets.elk.truthful_model_written import get_truthful_model_written
from repeng.datasets.elk.truthful_qa import get_truthful_qa
from repeng.datasets.elk.types import BinaryRow, DatasetId

DatasetCollectionId = Literal[
    "all",
    "representation-engineering",
    "geometry-of-truth",
    "geometry-of-truth-cities",
    "geometry-of-truth-cities-with-neg",
    "persona",
    "misc",
]


_DATASET_COLLECTIONS: dict[DatasetCollectionId, list[DatasetId]] = {
    "all": cast(list[DatasetId], list(get_args(DatasetId))),
    "representation-engineering": [
        "open_book_qa",
        "common_sense_qa",
        "race",
        "arc_challenge",
        "arc_easy",
    ],
    "geometry-of-truth": [
        "geometry_of_truth-cities",
        "geometry_of_truth-neg_cities",
        "geometry_of_truth-sp_en_trans",
        "geometry_of_truth-neg_sp_en_trans",
        "geometry_of_truth-larger_than",
        "geometry_of_truth-smaller_than",
        "geometry_of_truth-cities_cities_conj",
        "geometry_of_truth-cities_cities_disj",
    ],
    "geometry-of-truth-cities": [
        "geometry_of_truth-cities",
    ],
    "geometry-of-truth-cities-with-neg": [
        "geometry_of_truth-cities",
        "geometry_of_truth-neg_cities",
    ],
    "persona": [
        "truthful_model_written",
    ],
    "misc": [
        "true_false",
        "truthful_qa",
    ],
}

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


def get_dataset_collection(
    dataset_collection_id: DatasetCollectionId,
) -> dict[str, BinaryRow]:
    return get_datasets(get_dataset_ids_for_collection(dataset_collection_id))


def get_dataset_ids_for_collection(
    dataset_collection_id: DatasetCollectionId,
) -> list[DatasetId]:
    return _DATASET_COLLECTIONS[dataset_collection_id]


def get_datasets(dataset_ids: list[DatasetId]) -> dict[str, BinaryRow]:
    return {
        k: v
        for dataset_id in tqdm(dataset_ids, desc="loading datasets")
        for k, v in _DATASET_FNS[dataset_id]().items()
    }
