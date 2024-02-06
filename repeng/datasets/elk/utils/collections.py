from typing import Callable, Literal, cast, get_args

from tqdm import tqdm

from repeng.datasets.elk.arc import get_arc
from repeng.datasets.elk.common_sense_qa import get_common_sense_qa
from repeng.datasets.elk.dlk import get_dlk_dataset
from repeng.datasets.elk.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.elk.open_book_qa import get_open_book_qa
from repeng.datasets.elk.race import get_race
from repeng.datasets.elk.true_false import get_true_false_dataset
from repeng.datasets.elk.truthful_model_written import get_truthful_model_written
from repeng.datasets.elk.truthful_qa import get_truthful_qa
from repeng.datasets.elk.types import BinaryRow, DatasetId

DatasetCollectionId = Literal[
    "all",
    "repe",
    "geometry_of_truth",
    "dlk",
    "small-varied",
]


_DATASET_COLLECTIONS: dict[DatasetCollectionId, list[DatasetId]] = {
    "all": cast(list[DatasetId], list(get_args(DatasetId))),
    "repe": [
        "open_book_qa",
        "common_sense_qa",
        "race",
        "arc_challenge",
        "arc_easy",
    ],
    "geometry_of_truth": [
        "geometry_of_truth/cities",
        "geometry_of_truth/sp_en_trans",
        "geometry_of_truth/larger_than",
        "geometry_of_truth/cities_cities_conj",
        "geometry_of_truth/cities_cities_disj",
    ],
    "dlk": [
        "imdb",
        "amazon_polarity",
        "ag_news",
        "dbpedia_14",
        "rte",
        "qnli",
        "copa",
        "boolq",
        "piqa",
    ],
    "small-varied": [
        "geometry_of_truth/cities",
        "arc_easy",
        "dbpedia_14",
    ],
}

_DATASET_FNS: dict[DatasetId, Callable[[], dict[str, BinaryRow]]] = {
    "geometry_of_truth/cities": lambda: get_geometry_of_truth("cities"),
    "geometry_of_truth/sp_en_trans": lambda: get_geometry_of_truth("sp_en_trans"),
    "geometry_of_truth/larger_than": lambda: get_geometry_of_truth("larger_than"),
    "geometry_of_truth/cities_cities_conj": lambda: get_geometry_of_truth(
        "cities_cities_conj"
    ),
    "geometry_of_truth/cities_cities_disj": lambda: get_geometry_of_truth(
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
    "imdb": lambda: get_dlk_dataset("imdb"),
    "amazon_polarity": lambda: get_dlk_dataset("amazon_polarity"),
    "ag_news": lambda: get_dlk_dataset("ag_news"),
    "dbpedia_14": lambda: get_dlk_dataset("dbpedia_14"),
    "rte": lambda: get_dlk_dataset("rte"),
    "copa": lambda: get_dlk_dataset("copa"),
    "boolq": lambda: get_dlk_dataset("boolq"),
    "piqa": lambda: get_dlk_dataset("piqa"),
}


def get_dataset_collection(
    dataset_collection_id: DatasetCollectionId,
) -> dict[str, BinaryRow]:
    return get_datasets(resolve_dataset_ids(dataset_collection_id))


def resolve_dataset_ids(
    id: DatasetId | DatasetCollectionId,
) -> list[DatasetId]:
    if id in get_args(DatasetId):
        return [cast(DatasetId, id)]
    elif id in get_args(DatasetCollectionId):
        return _DATASET_COLLECTIONS[cast(DatasetCollectionId, id)]
    else:
        raise ValueError(f"Unknown ID: {id}")


def get_datasets(dataset_ids: list[DatasetId]) -> dict[str, BinaryRow]:
    result = {}
    pbar = tqdm(dataset_ids, desc="loading datasets")
    for dataset_id in pbar:
        pbar.set_postfix(dataset=dataset_id)
        result.update(_DATASET_FNS[dataset_id]())
    return result
