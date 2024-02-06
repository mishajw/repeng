from dataclasses import dataclass
from typing import Literal, cast, get_args

from overrides import override

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.filters import DatasetCollectionFilter, DatasetFilter
from repeng.datasets.elk.utils.fns import get_datasets

DatasetCollectionId = Literal[
    "all",
    "dlk",
    "dlk-val",
    "repe",
    "repe-val",
    "got",
    "got-val",
    "multis",
]

_DATASET_COLLECTIONS: dict[DatasetCollectionId, DatasetCollectionFilter] = {
    "all": DatasetCollectionFilter(
        "all", cast(list[DatasetId], list(get_args(DatasetId)))
    ),
    "multis": DatasetCollectionFilter(
        "dlk-repe-got",
        [
            "imdb",
            "amazon_polarity",
            "ag_news",
            "dbpedia_14",
            "rte",
            "copa",
            "piqa",
            "open_book_qa",
            "race",
            "arc_challenge",
            "arc_easy",
            "geometry_of_truth/cities",
            "geometry_of_truth/sp_en_trans",
            "geometry_of_truth/cities_cities_conj",
            "geometry_of_truth/cities_cities_disj",
        ],
    ),
    "dlk": DatasetCollectionFilter(
        "dlk",
        [
            "imdb",
            "amazon_polarity",
            "ag_news",
            "dbpedia_14",
            "rte",
            "copa",
            "piqa",
        ],
    ),
    "dlk-val": DatasetCollectionFilter("dlk-val", ["boolq"]),
    "repe": DatasetCollectionFilter(
        "repe",
        [
            "open_book_qa",
            "race",
            "arc_challenge",
            "arc_easy",
        ],
    ),
    "repe-val": DatasetCollectionFilter("repe-val", ["common_sense_qa"]),
    "got": DatasetCollectionFilter(
        "got",
        [
            "geometry_of_truth/cities",
            "geometry_of_truth/sp_en_trans",
            "geometry_of_truth/cities_cities_conj",
            "geometry_of_truth/cities_cities_disj",
        ],
    ),
    "got-val": DatasetCollectionFilter("got-val", ["geometry_of_truth/larger_than"]),
}


@dataclass
class DatasetCollectionIdFilter(DatasetFilter):
    collection: DatasetCollectionId

    @override
    def get_name(self) -> str:
        return self.collection

    @override
    def filter(self, dataset_id: DatasetId, answer_type: str | None) -> bool:
        return dataset_id in resolve_dataset_ids(self.collection)


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
        return _DATASET_COLLECTIONS[cast(DatasetCollectionId, id)].datasets
    else:
        raise ValueError(f"Unknown ID: {id}")
