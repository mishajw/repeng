from dataclasses import dataclass
from typing import Literal, cast, get_args

from overrides import override

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.filters import DatasetCollectionFilter, DatasetFilter
from repeng.datasets.elk.utils.fns import get_datasets

DatasetCollectionId = Literal[
    "all",
    "dlk",
    "repe",
    "got",
    "repe-qa",
]

_DATASET_COLLECTIONS: dict[DatasetCollectionId, DatasetCollectionFilter] = {
    "all": DatasetCollectionFilter(
        "all", cast(list[DatasetId], list(get_args(DatasetId)))
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
            "boolq",
        ],
    ),
    "repe": DatasetCollectionFilter(
        "repe",
        [
            "open_book_qa",
            "race",
            "arc_challenge",
            "arc_easy",
            "common_sense_qa",
        ],
    ),
    "repe-qa": DatasetCollectionFilter(
        "repe-qa",
        [
            "open_book_qa/qa",
            "race/qa",
            "arc_challenge/qa",
            "arc_easy/qa",
            "common_sense_qa/qa",
        ],
    ),
    "got": DatasetCollectionFilter(
        "got",
        [
            "got_cities",
            "got_sp_en_trans",
            "got_cities_cities_conj",
            "got_cities_cities_disj",
            "got_larger_than",
        ],
    ),
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
