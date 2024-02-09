from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from overrides import override

from repeng.datasets.elk.types import DatasetId


class DatasetFilter(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """
        Gets the name of the filter, for use in reporting and plotting.
        """
        ...

    @abstractmethod
    def filter(self, dataset_id: DatasetId, answer_type: str | None) -> bool:
        """
        Filters a row of a dataset.
        """
        ...


@dataclass
class DatasetIdFilter(DatasetFilter):
    dataset_id: DatasetId

    @override
    def get_name(self) -> str:
        return self.dataset_id

    @override
    def filter(self, dataset_id: DatasetId, answer_type: str | None) -> bool:
        return dataset_id == self.dataset_id


@dataclass
class ExactMatchFilter(DatasetFilter):
    name: str
    dataset_id: DatasetId
    answer_type: str | None

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def filter(self, dataset_id: DatasetId, answer_type: str | None) -> bool:
        return dataset_id == self.dataset_id and answer_type == self.answer_type


@dataclass
class DatasetCollectionFilter(DatasetFilter):
    name: str
    datasets: list[DatasetId]

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def filter(self, dataset_id: DatasetId, answer_type: str | None) -> bool:
        return dataset_id in self.datasets


DatasetFilterId = Literal[
    "got_cities/pos",
    "got_cities/neg",
    "got_sp_en_trans/pos",
    "got_sp_en_trans/neg",
    "got_larger_than/large",
    "got_larger_than/small",
]

DATASET_FILTER_FNS: dict[DatasetFilterId, DatasetFilter] = {
    "got_cities/pos": ExactMatchFilter(
        "got_cities/pos",
        dataset_id="got_cities",
        answer_type="pos",
    ),
    "got_cities/neg": ExactMatchFilter(
        "got_cities/neg",
        dataset_id="got_cities",
        answer_type="neg",
    ),
    "got_sp_en_trans/pos": ExactMatchFilter(
        "got_sp_en_trans/pos",
        dataset_id="got_sp_en_trans",
        answer_type="pos",
    ),
    "got_sp_en_trans/neg": ExactMatchFilter(
        "got_sp_en_trans/neg",
        dataset_id="got_sp_en_trans",
        answer_type="neg",
    ),
    "got_larger_than/large": ExactMatchFilter(
        "got_larger_than/large",
        dataset_id="got_larger_than",
        answer_type="pos",
    ),
    "got_larger_than/small": ExactMatchFilter(
        "got_larger_than/small",
        dataset_id="got_larger_than",
        answer_type="neg",
    ),
}
