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
    "geometry_of_truth/cities/pos",
    "geometry_of_truth/cities/neg",
    "geometry_of_truth/sp_en_trans/pos",
    "geometry_of_truth/sp_en_trans/neg",
    "geometry_of_truth/larger_than/large",
    "geometry_of_truth/larger_than/small",
]

_DATASET_FILTER_FNS: dict[DatasetFilterId, DatasetFilter] = {
    "geometry_of_truth/cities/pos": ExactMatchFilter(
        "geometry_of_truth/cities/pos",
        dataset_id="geometry_of_truth/cities",
        answer_type="pos",
    ),
    "geometry_of_truth/cities/neg": ExactMatchFilter(
        "geometry_of_truth/cities/neg",
        dataset_id="geometry_of_truth/cities",
        answer_type="neg",
    ),
    "geometry_of_truth/sp_en_trans/pos": ExactMatchFilter(
        "geometry_of_truth/sp_en_trans/pos",
        dataset_id="geometry_of_truth/sp_en_trans",
        answer_type="pos",
    ),
    "geometry_of_truth/sp_en_trans/neg": ExactMatchFilter(
        "geometry_of_truth/sp_en_trans/neg",
        dataset_id="geometry_of_truth/sp_en_trans",
        answer_type="neg",
    ),
    "geometry_of_truth/larger_than/large": ExactMatchFilter(
        "geometry_of_truth/larger_than/large",
        dataset_id="geometry_of_truth/larger_than",
        answer_type="pos",
    ),
    "geometry_of_truth/larger_than/small": ExactMatchFilter(
        "geometry_of_truth/larger_than/small",
        dataset_id="geometry_of_truth/larger_than",
        answer_type="neg",
    ),
}
