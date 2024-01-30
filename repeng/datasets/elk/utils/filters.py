from typing import Literal, Protocol, cast, get_args

from typing_extensions import runtime_checkable

from repeng.datasets.elk.types import DatasetId
from repeng.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)


@runtime_checkable
class _DatasetFilterFn(Protocol):
    def __call__(self, *, dataset_id: DatasetId, template_name: str | None) -> bool:
        ...


_DatasetFilterFnId = Literal[
    "geometry_of_truth/cities/pos",
    "geometry_of_truth/cities/neg",
    "geometry_of_truth/sp_en_trans/pos",
    "geometry_of_truth/sp_en_trans/neg",
    "geometry_of_truth/larger_than/large",
    "geometry_of_truth/larger_than/small",
]
DatasetFilterId = _DatasetFilterFnId | DatasetId | DatasetCollectionId

_DATASET_FILTER_FNS: dict[_DatasetFilterFnId, _DatasetFilterFn] = {
    "geometry_of_truth/cities/pos": lambda dataset_id, template_name: (
        dataset_id == "geometry_of_truth/cities" and template_name == "pos"
    ),
    "geometry_of_truth/cities/neg": lambda dataset_id, template_name: (
        dataset_id == "geometry_of_truth/cities" and template_name == "neg"
    ),
    "geometry_of_truth/sp_en_trans/pos": lambda dataset_id, template_name: (
        dataset_id == "geometry_of_truth/sp_en_trans" and template_name == "pos"
    ),
    "geometry_of_truth/sp_en_trans/neg": lambda dataset_id, template_name: (
        dataset_id == "geometry_of_truth/sp_en_trans" and template_name == "neg"
    ),
    "geometry_of_truth/larger_than/large": lambda dataset_id, template_name: (
        dataset_id == "geometry_of_truth/larger_than" and template_name == "pos"
    ),
    "geometry_of_truth/larger_than/small": lambda dataset_id, template_name: (
        dataset_id == "geometry_of_truth/larger_than" and template_name == "neg"
    ),
}


def filter_dataset(
    filter: DatasetFilterId,
    *,
    dataset_id: DatasetId,
    template_name: str | None,
) -> bool:
    if filter in get_args(_DatasetFilterFnId):
        return _DATASET_FILTER_FNS[cast(_DatasetFilterFnId, filter)](
            dataset_id=dataset_id, template_name=template_name
        )
    else:
        return dataset_id in resolve_dataset_ids(
            cast(DatasetId | DatasetCollectionId, filter)
        )
