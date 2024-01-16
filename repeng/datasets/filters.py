from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from repeng.datasets.types import BinaryRow, DatasetId, Split


@dataclass(frozen=True)
class _DatasetAndSplit:
    dataset_id: DatasetId
    split: Split


def limit_dataset_and_split_fn(limit: int) -> Callable[[str, BinaryRow], bool]:
    counts: dict[_DatasetAndSplit, int] = defaultdict(int)

    def fn(_: str, row: BinaryRow) -> bool:
        dataset_and_split = _DatasetAndSplit(dataset_id=row.dataset_id, split=row.split)
        counts[dataset_and_split] += 1
        return counts[dataset_and_split] <= limit

    return fn
