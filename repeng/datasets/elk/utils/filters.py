from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from repeng.datasets.elk.types import BinaryRow, DatasetId, Split


@dataclass(frozen=True)
class _DatasetAndSplit:
    dataset_id: DatasetId
    split: Split


def limit_dataset_and_split_fn(
    *,
    train_limit: int,
    validation_limit: int,
) -> Callable[[str, BinaryRow], bool]:
    counts: dict[_DatasetAndSplit, int] = defaultdict(int)

    def fn(_: str, row: BinaryRow) -> bool:
        dataset_and_split = _DatasetAndSplit(dataset_id=row.dataset_id, split=row.split)
        counts[dataset_and_split] += 1
        limit = train_limit if row.split == "train" else validation_limit
        return counts[dataset_and_split] <= limit

    return fn
