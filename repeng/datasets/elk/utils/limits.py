from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from repeng.datasets.elk.types import BinaryRow, DatasetId, Split


@dataclass
class Limits:
    default: "SplitLimits"
    by_dataset: dict[DatasetId, "SplitLimits"]


@dataclass
class SplitLimits:
    train: int | None
    train_hparams: int | None
    validation: int | None


@dataclass(frozen=True)
class _DatasetAndSplit:
    dataset_id: DatasetId
    split: Split
    template_name: str | None = None


@dataclass
class _GroupCount:
    groups: set[str]
    num_nones: int

    def add(self, group_id: str | None) -> None:
        if group_id is None:
            self.num_nones += 1
        else:
            self.groups.add(group_id)

    def count(self) -> int:
        return len(self.groups) + self.num_nones


def limit_groups(limits: Limits) -> Callable[[str, BinaryRow], bool]:
    group_counts: dict[_DatasetAndSplit, _GroupCount] = defaultdict(
        lambda: _GroupCount(set(), 0)
    )

    def fn(_, row: BinaryRow) -> bool:
        dataset_and_split = _DatasetAndSplit(
            dataset_id=row.dataset_id,
            split=row.split,
        )

        if row.dataset_id not in limits.by_dataset:
            split_limits = limits.default
        else:
            split_limits = limits.by_dataset[row.dataset_id]

        if row.split == "train":
            limit = split_limits.train
        elif row.split == "train-hparams":
            limit = split_limits.train_hparams
        elif row.split == "validation":
            limit = split_limits.validation
        else:
            raise ValueError()

        if limit is None:
            return True
        group_counts[dataset_and_split].add(row.group_id)
        return group_counts[dataset_and_split].count() <= limit

    return fn
