from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from repeng.datasets.elk.types import BinaryRow, DatasetId, Split


@dataclass(frozen=True)
class _DatasetAndSplit:
    dataset_id: DatasetId
    split: Split
    template_name: str | None = None


def limit_dataset_and_split_fn(
    *,
    train_limit: int,
    train_hparams_limit: int,
    validation_limit: int,
    limit_templates_separately: bool = True,
) -> Callable[[str, BinaryRow], bool]:
    counts: dict[_DatasetAndSplit, int] = defaultdict(int)

    def limit(split: Split) -> int:
        if split == "train":
            return train_limit
        elif split == "train-hparams":
            return train_hparams_limit
        elif split == "validation":
            return validation_limit
        else:
            raise ValueError()

    def fn(_: str, row: BinaryRow) -> bool:
        dataset_and_split = _DatasetAndSplit(
            dataset_id=row.dataset_id,
            split=row.split,
            template_name=row.template_name if limit_templates_separately else None,
        )
        counts[dataset_and_split] += 1
        return counts[dataset_and_split] <= limit(row.split)

    return fn
