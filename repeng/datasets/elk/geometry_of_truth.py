from typing import Literal, cast

import pandas as pd

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_to_all

Subset = Literal[
    "cities",
    "sp_en_trans",
    "larger_than",
    "cities_cities_conj",
    "cities_cities_disj",
]

_URL = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/91b2232/datasets"


def get_geometry_of_truth(subset: Subset) -> dict[str, BinaryRow]:
    if subset == "cities":
        return _get_paired(
            dataset_id="got_cities",
            csv_name_pos="cities",
            csv_name_neg="neg_cities",
            expected_identical_labels=["city", "country", "correct_country"],
        )
    elif subset == "sp_en_trans":
        return _get_paired(
            dataset_id="got_sp_en_trans",
            csv_name_pos="sp_en_trans",
            csv_name_neg="neg_sp_en_trans",
            expected_identical_labels=[],
        )
    elif subset == "larger_than":
        return _get_paired(
            dataset_id="got_larger_than",
            csv_name_pos="larger_than",
            csv_name_neg="smaller_than",
            expected_identical_labels=[],
        )
    elif subset == "cities_cities_conj":
        return _get_unpaired(
            dataset_id="got_cities_cities_conj",
            csv_name="cities_cities_conj",
        )
    elif subset == "cities_cities_disj":
        return _get_unpaired(
            dataset_id="got_cities_cities_disj",
            csv_name="cities_cities_disj",
        )


def _get_paired(
    dataset_id: DatasetId,
    *,
    csv_name_pos: str,
    csv_name_neg: str,
    expected_identical_labels: list[str],
) -> dict[str, BinaryRow]:
    result = {}
    csv_pos = _get_csv(csv_name_pos)
    csv_neg = _get_csv(csv_name_neg)
    assert len(csv_pos) == len(csv_neg)
    for index in deterministic_shuffle(list(range(len(csv_pos))), key=str):
        row_pos = csv_pos.iloc[index]
        row_neg = csv_neg.iloc[index]
        assert all(row_pos[x] == row_neg[x] for x in expected_identical_labels), (
            row_pos,
            row_neg,
        )
        assert row_pos["label"] != row_neg["label"], (row_pos, row_neg)
        for answer_type, row in [("pos", row_pos), ("neg", row_neg)]:
            result[f"{dataset_id}-{index}-{answer_type}"] = BinaryRow(
                dataset_id=dataset_id,
                group_id=str(index),
                split=split_to_all(dataset_id, str(index)),
                text=cast(str, row["statement"]),
                label=row["label"] == 1,
                format_args=dict(),
                answer_type=answer_type,
            )
    return result


def _get_unpaired(
    dataset_id: DatasetId,
    csv_name: str,
) -> dict[str, BinaryRow]:
    result = {}
    df = _get_csv(csv_name)
    for index, row in deterministic_shuffle(df.iterrows(), lambda row: str(row[0])):
        assert isinstance(index, int)
        result[f"{dataset_id}-{index}"] = BinaryRow(
            dataset_id=dataset_id,
            split=split_to_all(dataset_id, str(index)),
            text=cast(str, row["statement"]),
            label=row["label"] == 1,
            format_args=dict(),
        )
    return result


def _get_csv(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(f"{_URL}/{csv_name}.csv")
