from typing import Literal, cast

import pandas as pd

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import get_split

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
            "geometry_of_truth/cities",
            "cities",
            "neg_cities",
            ["city", "country", "correct_country"],
        )
    elif subset == "sp_en_trans":
        return _get_paired(
            "geometry_of_truth/sp_en_trans",
            "sp_en_trans",
            "neg_sp_en_trans",
            [],
        )
    elif subset == "larger_than":
        return _get_paired(
            "geometry_of_truth/larger_than",
            "larger_than",
            "smaller_than",
            [],
        )
    elif subset == "cities_cities_conj":
        return _get_unpaired(
            "geometry_of_truth/cities_cities_conj",
            "cities_cities_conj",
        )
    elif subset == "cities_cities_disj":
        return _get_unpaired(
            "geometry_of_truth/cities_cities_disj",
            "cities_cities_disj",
        )


def _get_paired(
    dataset_id: DatasetId,
    csv_name1: str,
    csv_name2: str,
    expected_identical_labels: list[str],
) -> dict[str, BinaryRow]:
    result = {}
    dataset_id = "geometry_of_truth/cities"
    csv1 = _get_csv(csv_name1)
    csv2 = _get_csv(csv_name2)
    assert len(csv1) == len(csv2)
    for index in deterministic_shuffle(list(range(len(csv1))), key=str):
        row1 = csv1.iloc[index]
        row2 = csv2.iloc[index]
        assert all(row1[x] == row2[x] for x in expected_identical_labels), (row1, row2)
        assert row1["label"] != row2["label"], (row1, row2)
        for label, row in [("pos", row1), ("neg", row2)]:
            result[f"{dataset_id}-{index}-{label}"] = BinaryRow(
                dataset_id=dataset_id,
                pair_id=str(index),
                split=get_split(dataset_id, str(index)),
                text=cast(str, row["statement"]),
                is_true=row["label"] == 1,
                format_args=dict(),
                format_style="misc",
                # TODO: Is this an abuse of this field?
                template_name=label,
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
            split=get_split(dataset_id, str(index)),
            text=cast(str, row["statement"]),
            is_true=row["label"] == 1,
            format_args=dict(),
            format_style="misc",
        )
    return result


def _get_csv(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(f"{_URL}/{csv_name}.csv")
