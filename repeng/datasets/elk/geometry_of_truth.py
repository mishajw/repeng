from typing import Literal, cast

import pandas as pd

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.split_partitions import get_split

Subset = Literal[
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
    "cities_cities_conj",
    "cities_cities_disj",
]

_URL = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/91b2232/datasets"
_SUBSET_TO_DATASET_ID: dict[Subset, DatasetId] = {
    "cities": "geometry_of_truth-cities",
    "neg_cities": "geometry_of_truth-neg_cities",
    "sp_en_trans": "geometry_of_truth-sp_en_trans",
    "neg_sp_en_trans": "geometry_of_truth-neg_sp_en_trans",
    "larger_than": "geometry_of_truth-larger_than",
    "smaller_than": "geometry_of_truth-smaller_than",
    "cities_cities_conj": "geometry_of_truth-cities_cities_conj",
    "cities_cities_disj": "geometry_of_truth-cities_cities_disj",
}


def get_geometry_of_truth(subset: Subset) -> dict[str, BinaryRow]:
    dataset_id = _SUBSET_TO_DATASET_ID[subset]
    result = {}
    df = pd.read_csv(f"{_URL}/{subset}.csv")
    for index, row in df.iterrows():
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
