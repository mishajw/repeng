from typing import Literal

import pandas as pd

from repeng.datasets.types import BinaryRow

_DATASET_ID = "geometry_of_truth"
_URL = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/91b2232/datasets"


def get_geometry_of_truth(
    subset: Literal[
        "cities",
        "neg_cities",
        "sp_en_trans",
        "neg_sp_en_trans",
        "larger_than",
        "smaller_than",
        "cities_cities_conj",
        "cities_cities_disj",
    ]
) -> dict[str, BinaryRow]:
    dataset_id = f"{_DATASET_ID}-{subset}"
    result = {}
    df = pd.read_csv(f"{_URL}/{subset}.csv")
    for index, row in df.iterrows():
        result[f"{dataset_id}-{index}"] = BinaryRow(
            dataset_id=dataset_id,
            text=row["statement"],
            is_true=row["label"] == 1,
            format_args=dict(),
            format_style="misc",
        )
    return result
