from typing import Literal, get_args

from pydantic import BaseModel

Split = Literal["train", "validation"]

DatasetId = Literal[
    "arc_challenge",
    "arc_easy",
    "geometry_of_truth/cities",
    "geometry_of_truth/neg_cities",
    "geometry_of_truth/sp_en_trans",
    "geometry_of_truth/neg_sp_en_trans",
    "geometry_of_truth/larger_than",
    "geometry_of_truth/smaller_than",
    "geometry_of_truth/cities_cities_conj",
    "geometry_of_truth/cities_cities_disj",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
    "truthful_model_written",
    "true_false",
]

GroupedDatasetId = Literal[
    "arc_challenge",
    "arc_easy",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
    "truthful_model_written",
    "true_false",
]


class BinaryRow(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    split: Split
    text: str
    is_true: bool
    format_args: dict[str, str]
    format_style: Literal["lat", "misc"]
    pair_id: str | None = None


# deprecated
class InputRow(BaseModel, extra="forbid"):
    pair_idx: str
    text: str
    is_text_true: bool
    does_text_contain_true: bool


def is_dataset_grouped(dataset_id: DatasetId) -> bool:
    return dataset_id in get_args(GroupedDatasetId)
