from typing import Literal

from pydantic import BaseModel

Split = Literal["train", "validation"]


class BinaryRow(BaseModel, extra="forbid"):
    dataset_id: str
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
