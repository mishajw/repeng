from typing import Literal

from pydantic import BaseModel

FormatStyle = Literal["lat", "misc"]


class BinaryRow(BaseModel, extra="forbid"):
    dataset_id: str
    text: str
    is_true: bool
    format_args: dict[str, str]
    format_style: FormatStyle


class PairedBinaryRow(BinaryRow, extra="forbid"):
    pair_id: str


# deprecated
class InputRow(BaseModel, extra="forbid"):
    pair_idx: str
    text: str
    is_text_true: bool
    does_text_contain_true: bool
