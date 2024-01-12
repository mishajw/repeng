from typing import Literal

from pydantic import BaseModel

FormatStyle = Literal["lat", "misc"]


class PairedBinaryRow(BaseModel, extra="forbid"):
    dataset_id: str
    pair_id: str
    text: str
    is_true: bool
    format_args: dict[str, str]
    format_style: FormatStyle


class BinaryRow(BaseModel, extra="forbid"):
    dataset_id: str
    text: str
    is_true: bool
    format_args: dict[str, str]
    format_style: FormatStyle


class InputRow(BaseModel, extra="forbid"):
    pair_idx: str
    text: str
    is_text_true: bool
    does_text_contain_true: bool
