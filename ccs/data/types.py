from pydantic import BaseModel


class InputRow(BaseModel, extra="forbid"):
    pair_idx: str
    text: str
    is_text_true: bool
    does_text_contain_true: bool
