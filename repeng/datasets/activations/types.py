from pydantic import BaseModel

from repeng.datasets.elk.types import DatasetId, Split
from repeng.models.llms import LlmId
from repeng.utils.pydantic_ndarray import NdArray


class ActivationResultRow(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    pair_id: str | None
    activations: dict[str, NdArray]  # (n, d)
    label: bool
    split: Split
    llm_id: LlmId

    class Config:
        arbitrary_types_allowed = True
