from pydantic import BaseModel

from repeng.datasets.elk.types import DatasetId, Split
from repeng.models.llms import LlmId
from repeng.utils.pydantic_ndarray import NdArray


class ActivationResultRow(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    group_id: str | None
    template_name: str | None
    answer_type: str | None
    activations: dict[str, NdArray]  # (s, d)
    prompt_logprobs: float
    label: bool
    split: Split
    llm_id: LlmId

    class Config:
        arbitrary_types_allowed = True
