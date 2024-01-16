# %%
import numpy as np
from pydantic import BeforeValidator, PlainSerializer
from typing_extensions import Annotated


def _serialize(array: np.ndarray) -> dict:
    return dict(
        dtype=str(array.dtype),
        array=array.tolist(),
    )


def _deserialize(obj: dict | np.ndarray) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    assert obj.keys() == {"dtype", "array"}
    return np.array(
        obj["array"],
        dtype=obj["dtype"],
    )


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(_deserialize),
    PlainSerializer(_serialize),
]


# class Test(BaseModel):
#     array: NdArray

#     class Config:
#         arbitrary_types_allowed = True


# json = Test(array=np.array([1, 2, 3])).model_dump_json()
# Test.model_validate_json(json)
