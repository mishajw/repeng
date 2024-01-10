from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

import torch
from pyparsing import Any

ModelT = TypeVar("ModelT", bound=torch.nn.Module)


class TensorExtractor(ABC):
    @abstractmethod
    def extract(self, output: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def insert(self, output: Any, tensor: torch.Tensor) -> Any:
        ...


class IdentityTensorExtractor(TensorExtractor):
    def extract(self, output: Any) -> torch.Tensor:
        assert isinstance(
            output, torch.Tensor
        ), f"Expected tensor, instead found: {type(output)}"
        return output

    def insert(self, output: Any, tensor: torch.Tensor) -> Any:
        return tensor


@dataclass
class TupleTensorExtractor(TensorExtractor):
    index: int

    def extract(self, output: Any) -> torch.Tensor:
        assert isinstance(
            output, tuple
        ), f"Expected tuple, instead found: {type(output)}"
        assert isinstance(
            output[self.index], torch.Tensor
        ), f"Expected tensor, instead found: {type(output[self.index])}"
        return output[self.index]

    def insert(self, output: Any, tensor: torch.Tensor) -> Any:
        assert isinstance(
            output, tuple
        ), f"Expected tuple, instead found: {type(output)}"
        return (*output[: self.index], tensor, *output[self.index + 1 :])


@dataclass
class Point(Generic[ModelT]):
    name: str
    module_fn: Callable[[ModelT], torch.nn.Module]
    tensor_extractor: "TensorExtractor" = field(default_factory=IdentityTensorExtractor)
