# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float


class BaseProbe(ABC):
    @abstractmethod
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> Bool[np.ndarray, "n"]:  # noqa: F821
        ...


@dataclass
class DotProductProbe(BaseProbe):
    probe: Float[np.ndarray, "d"]  # noqa: F821

    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> Bool[np.ndarray, "n"]:  # noqa: F821
        return activations @ self.probe > 0
