from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from typing_extensions import override


class BaseProbe(ABC):
    @abstractmethod
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> Float[np.ndarray, "n"]:  # noqa: F821
        """
        Predicts the probability of the label being true for each row.
        """
        ...


@dataclass
class DotProductProbe(BaseProbe):
    probe: Float[np.ndarray, "d"]  # noqa: F821

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> Float[np.ndarray, "n"]:  # noqa: F821
        return _sigmoid(activations @ self.probe)


def _sigmoid(
    x: Float[np.ndarray, "n"],  # noqa: F821
) -> Float[np.ndarray, "n"]:  # noqa: F821
    return 1 / (1 + np.exp(-x))
