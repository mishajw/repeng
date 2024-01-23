from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float
from typing_extensions import override


class BaseProbe(ABC):
    @abstractmethod
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> "PredictResult":
        """
        Predicts the probability of the label being true for each row.
        """
        ...


@dataclass
class PredictResult:
    logits: Float[np.ndarray, "n"]  # noqa: F821
    labels: Bool[np.ndarray, "n"]  # noqa: F821


@dataclass
class DotProductProbe(BaseProbe):
    probe: Float[np.ndarray, "d"]  # noqa: F821

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        logits = activations @ self.probe
        return PredictResult(
            logits=logits,
            labels=logits > 0,
        )
