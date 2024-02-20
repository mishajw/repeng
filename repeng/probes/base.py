from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int64
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


class BaseGroupedProbe(BaseProbe, ABC):
    @abstractmethod
    def predict_grouped(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
        pairs: Int64[np.ndarray, "n"],  # noqa: F821
    ) -> "PredictResult":
        """
        Predicts the probability of the label being true for each row.

        Activations are grouped into pairs, and the pair information is used for
        predictions.
        """
        ...


@dataclass
class PredictResult:
    logits: Float[np.ndarray, "n"]  # noqa: F821


@dataclass
class DotProductProbe(BaseProbe):
    probe: Float[np.ndarray, "d"]  # noqa: F821

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        logits = activations @ self.probe
        return PredictResult(logits=logits)
