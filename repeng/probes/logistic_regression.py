from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression

from repeng.activations.probe_preparations import LabeledActivationArray
from repeng.probes.base import BaseProbe, PredictResult


@dataclass
class LogisticRegressionProbe(BaseProbe):
    model: LogisticRegression

    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        logits = self.model.decision_function(activations)
        return PredictResult(
            logits=logits,
            labels=logits > 0,
        )


def train_lr_probe(
    activations: LabeledActivationArray,
) -> LogisticRegressionProbe:
    model = LogisticRegression(max_iter=1000)
    model.fit(activations.activations, activations.labels)
    return LogisticRegressionProbe(model)
