from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression

from repeng.activations.probe_preparations import (
    LabeledActivationArray,
    LabeledPairedActivationArray,
)
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
    model = LogisticRegression(max_iter=1000, fit_intercept=True)
    model.fit(activations.activations, activations.labels)
    return LogisticRegressionProbe(model)


def train_paired_lr_probe(
    activations: LabeledPairedActivationArray,
) -> LogisticRegressionProbe:
    df = pd.DataFrame(
        {
            "activations": list(activations.activations),
            "pairs": activations.pairs,
            "labels": activations.labels,
        }
    )
    pair_means = (
        df.groupby(["pairs"])["activations"]
        .apply(lambda a: np.mean(a, axis=0))
        .rename("pair_mean")
    )
    df = df.join(pair_means, on="pairs")
    df["activations"] = df["activations"] - df["pair_mean"]
    activations_pair_centered = LabeledActivationArray(
        activations=np.stack(df["activations"].to_list()),
        labels=df["labels"].to_numpy(),
    )
    return train_lr_probe(activations_pair_centered)
