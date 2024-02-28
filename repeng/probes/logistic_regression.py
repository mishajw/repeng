"""
Logistic regression based probes.

Methodology for ungrouped logistic regression:
1. Given a set of activations, and whether each activation is from a true or false
statement.
2. Fit a linear probe using scikit's LogisticRegression implementation. The probe takes
in the activations and predicts the label.

Methodology for grouped logistic regression:
1. Given a set of activations, and whether each activation is from a true or false
statement.
2. Subtract the average activation of each group from each group member.
2. Fit a linear probe using scikit's LogisticRegression implementation. The probe takes
in the group-normalized activations and predicts the label.

Regularization: C=1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int64
from sklearn.linear_model import LogisticRegression
from typing_extensions import override

from repeng.probes.base import BaseGroupedProbe, BaseProbe, PredictResult


@dataclass
class LrConfig:
    c: float = 1.0
    # We go for newton-cg as we've found it to be the fastest, see
    # experiments/scratch/lr_speed.py.
    solver: str = "newton-cg"
    max_iter: int = 10_000


@dataclass
class LogisticRegressionProbe(BaseProbe):
    model: LogisticRegression

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        logits = self.model.decision_function(activations)
        return PredictResult(logits=logits)


@dataclass
class LogisticRegressionGroupedProbe(BaseGroupedProbe, LogisticRegressionProbe):
    model: LogisticRegression

    @override
    def predict_grouped(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
        pairs: Int64[np.ndarray, "n"],  # noqa: F821
    ) -> PredictResult:
        activations_centered = _center_pairs(activations, pairs)
        logits = self.model.decision_function(activations_centered)
        return PredictResult(logits=logits)


def train_lr_probe(
    config: LrConfig,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> LogisticRegressionProbe:
    model = LogisticRegression(
        fit_intercept=True,
        solver=config.solver,
        C=config.c,
        max_iter=config.max_iter,
    )
    model.fit(activations, labels)
    return LogisticRegressionProbe(model)


def train_grouped_lr_probe(
    config: LrConfig,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    groups: Int64[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> LogisticRegressionGroupedProbe:
    probe = train_lr_probe(
        config,
        activations=_center_pairs(activations, groups),
        labels=labels,
    )
    return LogisticRegressionGroupedProbe(model=probe.model)


# TODO: Double check this preserves order.
def _center_pairs(
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    pairs: Int64[np.ndarray, "n"],  # noqa: F821
) -> Float[np.ndarray, "n d"]:  # noqa: F722
    df = pd.DataFrame(
        {
            "activations": list(activations),
            "pairs": pairs,
        }
    )
    pair_means = (
        df.groupby(["pairs"])["activations"]
        .apply(lambda a: np.mean(a, axis=0))
        .rename("pair_mean")  # type: ignore
    )
    df = df.join(pair_means, on="pairs")
    df["activations"] = df["activations"] - df["pair_mean"]
    return np.stack(df["activations"].to_list())
