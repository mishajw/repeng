import warnings
from typing import cast

import sklearn.metrics
from pydantic import BaseModel

from repeng.activations.probe_preparations import (
    LabeledActivationArray,
    LabeledPairedActivationArray,
)
from repeng.probes.base import BasePairedProbe, BaseProbe, PredictResult


class ProbeEvalResult(BaseModel, extra="forbid"):
    f1_score: float
    precision: float
    recall: float
    roc_auc_score: float
    fprs: list[float]
    tprs: list[float]
    logits: list[float]
    is_paired: bool


def evaluate_probe(
    probe: BaseProbe, activations: LabeledActivationArray
) -> ProbeEvalResult:
    result = probe.predict(activations.activations)
    return _evaluate(result, activations, is_paired=False)


def evaluate_paired_probe(
    probe: BasePairedProbe,
    activations: LabeledPairedActivationArray,
) -> ProbeEvalResult:
    result = probe.predict_paired(activations.activations, activations.pairs)
    return _evaluate(
        result,
        LabeledActivationArray(activations.activations, activations.labels),
        is_paired=True,
    )


def _evaluate(
    result: PredictResult,
    activations: LabeledActivationArray,
    *,
    is_paired: bool,
) -> ProbeEvalResult:
    if len(set(activations.labels)) == 1:
        warnings.warn("Only one class in labels")
        return ProbeEvalResult(
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            roc_auc_score=0.0,
            fprs=[],
            tprs=[],
            logits=[],
            is_paired=is_paired,
        )

    labels = activations.labels
    if cast(float, sklearn.metrics.roc_auc_score(labels, result.logits)) < 0.5:
        # TODO: Is this correct?
        result.logits = -result.logits
        result.labels = ~result.labels
    f1_score = sklearn.metrics.f1_score(labels, result.labels, zero_division=0)
    precision = sklearn.metrics.precision_score(labels, result.labels, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, result.labels, zero_division=0)
    roc_auc_score = sklearn.metrics.roc_auc_score(labels, result.logits)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, result.logits)
    assert (
        isinstance(f1_score, float)
        and isinstance(precision, float)
        and isinstance(recall, float)
        and isinstance(roc_auc_score, float)
    )
    return ProbeEvalResult(
        f1_score=f1_score,
        precision=precision,
        recall=recall,
        roc_auc_score=roc_auc_score,
        fprs=fpr.tolist(),
        tprs=tpr.tolist(),
        logits=result.logits.tolist(),
        is_paired=is_paired,
    )
