import warnings

import sklearn.metrics
from pydantic import BaseModel

from repeng.activations.probe_preparations import LabeledActivationArray
from repeng.probes.base import BaseProbe


class ProbeEvalResult(BaseModel, extra="forbid"):
    f1_score: float
    precision: float
    recall: float
    roc_auc_score: float
    fprs: list[float]
    tprs: list[float]
    logits: list[float]


def evaluate_probe(
    probe: BaseProbe, activations: LabeledActivationArray
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
        )

    result = probe.predict(activations.activations)
    labels = activations.labels
    # TODO:
    # if (predictions == labels).mean() < 0.5:
    #     predictions = ~predictions
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
    )
