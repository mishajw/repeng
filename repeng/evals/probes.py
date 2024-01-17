import sklearn.metrics
from pydantic import BaseModel

from repeng.activations.probe_preparations import LabeledActivationArray
from repeng.probes.base import BaseProbe


class ProbeEvalResult(BaseModel, extra="forbid"):
    f1_score: float
    precision: float
    recall: float


def evaluate_probe(
    probe: BaseProbe, activations: LabeledActivationArray
) -> ProbeEvalResult:
    predictions = probe.predict(activations.activations)
    labels = activations.labels
    if (predictions == labels).mean() < 0.5:
        predictions = ~predictions
    f1_score = sklearn.metrics.f1_score(labels, predictions)
    precision = sklearn.metrics.precision_score(labels, predictions)
    recall = sklearn.metrics.recall_score(labels, predictions)
    assert (
        isinstance(f1_score, float)
        and isinstance(precision, float)
        and isinstance(recall, float)
    )
    return ProbeEvalResult(f1_score=f1_score, precision=precision, recall=recall)
