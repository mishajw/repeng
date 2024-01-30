from dataclasses import dataclass

import numpy as np
import sklearn.metrics
from jaxtyping import Bool, Float, Int64

from repeng.evals.types import QuestionsEvalResult, RowsEvalResult


@dataclass
class LabeledGroupedLogits:
    logits: Float[np.ndarray, "n"]  # noqa: F821
    labels: Bool[np.ndarray, "n"]  # noqa: F821
    groups: Int64[np.ndarray, "n"]  # noqa: F821


@dataclass
class LabeledLogits:
    logits: Float[np.ndarray, "n"]  # noqa: F821
    labels: Bool[np.ndarray, "n"]  # noqa: F821


def eval_logits_by_question(
    labeled_logits: LabeledGroupedLogits,
) -> QuestionsEvalResult:
    group_labels: list[bool] = []
    for group in np.unique(labeled_logits.groups):
        labels = labeled_logits.labels[labeled_logits.groups == group]
        logits = labeled_logits.logits[labeled_logits.groups == group]
        if labels.sum() > 1:
            # Some datasets have multiple correct answers per question. We filter out
            # all but the first correct answer.
            mask = np.ones_like(labels)
            mask[labels] = False
            mask[labels.tolist().index(True)] = True
            labels = labels[mask]
            logits = logits[mask]
        group_labels.append(labels[np.argmax(logits)])
    accuracy = sum(group_labels) / len(group_labels)
    return QuestionsEvalResult(accuracy=accuracy, is_flipped=False)


def eval_logits_by_row(labeled_logits: LabeledLogits) -> RowsEvalResult:
    labels = labeled_logits.labels
    f1_score = sklearn.metrics.f1_score(
        labels,
        labeled_logits.labels,
        zero_division=0,  # type: ignore
    )
    precision = sklearn.metrics.precision_score(
        labels,
        labeled_logits.labels,
        zero_division=0,  # type: ignore
    )
    recall = sklearn.metrics.recall_score(
        labels,
        labeled_logits.labels,
        zero_division=0,  # type: ignore
    )
    accuracy = sklearn.metrics.accuracy_score(labels, labeled_logits.labels)
    predicted_true = labeled_logits.labels.mean().item()
    roc_auc_score = sklearn.metrics.roc_auc_score(labels, labeled_logits.logits)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, labeled_logits.logits)
    assert (
        isinstance(f1_score, float)
        and isinstance(precision, float)
        and isinstance(recall, float)
        and isinstance(roc_auc_score, float)
        and isinstance(accuracy, float)
    )
    return RowsEvalResult(
        f1_score=f1_score,
        precision=precision,
        recall=recall,
        roc_auc_score=roc_auc_score,
        accuracy=accuracy,
        predicted_true=predicted_true,
        fprs=fpr.tolist(),
        tprs=tpr.tolist(),
        logits=labeled_logits.logits.tolist(),
        is_flipped=False,
    )
