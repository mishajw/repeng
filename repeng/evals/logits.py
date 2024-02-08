import numpy as np
import sklearn.metrics
from jaxtyping import Bool, Float, Int64

from repeng.evals.types import QuestionsEvalResult, RowsEvalResult


def eval_logits_by_question(
    *,
    logits: Float[np.ndarray, "n"],  # noqa: F821
    labels: Bool[np.ndarray, "n"],  # noqa: F821
    groups: Int64[np.ndarray, "n"],  # noqa: F821
) -> QuestionsEvalResult:
    correct: list[bool] = []
    for group in np.unique(groups):
        group_labels = labels[groups == group].copy()
        group_logits = logits[groups == group].copy()
        max_logit = np.max(group_logits)
        # Instead of just doing argmax, we randomly select among the max logits. This is
        # because some of the datasets always have the correct answer first, so
        # outputting identical logits for all answers would have 100% accuracy.
        max_idx = np.random.choice(np.where(group_logits == max_logit)[0])
        # N.B.: Some datasets have multiple correct answers per question. This is fine,
        # we just check that one of the correct answers was chosen.
        correct.append(group_labels[max_idx])  # type: ignore
    accuracy = sum(correct) / len(correct)
    return QuestionsEvalResult(accuracy=accuracy, is_flipped=False)


def eval_logits_by_row(
    *,
    logits: Float[np.ndarray, "n"],  # noqa: F821
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> RowsEvalResult:
    # TODO: Should we infer the threshold from the ROC?
    labels_pred = logits > 0
    f1_score = sklearn.metrics.f1_score(
        labels,
        labels_pred,
        zero_division=0,  # type: ignore
    )
    precision = sklearn.metrics.precision_score(
        labels,
        labels_pred,
        zero_division=0,  # type: ignore
    )
    recall = sklearn.metrics.recall_score(
        labels,
        labels_pred,
        zero_division=0,  # type: ignore
    )
    accuracy = sklearn.metrics.accuracy_score(labels, labels_pred)
    predicted_true = labels_pred.mean().item()
    roc_auc_score = sklearn.metrics.roc_auc_score(labels, logits)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, logits)
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
        logits=logits.tolist(),
        is_flipped=False,
    )
