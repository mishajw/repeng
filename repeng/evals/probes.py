import numpy as np
from jaxtyping import Bool, Float, Int64

from repeng.evals.logits import eval_logits_by_question, eval_logits_by_row
from repeng.evals.types import QuestionsEvalResult, RowsEvalResult
from repeng.probes.base import BaseProbe


def eval_probe_by_row(
    probe: BaseProbe,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> RowsEvalResult:
    result = probe.predict(activations)
    eval_result = eval_logits_by_row(
        logits=result.logits,
        labels=labels,
    )
    eval_result_flipped = eval_logits_by_row(
        logits=-result.logits,
        labels=labels,
    )
    if eval_result.roc_auc_score < eval_result_flipped.roc_auc_score:
        return eval_result_flipped.model_copy(
            update=dict(is_flipped=True),
        )
    else:
        return eval_result


def eval_probe_by_question(
    probe: BaseProbe,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    groups: Int64[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> QuestionsEvalResult:
    result = probe.predict(activations)
    eval_result = eval_logits_by_question(
        logits=result.logits,
        labels=labels,
        groups=groups,
    )
    eval_result_flipped = eval_logits_by_question(
        logits=-result.logits,
        labels=labels,
        groups=groups,
    )
    if eval_result.accuracy < eval_result_flipped.accuracy:
        return QuestionsEvalResult(
            accuracy=eval_result_flipped.accuracy,
            is_flipped=True,
            n=eval_result.n,
        )
    else:
        return eval_result
