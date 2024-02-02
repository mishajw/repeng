from repeng.activations.probe_preparations import (
    LabeledActivationArray,
    LabeledGroupedActivationArray,
)
from repeng.evals.logits import (
    LabeledGroupedLogits,
    LabeledLogits,
    eval_logits_by_question,
    eval_logits_by_row,
)
from repeng.evals.types import QuestionsEvalResult, RowsEvalResult
from repeng.probes.base import BaseProbe


def eval_probe_by_row(
    probe: BaseProbe, activations: LabeledActivationArray
) -> RowsEvalResult:
    result = probe.predict(activations.activations)
    eval_result = eval_logits_by_row(
        LabeledLogits(
            logits=result.logits,
            labels=activations.labels,
        )
    )
    eval_result_flipped = eval_logits_by_row(
        LabeledLogits(
            logits=-result.logits,
            labels=activations.labels,
        )
    )
    if eval_result.roc_auc_score < eval_result_flipped.roc_auc_score:
        return eval_result_flipped.model_copy(
            update=dict(is_flipped=True),
        )
    else:
        return eval_result


def eval_probe_by_question(
    probe: BaseProbe,
    activations: LabeledGroupedActivationArray,
) -> QuestionsEvalResult:
    result = probe.predict(activations.activations)
    eval_result = eval_logits_by_question(
        LabeledGroupedLogits(
            logits=result.logits,
            labels=activations.labels,
            groups=activations.groups,
        )
    )
    eval_result_flipped = eval_logits_by_question(
        LabeledGroupedLogits(
            logits=-result.logits,
            labels=activations.labels,
            groups=activations.groups,
        )
    )
    if eval_result.accuracy < eval_result_flipped.accuracy:
        return QuestionsEvalResult(
            accuracy=eval_result_flipped.accuracy,
            is_flipped=True,
        )
    else:
        return eval_result
