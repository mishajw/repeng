from pydantic import BaseModel


class QuestionsEvalResult(BaseModel, extra="forbid"):
    accuracy: float
    is_flipped: bool
    n: int
    """
    Some probes are trained with unsupervised methods, thus the probe is predicting the
    inverse of what we expect. In these cases, we flip the logits.
    """


class RowsEvalResult(BaseModel, extra="forbid"):
    f1_score: float
    precision: float
    recall: float
    roc_auc_score: float
    accuracy: float
    predicted_true: float
    fprs: list[float]
    tprs: list[float]
    logits: list[float]
    is_flipped: bool
    n: int
