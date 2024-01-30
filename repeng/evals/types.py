from pydantic import BaseModel


class QuestionsEvalResult(BaseModel, extra="forbid"):
    accuracy: float
    is_flipped: bool
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
    # is_grouped: bool

    # def __str__(self) -> str:
    #     s = "ProbeEvalResult(\n"
    #     s += "\t"
    #     s += "\n\t".join(
    #         f"{name:<9} = {score*100:.1f}%"
    #         for name, score in [
    #             ("roc_auc", self.roc_auc_score),
    #             ("accuracy", self.accuracy),
    #             ("f1", self.f1_score),
    #             ("precision", self.precision),
    #             ("recall", self.recall),
    #             ("pred_true", self.predicted_true),
    #         ]
    #     )
    #     s += "\n)"
    #     return s
