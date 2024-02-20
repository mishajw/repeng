"""
Replication of linear discriminant analysis (LDA) probes.

See LDA probes in <https://arxiv.org/abs/2312.01037v1> and MMP-IID described in
<https://arxiv.org/abs/2310.06824>.

Methodology:
1. Given a set of activations, and whether they respond to true or false statements.
2. Train a linear discriminant analysis model to separate the true and false
activations.

See <https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html>
for more details.

Intuitively, this takes the truth direction and then accounts for interference from
other features.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float
from overrides import override
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from repeng.probes.base import BaseProbe, PredictResult


@dataclass
class LdaProbe(BaseProbe):
    model: LinearDiscriminantAnalysis

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        logits = self.model.decision_function(activations)
        return PredictResult(
            logits=logits,
        )


def train_lda_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> LdaProbe:
    lda = LinearDiscriminantAnalysis()
    lda.fit(activations, labels)
    return LdaProbe(lda)
