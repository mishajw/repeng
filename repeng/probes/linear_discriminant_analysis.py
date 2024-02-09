"""
Replication of linear discriminant analysis (LDA) probes.

See LDA probes in <https://arxiv.org/abs/2312.01037v1> and MMP-IID described in
<https://arxiv.org/abs/2310.06824>.
"""

import numpy as np
from jaxtyping import Bool, Float
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from repeng.probes.base import DotProductProbe


def train_lda_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> DotProductProbe:
    lda = LinearDiscriminantAnalysis()
    lda.fit(activations, labels)
    coefficients = lda.coef_[0]
    return DotProductProbe(coefficients)
