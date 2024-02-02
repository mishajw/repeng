import numpy as np
from jaxtyping import Float, Int64


def normalize_by_group(
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    group: Int64[np.ndarray, "n"],  # noqa: F821
) -> Float[np.ndarray, "n d"]:  # noqa: F722
    result = activations.copy()
    for g in np.unique(group):
        result[group == g] -= result[group == g].mean(axis=0)
    return result
