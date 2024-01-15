from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float


@dataclass
class Activations:
    activations: Float[np.ndarray, "n d"]  # noqa: F722


@dataclass
class LabeledActivations:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821


@dataclass
class PairedActivations:
    activations_1: Float[np.ndarray, "n d"]  # noqa: F722
    activations_2: Float[np.ndarray, "n d"]  # noqa: F722
