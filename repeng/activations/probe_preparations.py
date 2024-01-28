from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int64


@dataclass
class Activation:
    dataset_id: str
    pair_id: str | None
    activations: Float[np.ndarray, "d"]  # noqa: F821
    label: bool


@dataclass
class ProbeArrays:
    activations: "ActivationArray"
    labeled: "LabeledActivationArray"
    paired: "PairedActivationArray | None"
    labeled_paired: "LabeledPairedActivationArray | None"


@dataclass
class ActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722


@dataclass
class LabeledActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821


@dataclass
class PairedActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    pairs: Int64[np.ndarray, "n"]  # noqa: F821


@dataclass
class LabeledPairedActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    pairs: Int64[np.ndarray, "n"]  # noqa: F821
    labels: Bool[np.ndarray, "n"]  # noqa: F821


def prepare_activations_for_probes(activations: Sequence[Activation]) -> ProbeArrays:
    df = pd.DataFrame([asdict(activation) for activation in activations])
    activation_array = ActivationArray(
        activations=np.stack(df["activations"].tolist()),
    )
    labeled_activations = LabeledActivationArray(
        activations=activation_array.activations,
        labels=df["label"].to_numpy(),
    )

    df_paired = df[df["pair_id"].notnull()]
    if len(df_paired) == 0:
        paired_activations = None
        labeled_paired_activations = None
    else:
        paired_activations = np.stack(df_paired["activations"].tolist())
        pairs = (
            df_paired["pair_id"].astype("category").cat.codes.to_numpy()  # type: ignore
        )
        paired_activations = PairedActivationArray(
            activations=paired_activations,
            pairs=pairs,
        )
        labeled_paired_activations = LabeledPairedActivationArray(
            activations=paired_activations.activations,
            pairs=paired_activations.pairs,
            labels=df_paired["label"].to_numpy(),  # type: ignore
        )

    return ProbeArrays(
        activations=activation_array,
        labeled=labeled_activations,
        paired=paired_activations,
        labeled_paired=labeled_paired_activations,
    )
