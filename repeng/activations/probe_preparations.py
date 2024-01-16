from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float


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
    paired: "PairedActivationArray"


@dataclass
class ActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722


@dataclass
class LabeledActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821


@dataclass
class PairedActivationArray:
    activations_1: Float[np.ndarray, "n d"]  # noqa: F722
    activations_2: Float[np.ndarray, "n d"]  # noqa: F722


def prepare_activations_for_probes(activations: Sequence[Activation]) -> ProbeArrays:
    df = pd.DataFrame([asdict(activation) for activation in activations])

    activation_array = ActivationArray(
        activations=np.stack(df["activations"].to_list()),
    )

    labeled_activations = LabeledActivationArray(
        activations=activation_array.activations,
        labels=df["label"].to_numpy(),
    )

    df_paired = df.groupby(["dataset_id", "pair_id", "label"]).first()
    df_paired = df_paired.reset_index()
    df_paired = df_paired.pivot(index="pair_id", columns="label", values="activations")
    df_paired = df_paired.dropna()
    paired_activations = PairedActivationArray(
        activations_1=np.stack(df_paired[True].to_list()),
        activations_2=np.stack(df_paired[False].to_list()),
    )
    print(paired_activations.activations_1[0, :10])
    print(paired_activations.activations_2[0, :10])

    return ProbeArrays(
        activations=activation_array,
        labeled=labeled_activations,
        paired=paired_activations,
    )
