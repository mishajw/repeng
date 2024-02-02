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
    grouped: "GroupedActivationArray | None"
    labeled_grouped: "LabeledGroupedActivationArray | None"


@dataclass
class ActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722


@dataclass
class LabeledActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821


@dataclass
class GroupedActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    groups: Int64[np.ndarray, "n"]  # noqa: F821


@dataclass
class LabeledGroupedActivationArray:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    groups: Int64[np.ndarray, "n"]  # noqa: F821
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

    df_grouped = df[df["pair_id"].notnull()]
    group_sizes = df_grouped.groupby("pair_id").size()
    valid_groups = (group_sizes[group_sizes > 1]).index
    df_grouped = df_grouped[df_grouped["pair_id"].isin(valid_groups)]
    if len(df_grouped) == 0:
        grouped_activations = None
        labeled_grouped_activations = None
    else:
        grouped_activations = np.stack(df_grouped["activations"].tolist())
        groups = (
            df_grouped["pair_id"]
            .astype("category")
            .cat.codes.to_numpy()  # type: ignore
        )
        grouped_activations = GroupedActivationArray(
            activations=grouped_activations,
            groups=groups,
        )
        labeled_grouped_activations = LabeledGroupedActivationArray(
            activations=grouped_activations.activations,
            groups=grouped_activations.groups,
            labels=df_grouped["label"].to_numpy(),  # type: ignore
        )

    return ProbeArrays(
        activations=activation_array,
        labeled=labeled_activations,
        grouped=grouped_activations,
        labeled_grouped=labeled_grouped_activations,
    )
