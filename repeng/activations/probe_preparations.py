from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int64

from repeng.datasets.elk.types import DatasetId, Split
from repeng.datasets.elk.utils.filters import DatasetFilterId, filter_dataset


@dataclass
class ActivationArrays:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821
    groups: Int64[np.ndarray, "n"] | None  # noqa: F821
    answer_types: Int64[np.ndarray, "n"] | None  # noqa: F821


@dataclass
class ActivationRow:
    dataset_id: DatasetId
    split: Split
    activations: Float[np.ndarray, "d"]  # noqa: F821
    label: bool
    group_id: str | None
    answer_type: str | None


@dataclass
class ActivationDataset:
    rows: list[ActivationRow]

    def get_unsupervised(
        self, dataset_filter_id: DatasetFilterId, split: Split
    ) -> ActivationArrays | None:
        df = pd.DataFrame(
            [
                asdict(row)
                for row in self.rows
                if filter_dataset(
                    dataset_filter_id,
                    dataset_id=row.dataset_id,
                    template_name=row.group_id,
                )
                and row.split == split
            ]
        )

        group_counts = df["group_id"].value_counts().rename("group_count")
        df = df.join(group_counts, on="group_id")
        df[df["group_count"] <= 1]["group_id"] = None
        if df["group_id"].isna().any():  # type: ignore
            groups = None
        else:
            groups = (
                df["group_id"].astype("category").cat.codes.to_numpy()  # type: ignore
            )
            df = df[df["group_id"].notna()]

        if df["answer_type"].isna().any():  # type: ignore
            answer_types = None
        else:
            answer_types = (
                df["answer_type"]
                .astype("category")
                .cat.codes.to_numpy()  # type: ignore
            )

        return ActivationArrays(
            activations=np.stack(df["activations"].tolist()),
            labels=df["label"].to_numpy(),  # type: ignore
            groups=groups,
            answer_types=answer_types,
        )
