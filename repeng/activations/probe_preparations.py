from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int64

from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import Split
from repeng.datasets.elk.utils.filters import DatasetFilterId, filter_dataset
from repeng.models.types import LlmId


@dataclass
class ActivationArrays:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821
    groups: Int64[np.ndarray, "n"] | None  # noqa: F821
    answer_types: Int64[np.ndarray, "n"] | None  # noqa: F821


# @dataclass
# class ActivationRow:
#     dataset_id: DatasetId
#     split: Split
#     activations: Float[np.ndarray, "d"]  # noqa: F821
#     label: bool
#     group_id: str | None
#     answer_type: str | None


@dataclass
class ActivationArrayDataset:
    rows: list[ActivationResultRow]

    def get(
        self,
        *,
        llm_id: LlmId,
        dataset_filter_id: DatasetFilterId,
        split: Split,
        point_name: str | Literal["logprobs"],
        token_idx: int,
        limit: int | None,
    ) -> ActivationArrays:
        df = pd.DataFrame(
            [
                dict(
                    label=row.label,
                    group_id=row.group_id,
                    answer_type=row.answer_type,
                    activations=(
                        row.activations[point_name][token_idx].copy()
                        if point_name != "logprobs"
                        else np.array(row.prompt_logprobs)
                    ),
                )
                for row in self.rows
                if filter_dataset(
                    dataset_filter_id,
                    dataset_id=row.dataset_id,
                    answer_type=row.answer_type,
                )
                and row.split == split
                and row.llm_id == llm_id
            ][:limit]
        )
        assert not df.empty, (llm_id, dataset_filter_id, split, point_name, token_idx)

        group_counts = df["group_id"].value_counts().rename("group_count")
        df = df.join(group_counts, on="group_id")
        df.loc[df["group_count"] <= 1, "group_id"] = np.nan
        if df["group_id"].isna().all():  # type: ignore
            groups = None
        else:
            df = df[df["group_id"].notna()]
            groups = (
                df["group_id"].astype("category").cat.codes.to_numpy()  # type: ignore
            )

        if df["answer_type"].isna().any():  # type: ignore
            answer_types = None
        else:
            answer_types = (
                df["answer_type"]
                .astype("category")
                .cat.codes.to_numpy()  # type: ignore
            )

        return ActivationArrays(
            activations=np.stack(df["activations"].tolist()).astype(np.float32),
            labels=df["label"].to_numpy(),  # type: ignore
            groups=groups,
            answer_types=answer_types,
        )
