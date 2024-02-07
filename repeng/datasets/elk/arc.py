from typing import Any, Literal

import datasets

from repeng.datasets.elk.types import BinaryRow, DatasetId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_train

# Taken from https://arxiv.org/abs/2310.01405 D.1.4.
_TEMPLATE = (
    "Consider the correctness of the answer to the following question:\n"
    "\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being correct is"
)
_SUBSET_TO_NAME = {"easy": "ARC-Easy", "challenge": "ARC-Challenge"}
_SUBSET_TO_ID: dict[str, DatasetId] = {"easy": "arc_easy", "challenge": "arc_challenge"}


def get_arc(subset: Literal["easy", "challenge"]) -> dict[str, BinaryRow]:
    return {
        **_get_arc_split(subset, "train"),
        **_get_arc_split(subset, "validation"),
    }


def _get_arc_split(
    subset: Literal["easy", "challenge"],
    split: Split,
) -> dict[str, BinaryRow]:
    pass
    dataset: Any = datasets.load_dataset("ai2_arc", _SUBSET_TO_NAME[subset])
    dataset_id = _SUBSET_TO_ID[subset]
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{dataset_id}-{group_id}-{choice_label}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split_train(split, seed=dataset_id, row_id=group_id),
                group_id=group_id,
                text=_TEMPLATE.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
