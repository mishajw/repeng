from typing import Any

import datasets

from repeng.datasets.elk.types import BinaryRow, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_train

_DATASET_ID = "common_sense_qa"
# Taken from https://arxiv.org/abs/2310.01405 D.1.6.
_TEMPLATE = (
    "Based on commonsense reasoning, consider the plausibility of the\n"
    "answer to the following question:\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being plausible is"
)


def get_common_sense_qa() -> dict[str, BinaryRow]:
    return {
        **_get_common_sense_qa_split("train"),
        **_get_common_sense_qa_split("validation"),
    }


def _get_common_sense_qa_split(split: Split) -> dict[str, BinaryRow]:
    dataset: Any = datasets.load_dataset("commonsense_qa")
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{_DATASET_ID}-{group_id}-{choice_label}"] = BinaryRow(
                dataset_id=_DATASET_ID,
                split=split_train(split, seed=_DATASET_ID, row_id=group_id),
                group_id=group_id,
                text=_TEMPLATE.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
