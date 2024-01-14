from typing import Any

import datasets
from repeng.datasets.types import BinaryRow

_DATASET_ID = "open_book_qa"
# Taken from https://arxiv.org/abs/2310.01405 D.1.5.
_TEMPLATE = (
    "Consider the correctness of the following fact:\n"
    "Fact: {question_stem} {choice}\n"
    "The probability of the fact being correct is"
)


def get_open_book_qa() -> dict[str, BinaryRow]:
    dataset: Any = datasets.load_dataset("openbookqa")
    results = {}
    for row in dataset["train"]:
        pair_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question_stem=row["question_stem"], choice=choice)
            results[f"{_DATASET_ID}-{pair_id}-{choice_label}"] = BinaryRow(
                dataset_id=_DATASET_ID,
                pair_id=pair_id,
                text=_TEMPLATE.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
                format_style="lat",
            )
    return results
