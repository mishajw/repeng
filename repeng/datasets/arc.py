from typing import Any, Literal

import datasets

from repeng.datasets.types import PairedBinaryRow

# Taken from https://arxiv.org/abs/2310.01405 D.1.4.
_TEMPLATE = (
    "Consider the correctness of the answer to the following question:\n"
    "\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being correct is"
)
_SUBSET_TO_NAME = {"easy": "ARC-Easy", "challenge": "ARC-Challenge"}
_SUBSET_TO_ID = {"easy": "arc_easy", "challenge": "arc_challenge"}


def get_arc(subset: Literal["easy", "challenge"]) -> dict[str, PairedBinaryRow]:
    dataset: Any = datasets.load_dataset("ai2_arc", _SUBSET_TO_NAME[subset])
    dataset_id = _SUBSET_TO_ID[subset]
    results = {}
    for row in dataset["train"]:
        pair_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{dataset_id}-{pair_id}-{choice_label}"] = PairedBinaryRow(
                dataset_id=dataset_id,
                pair_id=pair_id,
                text=_TEMPLATE.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
                format_style="lat",
            )
    return results
