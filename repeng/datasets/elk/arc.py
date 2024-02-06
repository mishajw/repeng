from typing import Any, Literal

import datasets

from repeng.datasets.elk.types import BinaryRow, DatasetId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle

# Taken from https://arxiv.org/abs/2310.01405 D.1.4.
_TEMPLATE_REPE = (
    "Consider the correctness of the answer to the following question:\n"
    "\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being correct is"
)
_TEMPLATE_QNA = "Question: {question}\nAnswer: {answer}"
_SUBSET_TO_NAME = {"easy": "ARC-Easy", "challenge": "ARC-Challenge"}
_SUBSET_TO_ID: dict[tuple[str, str], DatasetId] = {
    ("easy", "repe"): "arc_easy",
    ("challenge", "repe"): "arc_challenge",
    ("easy", "qna"): "arc_easy/qna",
    ("challenge", "qna"): "arc_challenge/qna",
}


def get_arc(subset: Literal["easy", "challenge"]) -> dict[str, BinaryRow]:
    return {
        **_get_arc_split(subset, split="train", template_type="repe"),
        **_get_arc_split(subset, split="validation", template_type="repe"),
    }


def get_arc_qna(subset: Literal["easy", "challenge"]) -> dict[str, BinaryRow]:
    return {
        **_get_arc_split(subset, split="train", template_type="qna"),
        **_get_arc_split(subset, split="validation", template_type="qna"),
    }


def _get_arc_split(
    subset: Literal["easy", "challenge"],
    *,
    split: Split,
    template_type: Literal["repe", "qna"],
) -> dict[str, BinaryRow]:
    if template_type == "repe":
        template = _TEMPLATE_REPE
    elif template_type == "qna":
        template = _TEMPLATE_QNA
    else:
        raise ValueError(template_type)

    dataset: Any = datasets.load_dataset("ai2_arc", _SUBSET_TO_NAME[subset])
    dataset_id = _SUBSET_TO_ID[(subset, template_type)]
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[
                f"{dataset_id}-{group_id}-{choice_label}-{template_type}"
            ] = BinaryRow(
                dataset_id=dataset_id,
                split=split,
                group_id=group_id,
                text=template.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
