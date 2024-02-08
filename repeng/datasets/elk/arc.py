from typing import Any, Literal

import datasets

from repeng.datasets.elk.types import BinaryRow, DatasetId, RepeTemplateId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_train

ArcSubset = Literal["easy", "challenge"]

_DATASET_IDS: dict[tuple[ArcSubset, RepeTemplateId], DatasetId] = {
    ("easy", "repe"): "arc_easy",
    ("challenge", "repe"): "arc_challenge",
    ("easy", "qa"): "arc_easy/qa",
    ("challenge", "qa"): "arc_challenge/qa",
}
_TEMPLATE: dict[RepeTemplateId, str] = {
    # Taken from https://arxiv.org/abs/2310.01405 D.1.4.
    "repe": (
        "Consider the correctness of the answer to the following question:\n"
        "\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "The probability of the answer being correct is"
    ),
    "qa": "Question: {question}\nAnswer: {answer}",
}
_SUBSET_TO_NAME: dict[ArcSubset, str] = {
    "easy": "ARC-Easy",
    "challenge": "ARC-Challenge",
}


def get_arc(subset: ArcSubset, template_id: RepeTemplateId) -> dict[str, BinaryRow]:
    return {
        **_get_arc_split(subset=subset, split="train", template_id=template_id),
        **_get_arc_split(subset=subset, split="validation", template_id=template_id),
    }


def _get_arc_split(
    *,
    subset: ArcSubset,
    split: Split,
    template_id: RepeTemplateId,
) -> dict[str, BinaryRow]:
    dataset_id = _DATASET_IDS[(subset, template_id)]
    template = _TEMPLATE[template_id]
    dataset: Any = datasets.load_dataset("ai2_arc", _SUBSET_TO_NAME[subset])
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{dataset_id}-{group_id}-{choice_label}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split_train(split, seed="arc" + subset, row_id=group_id),
                group_id=group_id,
                text=template.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
