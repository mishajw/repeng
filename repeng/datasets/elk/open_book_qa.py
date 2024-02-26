from typing import Any

import datasets

from repeng.datasets.elk.types import BinaryRow, DatasetId, Split, TemplateType
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_train

_DATASET_IDS: dict[TemplateType, DatasetId] = {
    "repe": "open_book_qa",
    "simple": "open_book_qa/simple",
}
_TEMPLATES: dict[TemplateType, str] = {
    # Taken from https://arxiv.org/abs/2310.01405 D.1.5.
    "repe": (
        "Consider the correctness of the following fact:\n"
        "Fact: {question_stem} {choice}\n"
        "The probability of the fact being correct is"
    ),
    "simple": "{question_stem} {choice}",
}


def get_open_book_qa(template_type: TemplateType) -> dict[str, BinaryRow]:
    return {
        **_get_open_book_qa_split("train", template_type=template_type),
        **_get_open_book_qa_split("validation", template_type=template_type),
    }


def _get_open_book_qa_split(
    split: Split, template_type: TemplateType
) -> dict[str, BinaryRow]:
    dataset_id = _DATASET_IDS[template_type]
    template = _TEMPLATES[template_type]
    dataset: Any = datasets.load_dataset("openbookqa")
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question_stem=row["question_stem"], choice=choice)
            results[f"{dataset_id}-{group_id}-{choice_label}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split_train(split, seed="open_book_qa", row_id=group_id),
                group_id=group_id,
                text=template.format(**format_args),
                label=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
