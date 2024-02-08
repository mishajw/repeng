from typing import Any

import datasets

from repeng.datasets.elk.types import BinaryRow, DatasetId, RepeTemplateId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_train

_DATASET_IDS: dict[RepeTemplateId, DatasetId] = {
    "repe": "common_sense_qa",
    "qa": "common_sense_qa/qa",
}
# Taken from https://arxiv.org/abs/2310.01405 D.1.6.
_TEMPLATES: dict[RepeTemplateId, str] = {
    "repe": (
        "Based on commonsense reasoning, consider the plausibility of the\n"
        "answer to the following question:\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "The probability of the answer being plausible is"
    ),
    "qa": "Question: {question}\nAnswer: {answer}",
}


def get_common_sense_qa(template_id: RepeTemplateId) -> dict[str, BinaryRow]:
    return {
        **_get_common_sense_qa_split("train", template_id=template_id),
        **_get_common_sense_qa_split("validation", template_id=template_id),
    }


def _get_common_sense_qa_split(
    split: Split, template_id: RepeTemplateId
) -> dict[str, BinaryRow]:
    dataset_id = _DATASET_IDS[template_id]
    template = _TEMPLATES[template_id]
    dataset: Any = datasets.load_dataset("commonsense_qa")
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{dataset_id}-{group_id}-{choice_label}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split_train(split, seed="common_sense_qa", row_id=group_id),
                group_id=group_id,
                text=template.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
