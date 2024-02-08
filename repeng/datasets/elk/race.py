from typing import Any

import datasets

from repeng.datasets.elk.types import BinaryRow, DatasetId, RepeTemplateId, Split
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_train

_DATASET_IDS: dict[RepeTemplateId, DatasetId] = {
    "repe": "race",
    "qa": "race/qa",
}
_TEMPLATES: dict[RepeTemplateId, str] = {
    # Taken from https://arxiv.org/abs/2310.01405 D.1.7.
    "repe": (
        "Consider the correctness of the answer to the following question based on the "
        "article:\n"
        "\n"
        "Article: {article}\n"
        "\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "The probability of the answer being correct is"
    ),
    "qa": "Article: {article}\n\nQuestion: {question}\nAnswer: {answer}",
}
_ANSWER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def get_race(template_id: RepeTemplateId) -> dict[str, BinaryRow]:
    return {
        **_get_race_split("train", template_id=template_id),
        **_get_race_split("validation", template_id=template_id),
    }


def _get_race_split(split: Split, template_id: RepeTemplateId) -> dict[str, BinaryRow]:
    dataset_id = _DATASET_IDS[template_id]
    template = _TEMPLATES[template_id]
    dataset: Any = datasets.load_dataset("race", "all")
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["example_id"]):
        group_id = row["example_id"]
        for option_idx, option in enumerate(row["options"]):
            format_args = dict(
                article=row["article"], question=row["question"], answer=option
            )
            results[f"{dataset_id}-{group_id}-{option_idx}"] = BinaryRow(
                dataset_id=dataset_id,
                split=split_train(split, seed="race", row_id=group_id),
                group_id=group_id,
                text=template.format(**format_args),
                is_true=_ANSWER_TO_INDEX[row["answer"]] == option_idx,
                format_args=format_args,
            )
    return results
