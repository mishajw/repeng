from typing import Any

import datasets

from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import get_split

_DATASET_ID = "truthful_qa"
# Taken from https://arxiv.org/abs/2310.01405 D.1.1.
_TEMPLATE = (
    "Consider the amount of truthfulness in the following answer:\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The amount of truthfulness in the answer is"
)


def get_truthful_qa() -> dict[str, BinaryRow]:
    dataset: Any = datasets.load_dataset("truthful_qa", "multiple_choice")
    results = {}
    for group_id, row in deterministic_shuffle(
        enumerate(dataset["validation"]), lambda row: str(row[0])
    ):
        for answer_idx, (answer, is_correct) in enumerate(
            zip(
                row["mc1_targets"]["choices"],
                row["mc1_targets"]["labels"],
            )
        ):
            format_args = dict(question=row["question"], answer=answer)
            results[f"{_DATASET_ID}-{group_id}-{answer_idx}"] = BinaryRow(
                dataset_id=_DATASET_ID,
                split=get_split(_DATASET_ID, str(group_id)),
                group_id=str(group_id),
                text=_TEMPLATE.format(**format_args),
                is_true=is_correct == 1,
                format_args=format_args,
            )
    return results
