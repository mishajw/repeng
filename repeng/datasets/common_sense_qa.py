from typing import Any

import datasets

from repeng.datasets.types import PairedText

_DATASET_ID = "common_sense_qa"
# Taken from https://arxiv.org/abs/2310.01405 D.1.6.
_TEMPLATE = (
    "Based on commonsense reasoning, consider the plausibility of the\n"
    "answer to the following question:\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being plausible is"
)


def get_common_sense_qa() -> dict[str, PairedText]:
    dataset: Any = datasets.load_dataset("commonsense_qa")
    results = {}
    for row in dataset["train"]:
        pair_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{_DATASET_ID}-{pair_id}-{choice_label}"] = PairedText(
                dataset_id=_DATASET_ID,
                pair_id=pair_id,
                text=_TEMPLATE.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
                format_style="lat",
            )
    return results