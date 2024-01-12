from typing import Any, Literal

import datasets

from repeng.datasets.types import PairedText

# Taken from https://arxiv.org/abs/2310.01405 D.1.4.
_TEMPLATE = (
    "Consider the correctness of the answer to the following question:\n"
    "\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "The probability of the answer being correct is"
)


def get_arc(subset: Literal["ARC-Easy", "ARC-Challenge"]) -> dict[str, PairedText]:
    dataset: Any = datasets.load_dataset("ai2_arc", subset)
    results = {}
    for row in dataset["train"]:
        pair_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{subset}-{pair_id}-{choice_label}"] = PairedText(
                dataset_id=subset,
                pair_id=pair_id,
                text=_TEMPLATE.format(**format_args),
                is_true=row["answerKey"] == choice_label,
                format_args=format_args,
                format_style="lat",
            )
    return results
