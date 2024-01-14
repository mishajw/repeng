from typing import Literal

import datasets
from pyparsing import Any

from repeng.datasets.types import InputRow


def create_boolq_rows() -> dict[str, InputRow]:
    def format(passage: str, question: str, answer: Literal["Yes", "No"]) -> str:
        return "\n".join(
            [
                passage,
                f"Question: {question}?",
                f"Answer: {answer}",
            ]
        )

    boolq: Any = datasets.load_dataset("boolq")
    rows = {}
    for i, row in enumerate(boolq["train"]):
        rows[f"boolq-{i}-true"] = InputRow(
            pair_idx=str(i),
            text=format(row["passage"], row["question"], "Yes"),
            is_text_true=row["answer"],
            does_text_contain_true=True,
        )
        rows[f"boolq-{i}-false"] = InputRow(
            pair_idx=str(i),
            text=format(row["passage"], row["question"], "No"),
            is_text_true=not row["answer"],
            does_text_contain_true=False,
        )
    return rows
