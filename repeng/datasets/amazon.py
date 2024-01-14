import itertools
from typing import Literal

from pyparsing import Any

import datasets
from repeng.datasets.types import InputRow


def create_amazon_rows() -> dict[str, InputRow]:
    amazon_polarity: Any = datasets.load_dataset("amazon_polarity")

    def format(title: str, content: str, label: Literal["Positive", "Negative"]) -> str:
        return (
            f"Title: {title}\n"
            f"Review: {content}\n"
            f"Is the review positive or negative? {label}"
        )

    rows = {}
    for i, row in enumerate(itertools.islice(amazon_polarity["train"], 0, 1000)):
        assert row["label"] in [0, 1], row
        is_positive = row["label"] == 1
        rows[f"amazon-{i}-true"] = InputRow(
            pair_idx=str(i),
            text=format(row["title"], row["content"], "Positive"),
            is_text_true=is_positive,
            does_text_contain_true=True,
        )
        rows[f"amazon-{i}-false"] = InputRow(
            pair_idx=str(i),
            text=format(row["title"], row["content"], "Negative"),
            is_text_true=not is_positive,
            does_text_contain_true=False,
        )
    return rows
