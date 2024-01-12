import random
from typing import Literal

from repeng.datasets.types import InputRow


def create_addition_rows() -> dict[str, InputRow]:
    rows = {}
    for i in range(1000):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        if random.random() < 0.5:
            error = 1 if random.random() < 0.5 else -1
        else:
            error = 0
        c = a + b + error
        rows[f"{i}_true"] = InputRow(
            pair_idx=str(i),
            text=_add_prefix_v1(f"Does {a} + {b} = {c}? Yes"),
            is_text_true=a + b == c,
            does_text_contain_true=True,
        )
        rows[f"{i}_false"] = InputRow(
            pair_idx=str(i),
            text=_add_prefix_v1(f"Does {a} + {b} = {c}? No"),
            is_text_true=a + b != c,
            does_text_contain_true=False,
        )
    return rows


def create_addition_rows_v2() -> dict[str, InputRow]:
    rows = {}
    for i in range(1000):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        c = a + b
        error = 1 if random.random() < 0.5 else -1
        if random.random() < 0.5:
            c = c + error
            error = -error
        rows[f"{i}_a"] = InputRow(
            pair_idx=str(i),
            text=_add_prefix_v2(f"{a} + {b} = {c}"),
            is_text_true=a + b == c,
            does_text_contain_true=True,
        )
        rows[f"{i}_b"] = InputRow(
            pair_idx=str(i),
            text=_add_prefix_v2(f"{a} + {b} = {c + error}"),
            is_text_true=a + b == c + error,
            does_text_contain_true=False,
        )
    return rows


def create_addition_rows_chat() -> dict[str, InputRow]:
    def format(a: int, b: int, c: int, answer: Literal["Yes", "No"]) -> str:
        return (
            "<|SYSTEM|>You are a professional mathematician. "
            "You are responsible for answering simple maths questions. "
            "Only answer with Yes/No.\n"
            f"<|USER|>Does {a} + {b} = {c}?\n"
            f"<|SYSTEM|>{answer}"
        )

    rows = {}
    for i in range(1000):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        if random.random() < 0.5:
            error = 1 if random.random() < 0.5 else -1
        else:
            error = 0
        c = a + b + error
        rows[f"{i}_true"] = InputRow(
            pair_idx=str(i),
            text=format(a, b, c, "Yes"),
            is_text_true=a + b == c,
            does_text_contain_true=True,
        )
        rows[f"{i}_false"] = InputRow(
            pair_idx=str(i),
            text=format(a, b, c, "No"),
            is_text_true=a + b != c,
            does_text_contain_true=False,
        )
    return rows


def _add_prefix_v1(text: str) -> str:
    return "\n".join(
        [
            "Here, we'll be adding two numbers together, "
            "and seeing if the answer is correct.",
            "Does 45 + 21 = 66? Yes",
            "Does 9 + 74 = 84? No",
            text,
        ]
    )


def _add_prefix_v2(text: str) -> str:
    return f"4 + 2 = 6\n{text}"
