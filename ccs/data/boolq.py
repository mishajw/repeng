import datasets
from pyparsing import Any

from main import InputRow


def create_boolq_rows() -> list[InputRow]:
    boolq: Any = datasets.load_dataset("boolq")
    rows = []
    for i, row in enumerate(boolq["train"]):
        passage = row["passage"]
        question = row["question"]
        text = f"{passage}\n\n{question}?"
        rows.append(
            InputRow(
                pair_idx=str(i),
                text=f"{text}\nYes",
                is_text_true=row["answer"],
                does_text_contain_true=True,
            )
        )
        rows.append(
            InputRow(
                pair_idx=str(i),
                text=f"{text}\nNo",
                is_text_true=not row["answer"],
                does_text_contain_true=False,
            )
        )
    return rows
