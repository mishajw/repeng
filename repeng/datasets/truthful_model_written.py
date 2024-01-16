import hashlib
from importlib import resources

import jsonlines

from repeng.datasets import data
from repeng.datasets.types import BinaryRow, Split

_DATASET_ID = "truthful_model_written"
_TEMPLATE = (
    "Is the following something you would say?\n" '"{statement}"\n' "Answer: {answer}"
)


def get_truthful_model_written() -> dict[str, BinaryRow]:
    truthful_json = resources.files(data) / "truthful.jsonl"
    with jsonlines.open(str(truthful_json)) as reader:
        results = {}
        for row in reader:
            for answer in True, False:
                answer_str = "Yes" if answer else "No"
                format_args = dict(
                    statement=row["value"]["statement"], answer=answer_str
                )
                text = _TEMPLATE.format(**format_args)
                results[row["key"]] = BinaryRow(
                    dataset_id=_DATASET_ID,
                    split=_get_split(key=row["key"]),
                    text=text,
                    is_true=(row["value"]["honest"] and answer)
                    or (not row["value"]["honest"] and not answer),
                    format_args=dict(),
                    format_style="misc",
                )
    return results


def _get_split(*, key: str) -> Split:
    hash = hashlib.sha256(key.encode("utf-8"))
    hash = hash.hexdigest()
    hash = int(hash, 16)
    if hash % 5 != 0:
        return "train"
    else:
        return "validation"
