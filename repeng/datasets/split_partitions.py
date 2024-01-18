import hashlib

from repeng.datasets.elk.types import Split

_VALIDATION_SPLIT = 0.2


def get_split(seed: str, row_id: str) -> Split:
    hash = hashlib.sha256(f"{seed}-{row_id}".encode("utf-8"))
    hash = int(hash.hexdigest(), 16)
    prob = (hash % 1000) / 1000
    return "validation" if prob < _VALIDATION_SPLIT else "train"
