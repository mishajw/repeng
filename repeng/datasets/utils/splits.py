import hashlib

from repeng.datasets.elk.types import Split

_TRAIN_WEIGHT = 0.6
_TRAIN_HPARAMS_WEIGHT = 0.2
# unused: _VALIDATION_WEIGHT = 0.2


def split_to_all(seed: str, row_id: str) -> Split:
    """
    Splits into train (60%), train-hparams (20%), and validation (20%).
    """
    hash = hashlib.sha256(f"{seed}-{row_id}".encode("utf-8"))
    hash = int(hash.hexdigest(), 16)
    prob = (hash % 1000) / 1000
    if prob < _TRAIN_WEIGHT:
        return "train"
    elif prob < _TRAIN_WEIGHT + _TRAIN_HPARAMS_WEIGHT:
        return "train-hparams"
    else:
        return "validation"


def split_train(split: Split, *, seed: str, row_id: str) -> Split:
    """
    Splits:
    - train -> train (75%), train-hparams (25%)
    - validation -> validation
    """
    if split == "validation":
        return "validation"
    hash = hashlib.sha256(f"{seed}-{row_id}".encode("utf-8"))
    hash = int(hash.hexdigest(), 16)
    prob = (hash % 1000) / 1000
    train_weight = _TRAIN_WEIGHT / (_TRAIN_WEIGHT + _TRAIN_HPARAMS_WEIGHT)
    if prob < train_weight:
        return "train"
    else:
        return "train-hparams"
