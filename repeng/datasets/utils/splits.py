import hashlib

from repeng.datasets.elk.types import Split

_TRAIN_WEIGHT = 0.6
_TRAIN_HPARAMS_WEIGHT = 0.2
# unused: _VALIDATION_WEIGHT = 0.2


def split_to_all(seed: str, row_id: str) -> Split:
    """
    Splits into train (60%), train-hparams (20%), and validation (20%).
    """
    prob = _get_prob(seed=seed, row_id=row_id)
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
    prob = _get_prob(seed=seed, row_id=row_id)
    train_weight = _TRAIN_WEIGHT / (_TRAIN_WEIGHT + _TRAIN_HPARAMS_WEIGHT)
    if prob < train_weight:
        return "train"
    else:
        return "train-hparams"


def split_validation(*, seed: str, row_id: str) -> Split:
    """
    Splits into validation (80%) and train-hparams (20%).

    Used for datasets that are never trained on, for example TruthfulQA.
    """
    prob = _get_prob(seed=seed, row_id=row_id)
    validation_weight = 0.8
    if prob < validation_weight:
        return "validation"
    else:
        return "train-hparams"


def _get_prob(*, seed: str, row_id: str) -> float:
    hash = hashlib.sha256(f"{seed}-{row_id}".encode("utf-8"))
    hash = int(hash.hexdigest(), 16)
    return (hash % 1000) / 1000
