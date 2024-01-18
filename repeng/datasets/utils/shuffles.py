import hashlib
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


def deterministic_shuffle(data: Iterable[T], key: Callable[[T], str]) -> list[T]:
    return sorted(
        data,
        key=lambda t: deterministic_shuffle_sort_fn(key(t), None),
    )


def deterministic_shuffle_sort_fn(key: str, _: Any) -> int:
    hash = hashlib.sha256(key.encode("utf-8"))
    return int(hash.hexdigest(), 16)
