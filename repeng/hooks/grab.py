from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

import torch

from repeng.hooks.points import Point

ModelT = TypeVar("ModelT", bound=torch.nn.Module)


@contextmanager
def grab(
    model: ModelT, point: Point[ModelT]
) -> Generator[Callable[[], torch.Tensor], None, None]:
    result: torch.Tensor | None = None

    def hook(
        _module: torch.nn.Module,
        _input: Any,
        output: Any,
    ) -> None:
        nonlocal result
        assert result is None, f"Hook called multiple times for point {point.name}"
        result = point.tensor_extractor.extract(output)
        assert isinstance(
            result, torch.Tensor
        ), f"Hook returned non-tensor for point {point.name}, type={type(result)}"

    def get_result() -> torch.Tensor:
        assert result is not None, f"Hook not called for point {point.name}"
        return result

    hook_handle = None
    try:
        hook_handle = point.module_fn(model).register_forward_hook(hook)
        yield get_result
    finally:
        if hook_handle is not None:
            hook_handle.remove()


@contextmanager
def grab_many(
    model: ModelT, points: list[Point[ModelT]]
) -> Generator[Callable[[], dict[str, torch.Tensor]], None, None]:
    context_managers = None
    try:
        context_managers = {point.name: grab(model, point) for point in points}
        result_fns = {
            name: context_manager.__enter__()
            for name, context_manager in context_managers.items()
        }
        yield lambda: {name: result_fn() for name, result_fn in result_fns.items()}
    finally:
        if context_managers is not None:
            for context_manager in context_managers.values():
                context_manager.__exit__(None, None, None)
