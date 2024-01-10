from contextlib import contextmanager
from typing import Callable, Generator, TypeVar

import torch

from repeng.hooks.points import Point

ModelT = TypeVar("ModelT", bound=torch.nn.Module)


@contextmanager
def patch(
    model: ModelT,
    point: Point[ModelT],
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Generator[None, None, None]:
    hook_handle = None
    try:
        hook_handle = point.module_fn(model).register_forward_hook(
            lambda _module, _input, output: point.tensor_extractor.insert(
                output, fn(point.tensor_extractor.extract(output))
            )
        )
        yield None
    finally:
        if hook_handle is not None:
            hook_handle.remove()
