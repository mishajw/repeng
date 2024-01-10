from typing import TypeVar

import numpy as np
import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from repeng.hooks.grab import grab_many
from repeng.hooks.points import Point

ModelT = TypeVar("ModelT", bound=PreTrainedModel)


class ActivationRow(BaseModel, extra="forbid"):
    text: str
    activations: dict[str, np.ndarray]
    token_logprobs: np.ndarray

    class Config:
        arbitrary_types_allowed = True


@torch.inference_mode()
def get_activations(
    model: ModelT,
    tokenizer: PreTrainedTokenizerFast,
    points: list[Point[ModelT]],
    text: str,
) -> ActivationRow:
    tokens = tokenizer.encode(text)
    tokens = torch.tensor([tokens], device=next(model.parameters()).device)

    with grab_many(model, points) as activation_fn:
        output = model.forward(tokens)
        logits: torch.Tensor = output.logits
        layer_activations = activation_fn()

    logprobs = logits.log_softmax(dim=-1)
    logprobs_shifted = logprobs[0, :-1]
    tokens_shifted = tokens[0, 1:, None]
    token_logprobs = (
        logprobs_shifted.gather(dim=-1, index=tokens_shifted).squeeze(0).detach()
    )

    return ActivationRow(
        text=text,
        activations={
            name: activations.float().squeeze(0)[-1].detach().cpu().numpy()
            for name, activations in layer_activations.items()
        },
        token_logprobs=token_logprobs.float().cpu().numpy(),
    )
