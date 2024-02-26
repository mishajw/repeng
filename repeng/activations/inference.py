from typing import TypeVar

import numpy as np
import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from repeng.hooks.grab import grab_many
from repeng.models.llms import Llm
from repeng.utils.pydantic_ndarray import NdArray

ModelT = TypeVar("ModelT", bound=PreTrainedModel)


class ActivationRow(BaseModel, extra="forbid"):
    text: str
    text_tokenized: list[str]
    activations: dict[str, NdArray]
    token_logprobs: NdArray

    class Config:
        arbitrary_types_allowed = True


@torch.inference_mode()
def get_model_activations(
    llm: Llm[ModelT, PreTrainedTokenizerFast],
    *,
    text: str,
    last_n_tokens: int | None,
    points_start: int | None,
    points_end: int | None,
    points_skip: int | None,
) -> ActivationRow:
    assert last_n_tokens is None or last_n_tokens > 0, last_n_tokens

    tokens = llm.tokenizer.encode(text, return_tensors="pt")
    assert isinstance(tokens, torch.Tensor)
    tokens = tokens.to(next(llm.model.parameters()).device)
    tokens_str = llm.tokenizer.convert_ids_to_tokens(tokens.squeeze().tolist())

    points = llm.points[points_start:points_end:points_skip]
    with grab_many(llm.model, points) as activation_fn:
        output = llm.model.forward(tokens)
        logits: torch.Tensor = output.logits
        layer_activations = activation_fn()

    logprobs = logits.log_softmax(dim=-1)
    logprobs_shifted = logprobs[0, :-1]
    tokens_shifted = tokens[0, 1:, None]
    token_logprobs = (
        logprobs_shifted.gather(dim=-1, index=tokens_shifted).squeeze(0).detach()
    )

    def get_activation(activations: torch.Tensor) -> np.ndarray:
        activations = activations.squeeze(0)
        if last_n_tokens is not None:
            activations = activations[-last_n_tokens:]
        activations = activations.detach()
        # bfloat16 is not supported by numpy. We lose some precision by converting to
        # float16, but it significantly saves space an empirically makes little
        # difference.
        activations = activations.to(dtype=torch.float16)
        return activations.cpu().numpy()

    return ActivationRow(
        text=text,
        text_tokenized=tokens_str,
        activations={
            name: get_activation(activations)
            for name, activations in layer_activations.items()
        },
        token_logprobs=token_logprobs.float().cpu().numpy(),
    )
