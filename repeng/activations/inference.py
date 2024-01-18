from typing import TypeVar

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
    text: str,
) -> ActivationRow:
    tokens = llm.tokenizer.encode(text)
    tokens_str = llm.tokenizer.tokenize(text)
    tokens = llm.tokenizer.convert_tokens_to_ids(tokens_str)
    tokens = torch.tensor([tokens], device=next(llm.model.parameters()).device)

    with grab_many(llm.model, llm.points) as activation_fn:
        output = llm.model.forward(tokens)
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
        text_tokenized=tokens_str,
        activations={
            name: activations.float().squeeze(0)[-1].detach().cpu().numpy()
            for name, activations in layer_activations.items()
        },
        token_logprobs=token_logprobs.float().cpu().numpy(),
    )