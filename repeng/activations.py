import numpy as np
import torch
from pydantic import BaseModel
from transformer_lens import HookedTransformer


class ActivationRow(BaseModel, extra="forbid"):
    text: str
    activations: dict[str, np.ndarray]
    token_logprobs: np.ndarray

    class Config:
        arbitrary_types_allowed = True


@torch.inference_mode()
def get_activations(
    model: HookedTransformer, text: str, layers: list[str]
) -> ActivationRow:
    assert model.tokenizer is not None
    tokens = model.tokenizer.encode(text)
    tokens = torch.tensor([tokens], device=next(model.parameters()).device)

    logits: torch.Tensor
    logits, cache = model.run_with_cache(  # type: ignore
        tokens,
        names_filter=layers,
    )

    logprobs = logits.log_softmax(dim=-1)
    logprobs_shifted = logprobs[0, :-1]
    tokens_shifted = tokens[0, 1:, None]
    token_logprobs = (
        logprobs_shifted.gather(dim=-1, index=tokens_shifted).squeeze(0).detach()
    )

    return ActivationRow(
        text=text,
        activations={
            layer: cache[layer].float().squeeze(0)[-1].detach().cpu().numpy()
            for layer in layers
        },
        token_logprobs=token_logprobs.float().cpu().numpy(),
    )
