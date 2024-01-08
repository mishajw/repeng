from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel
from transformer_lens import HookedTransformer

from ccs.data.types import InputRow


class ActivationRow(BaseModel, extra="forbid"):
    input_row: InputRow
    activations: dict[str, np.ndarray]
    token_logprobs: np.ndarray

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ActivationArrays:
    rows_1: list[ActivationRow]
    rows_2: list[ActivationRow]
    activations_1: np.ndarray
    activations_2: np.ndarray
    logprobs_1: np.ndarray
    logprobs_2: np.ndarray
    is_text_true: np.ndarray


@torch.inference_mode()
def get_activations(
    model: HookedTransformer, input_row: InputRow, layers: list[str]
) -> ActivationRow:
    assert model.tokenizer is not None
    tokens = model.tokenizer.encode(input_row.text)
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
        input_row=input_row,
        activations={
            layer: cache[layer].float().squeeze(0)[-1].detach().cpu().numpy()
            for layer in layers
        },
        token_logprobs=token_logprobs.float().cpu().numpy(),
    )


def get_activation_arrays(
    rows: list[ActivationRow],
    layer: str,
    normalize: bool = True,
) -> ActivationArrays:
    rows = sorted(rows, key=lambda r: r.input_row.does_text_contain_true)
    rows = sorted(rows, key=lambda r: r.input_row.pair_idx)
    rows_1 = [row for row in rows if row.input_row.does_text_contain_true]
    rows_2 = [row for row in rows if not row.input_row.does_text_contain_true]
    assert all(
        r1.input_row.pair_idx == r2.input_row.pair_idx
        and r1.input_row.does_text_contain_true
        and not r2.input_row.does_text_contain_true
        for r1, r2 in zip(rows_1, rows_2)
    )

    activations_1 = np.stack([row.activations[layer] for row in rows_1])
    activations_2 = np.stack([row.activations[layer] for row in rows_2])
    if normalize:
        activations_1 = (activations_1 - np.mean(activations_1, axis=0)) / np.std(
            activations_1, axis=0
        )
        activations_2 = (activations_2 - np.mean(activations_2, axis=0)) / np.std(
            activations_2, axis=0
        )

    logprobs_1 = np.array([row.token_logprobs.sum().item() for row in rows_1])
    logprobs_2 = np.array([row.token_logprobs.sum().item() for row in rows_2])

    is_text_true = np.array([row.input_row.is_text_true for row in rows_1])

    return ActivationArrays(
        rows_1=rows_1,
        rows_2=rows_2,
        activations_1=activations_1,
        activations_2=activations_2,
        logprobs_1=logprobs_1,
        logprobs_2=logprobs_2,
        is_text_true=is_text_true,
    )
