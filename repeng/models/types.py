from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from transformers import PreTrainedModel, PreTrainedTokenizerFast

from repeng.hooks.points import Point

_ModelT = TypeVar("_ModelT", bound=PreTrainedModel)
_TokenizerT = TypeVar("_TokenizerT", bound=PreTrainedTokenizerFast)

PythiaId = Literal[
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]
PythiaDpoId = Literal[
    "pythia-dpo-1b",
    "pythia-dpo-1.4b",
    "pythia-sft-1b",
    "pythia-sft-1.4b",
]
Gpt2Id = Literal["gpt2"]
Llama2Id = Literal[
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]
LlmId = PythiaId | PythiaDpoId | Gpt2Id | Llama2Id


@dataclass
class Llm(Generic[_ModelT, _TokenizerT]):
    model: _ModelT
    tokenizer: _TokenizerT
    points: list[Point[_ModelT]]
