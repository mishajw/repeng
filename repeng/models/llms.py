from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, get_args

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from repeng.hooks.points import Point, TupleTensorExtractor

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
Gpt2Id = Literal["gpt2"]
Llama2Id = Literal[
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]
LlmId = PythiaId | Gpt2Id | Llama2Id


@dataclass
class Llm(Generic[_ModelT, _TokenizerT]):
    model: _ModelT
    tokenizer: _TokenizerT
    points: list[Point[_ModelT]]


def get_llm(llm_id: LlmId) -> Llm[Any, Any]:
    if llm_id in get_args(PythiaId):
        return pythia(cast(PythiaId, llm_id))
    elif llm_id in get_args(Gpt2Id):
        return gpt2()
    elif llm_id in get_args(Llama2Id):
        return llama2(cast(Llama2Id, llm_id))
    else:
        raise ValueError(f"Unknown LLM ID: {llm_id}")


def gpt2() -> Llm[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        [
            Point(
                f"h{i}",
                lambda model: model.transformer.h[i],
                tensor_extractor=TupleTensorExtractor(0),
            )
            for i in range(11)
        ],
    )


def pythia(
    pythia_id: PythiaId,
) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{pythia_id}",
        revision="step3000",
    )
    assert isinstance(model, GPTNeoXForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{pythia_id}",
        revision="step3000",
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        [
            Point(
                f"h{i}",
                lambda model: model.gpt_neox.layers[i],
                tensor_extractor=TupleTensorExtractor(0),
            )
            for i in range(len(model.gpt_neox.layers))
        ],
    )


def llama2(
    llama_id: Llama2Id,
) -> Llm[LlamaForCausalLM, PreTrainedTokenizerFast]:
    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{llama_id}")
    assert isinstance(model, LlamaForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{llama_id}")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        [
            Point(
                f"h{i}",
                lambda model: model.model.layers[i],
                tensor_extractor=TupleTensorExtractor(0),
            )
            for i in range(len(model.model.layers))
        ],
    )
