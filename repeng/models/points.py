from typing import Any, cast, get_args, overload

from transformers import (
    GemmaForCausalLM,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
)

from repeng.hooks.points import Point, TupleTensorExtractor
from repeng.models.types import (
    PYTHIA_DPO_TO_PYTHIA,
    GemmaId,
    Gpt2Id,
    Llama2Id,
    LlmId,
    MistralId,
    PythiaDpoId,
    PythiaId,
)

_GPT2_NUM_LAYERS = 12
_PYTHIA_NUM_LAYERS: dict[PythiaId, int] = {
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
    "pythia-2.8b": 32,
    "pythia-6.9b": 32,
    "pythia-12b": 36,
}
_LLAMA2_NUM_LAYERS: dict[Llama2Id, int] = {
    "Llama-2-7b-hf": 32,
    "Llama-2-7b-chat-hf": 32,
    "Llama-2-13b-hf": 40,
    "Llama-2-13b-chat-hf": 40,
    "Llama-2-70b-hf": 80,
    "Llama-2-70b-chat-hf": 80,
}
_MISTRAL_NUM_LAYERS: dict[MistralId, int] = {
    "Mistral-7B": 32,
    "Mistral-7B-Instruct": 32,
}
_GEMMA_NUM_LAYERS: dict[GemmaId, int] = {
    "gemma-2b": 18,
    "gemma-2b-it": 18,
    "gemma-7b": 28,
    "gemma-7b-it": 28,
}


@overload
def get_points(llm_id: PythiaId | PythiaDpoId) -> list[Point[GPTNeoXForCausalLM]]:
    ...


@overload
def get_points(llm_id: Gpt2Id) -> list[Point[GPT2LMHeadModel]]:
    ...


@overload
def get_points(llm_id: Llama2Id) -> list[Point[LlamaForCausalLM]]:
    ...


@overload
def get_points(llm_id: MistralId) -> list[Point[MistralForCausalLM]]:
    ...


@overload
def get_points(llm_id: GemmaId) -> list[Point[GemmaForCausalLM]]:
    ...


def get_points(llm_id: LlmId) -> list[Point[Any]]:
    if llm_id in get_args(PythiaId):
        return pythia(cast(PythiaId, llm_id))
    elif llm_id in get_args(Gpt2Id):
        return gpt2()
    elif llm_id in get_args(Llama2Id):
        return llama2(cast(Llama2Id, llm_id))
    elif llm_id in get_args(PythiaDpoId):
        return pythia(PYTHIA_DPO_TO_PYTHIA[cast(PythiaDpoId, llm_id)])
    else:
        raise ValueError(f"Unknown LLM ID: {llm_id}")


def gpt2() -> list[Point[GPT2LMHeadModel]]:
    return [
        Point(
            f"h{i}",
            lambda model, i=i: model.transformer.h[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_GPT2_NUM_LAYERS)
    ]


def pythia(pythia_id: PythiaId) -> list[Point[GPTNeoXForCausalLM]]:
    return [
        Point(
            f"h{i}",
            lambda model, i=i: model.gpt_neox.layers[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_PYTHIA_NUM_LAYERS[pythia_id])
    ]


def llama2(llama2_id: Llama2Id) -> list[Point[LlamaForCausalLM]]:
    return [
        Point(
            f"h{i}",
            lambda model, i=i: model.model.layers[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_LLAMA2_NUM_LAYERS[llama2_id])
    ]


def mistral(mistral_id: MistralId) -> list[Point[MistralForCausalLM]]:
    return [
        Point(
            f"h{i}",
            lambda model, i=i: model.model.layers[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_MISTRAL_NUM_LAYERS[mistral_id])
    ]


def gemma(gemma_id: GemmaId) -> list[Point[GemmaForCausalLM]]:
    return [
        Point(
            f"h{i}",
            lambda model, i=i: model.model.layers[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_GEMMA_NUM_LAYERS[gemma_id])
    ]
