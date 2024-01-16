from typing import Any, cast, get_args, overload

from transformers import GPT2LMHeadModel, GPTNeoXForCausalLM, LlamaForCausalLM

from repeng.hooks.points import Point, TupleTensorExtractor
from repeng.models.types import Gpt2Id, Llama2Id, LlmId, PythiaId

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


@overload
def get_points(llm_id: PythiaId) -> list[Point[GPTNeoXForCausalLM]]:
    ...


@overload
def get_points(llm_id: Gpt2Id) -> list[Point[GPT2LMHeadModel]]:
    ...


@overload
def get_points(llm_id: Llama2Id) -> list[Point[LlamaForCausalLM]]:
    ...


def get_points(llm_id: LlmId) -> list[Point[Any]]:
    if llm_id in get_args(PythiaId):
        return pythia(cast(PythiaId, llm_id))
    elif llm_id in get_args(Gpt2Id):
        return gpt2()
    elif llm_id in get_args(Llama2Id):
        return llama2(cast(Llama2Id, llm_id))
    else:
        raise ValueError(f"Unknown LLM ID: {llm_id}")


def gpt2() -> list[Point[GPT2LMHeadModel]]:
    return [
        Point(
            f"h{i}",
            lambda model: model.transformer.h[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_GPT2_NUM_LAYERS)
    ]


def pythia(pythia_id: PythiaId) -> list[Point[GPTNeoXForCausalLM]]:
    return [
        Point(
            f"h{i}",
            lambda model: model.gpt_neox.layers[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_PYTHIA_NUM_LAYERS[pythia_id])
    ]


def llama2(llama2_id: Llama2Id) -> list[Point[LlamaForCausalLM]]:
    return [
        Point(
            f"h{i}",
            lambda model: model.model.layers[i],
            tensor_extractor=TupleTensorExtractor(0),
        )
        for i in range(_LLAMA2_NUM_LAYERS[llama2_id])
    ]
