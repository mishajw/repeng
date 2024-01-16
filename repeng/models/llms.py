from typing import Any, TypeVar, cast, get_args, overload

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from repeng.models import points
from repeng.models.types import Gpt2Id, Llama2Id, Llm, LlmId, PythiaId

_ModelT = TypeVar("_ModelT", bound=PreTrainedModel)
_TokenizerT = TypeVar("_TokenizerT", bound=PreTrainedTokenizerFast)


@overload
def get_llm(llm_id: PythiaId) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
    ...


@overload
def get_llm(llm_id: Gpt2Id) -> Llm[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    ...


@overload
def get_llm(llm_id: Llama2Id) -> Llm[LlamaForCausalLM, PreTrainedTokenizerFast]:
    ...


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
        points.gpt2(),
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
        points.pythia(pythia_id),
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
        points.llama2(llama_id),
    )
