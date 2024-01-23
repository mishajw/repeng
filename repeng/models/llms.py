from typing import Any, cast, get_args, overload

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from repeng.models import points
from repeng.models.types import Gpt2Id, Llama2Id, Llm, LlmId, PythiaDpoId, PythiaId


@overload
def get_llm(
    llm_id: PythiaId | PythiaDpoId,
) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
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
    elif llm_id in get_args(PythiaDpoId):
        return pythia_dpo(cast(PythiaDpoId, llm_id))
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
        revision="step143000",
    )
    assert isinstance(model, GPTNeoXForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{pythia_id}",
        revision="step143000",
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        points.pythia(pythia_id),
    )


def pythia_dpo(
    pythia_dpo_id: PythiaDpoId,
) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
    if pythia_dpo_id == "pythia-dpo-1b":
        model_id = "Leogrin/eleuther-pythia1b-hh-dpo"
        pythia_id = "pythia-1b"
    elif pythia_dpo_id == "pythia-sft-1b":
        model_id = "Leogrin/eleuther-pythia1b-hh-sft"
        pythia_id = "pythia-1b"
    elif pythia_dpo_id == "pythia-dpo-1.4b":
        model_id = "Leogrin/eleuther-pythia1.4b-hh-dpo"
        pythia_id = "pythia-1.4b"
    elif pythia_dpo_id == "pythia-sft-1.4b":
        model_id = "Leogrin/eleuther-pythia1.4b-hh-sft"
        pythia_id = "pythia-1.4b"
    else:
        raise ValueError(f"Unknown Pythia DPO ID: {pythia_dpo_id}")
    model = GPTNeoXForCausalLM.from_pretrained(model_id)
    assert isinstance(model, GPTNeoXForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(model, tokenizer, points.pythia(pythia_id))


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
