from typing import Any

import torch

from repeng.models.llms import Llm, LlmId, get_llm

_loaded_llm_id: LlmId | None = None
_loaded_llm: Llm[Any, Any] | None = None


def load_llm_oioo(
    llm_id: LlmId,
    device: torch.device,
    dtype: torch.dtype,
) -> Llm[Any, Any]:
    """
    Loads an LLM with a one-in-one-out policy, i.e. only one model is loaded into
    memory at a time.
    """
    global _loaded_llm_id
    global _loaded_llm
    if llm_id != _loaded_llm_id:
        if _loaded_llm is not None:
            _loaded_llm.model = _loaded_llm.model.cpu()
            del _loaded_llm
            torch.cuda.empty_cache()
            print(f"Unloaded LLM {_loaded_llm_id}, loading LLM {llm_id}")
        else:
            print(f"Loading LLM {llm_id}")
        _loaded_llm_id = llm_id
        _loaded_llm = get_llm(llm_id)
        _loaded_llm.model = _loaded_llm.model.to(dtype=dtype).to(device=device)
    assert _loaded_llm is not None
    return _loaded_llm
