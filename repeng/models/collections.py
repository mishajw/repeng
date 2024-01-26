from typing import Literal, cast, get_args

from repeng.models.types import LlmId

LlmCollectionId = Literal[
    "pythia",
    "pythia-tiny",
    "pythia-rtx3090",
]


def resolve_llm_ids(llm_collection_id: LlmCollectionId | LlmId) -> list[LlmId]:
    if llm_collection_id in get_args(LlmCollectionId):
        return _get_llm_collection(cast(LlmCollectionId, llm_collection_id))
    elif llm_collection_id in get_args(LlmId):
        return [cast(LlmId, llm_collection_id)]
    else:
        raise ValueError(f"Unknown LLM collection ID: {llm_collection_id}")


def _get_llm_collection(llm_collection_id: LlmCollectionId) -> list[LlmId]:
    if llm_collection_id == "pythia":
        return [
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b",
        ]
    elif llm_collection_id == "pythia-tiny":
        return [
            "pythia-70m",
            "pythia-160m",
        ]
    elif llm_collection_id == "pythia-rtx3090":
        return [
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
        ]
    else:
        raise ValueError(f"Unknown LLM collection ID: {llm_collection_id}")
