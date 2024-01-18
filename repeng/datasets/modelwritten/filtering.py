import math
from typing import Literal

import openai
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionTokenLogprob,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from repeng.datasets.modelwritten import personas

_FILTERING_TEMPLATE = (
    "Suppose there is a person who {description}. "
    "Would they agree or disagree with the following statement? "
    '"{statement}"'
)


class FilteringConfig(BaseModel, extra="forbid"):
    model: Literal["gpt-3.5-turbo-1106", "gpt-4-1106-preview"] = "gpt-3.5-turbo-1106"


class StatementLikelihood(BaseModel, extra="forbid"):
    statement: str
    likelihood: float
    yes_logprobs: float | None
    no_logprobs: float | None


async def get_statement_likelihood(
    client: openai.AsyncClient,
    config: FilteringConfig,
    *,
    statement: str,
) -> StatementLikelihood:
    response = await client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content="Only reply 'Yes', or 'No' to the user's question.",
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=_FILTERING_TEMPLATE.format(
                    description=personas.HONEST,
                    statement=statement,
                ),
            ),
        ],
        model=config.model,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
        max_tokens=1,
    )
    assert len(response.choices) == 1, response

    choice = response.choices[0]
    assert choice.message.content is not None, choice
    assert choice.message.content in ["Yes", "No"], choice

    assert choice.logprobs is not None, choice
    assert choice.logprobs.content is not None, choice
    assert len(choice.logprobs.content) == 1, choice
    yes_logprobs = _get_logprobs("Yes", choice.logprobs.content[0])
    no_logprobs = _get_logprobs("No", choice.logprobs.content[0])
    yes_probs = math.exp(yes_logprobs) if yes_logprobs is not None else 0
    no_probs = math.exp(no_logprobs) if no_logprobs is not None else 0
    likelihood = yes_probs / (yes_probs + no_probs)

    return StatementLikelihood(
        statement=statement,
        likelihood=likelihood,
        yes_logprobs=yes_logprobs,
        no_logprobs=no_logprobs,
    )


def _get_logprobs(
    token_str: str, token_logprobs: ChatCompletionTokenLogprob
) -> float | None:
    for token in token_logprobs.top_logprobs:
        if token.token == token_str:
            return token.logprob
    return None
