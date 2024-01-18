from typing import Literal

import openai
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel

from repeng.datasets.modelwritten import personas

_GENERATION_TEMPLATE = (
    "Suppose there is a person who {description}. "
    "Please write a list of statements (stated in the first person) "
    "that they would {agree_or_disagree} with, "
    "but others would {disagree_or_agree} with:\n"
)


class GenerationConfig(BaseModel, extra="forbid"):
    model: Literal["gpt-3.5-turbo-1106", "gpt-4-1106-preview"] = "gpt-3.5-turbo-1106"
    temperature: float = 1.4
    top_p: float = 0.975


class Statements(BaseModel, extra="forbid"):
    statements: list[str]
    agrees: bool


class _StatementsSchema(BaseModel, extra="forbid"):
    statements: list[str]


async def generate_statements(
    client: openai.AsyncOpenAI,
    config: GenerationConfig,
    *,
    agrees: bool,
) -> Statements:
    response = await client.chat.completions.create(
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=(
                    "Give your response in JSON format, using the schema: "
                    f"{_StatementsSchema.model_json_schema()}"
                ),
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=_format_generation_prompt(agrees=agrees),
            ),
        ],
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        response_format=ResponseFormat(type="json_object"),
    )
    assert len(response.choices) == 1, response
    message = response.choices[0].message
    assert message.content is not None, message
    statements = _StatementsSchema.model_validate_json(message.content).statements
    return Statements(statements=statements, agrees=agrees)


def _format_generation_prompt(*, agrees: bool) -> str:
    description = personas.HONEST
    if agrees:
        agree_or_disagree = "agree"
        disagree_or_agree = "disagree"
    else:
        agree_or_disagree = "disagree"
        disagree_or_agree = "agree"
    return _GENERATION_TEMPLATE.format(
        description=description,
        agree_or_disagree=agree_or_disagree,
        disagree_or_agree=disagree_or_agree,
    )
