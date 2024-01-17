# %%
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import openai
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from mppr import MContext
from pydantic import BaseModel

from repeng.modelwrittenevals.filtering import (
    FilteringConfig,
    StatementLikelihood,
    get_statement_likelihood,
)
from repeng.modelwrittenevals.generation import (
    GenerationConfig,
    Statements,
    generate_statements,
)

NUM_HONEST_GENERATIONS = 100
NUM_DISHONEST_GENERATIONS = 100
GENERATION_CONFIG = GenerationConfig()
FILTERING_CONFIG = FilteringConfig()


class Statement(BaseModel, extra="forbid"):
    statement: str
    honest: bool


class TruthfulModelWrittenEvalRow(BaseModel, extra="forbid"):
    statement: str
    honest: bool
    likelihood: float
    yes_logprobs: float | None
    no_logprobs: float | None


# %%
load_dotenv("../.env")
client = openai.AsyncClient()
mcontext = MContext(Path("../output/truthful_model_written_evals-v2"))


# %%
async def create_df() -> pd.DataFrame:
    init = mcontext.create(
        {
            **{f"honest_{i}": True for i in range(NUM_HONEST_GENERATIONS)},
            **{f"dishonest_{i}": False for i in range(NUM_HONEST_GENERATIONS)},
        },
    )
    statements = await init.amap_cached(
        "generation",
        fn=lambda _, value: generate_statements(
            client,
            GENERATION_CONFIG,
            agrees=value,
        ),
        to=Statements,
    )
    statements_flat = statements.flat_map(
        lambda key, statements: {
            f"{key}-statement{i}": Statement(
                statement=statement,
                honest=statements.agrees,
            )
            for i, statement in enumerate(statements.statements)
        },
    ).filter(
        filter_repeating_statements(),
    )
    statement_likelihoods = await statements_flat.amap_cached(
        "filter",
        fn=lambda _, value: get_statement_likelihood(
            client,
            FILTERING_CONFIG,
            statement=value.statement,
        ),
        to=StatementLikelihood,
    )
    result = statement_likelihoods.join(
        statements_flat,
        lambda _, likelihood, statement: TruthfulModelWrittenEvalRow(
            statement=statement.statement,
            honest=statement.honest,
            likelihood=likelihood.likelihood,
            yes_logprobs=likelihood.yes_logprobs,
            no_logprobs=likelihood.no_logprobs,
        ),
    ).filter(
        filter_low_likelihood,
    )
    result.upload(
        "../repeng/datasets/data/truthful",
        to=TruthfulModelWrittenEvalRow,
    )
    # TODO: Filter out statements that don't pass the filter.
    return result.to_dataframe(lambda row: row.model_dump())


def filter_repeating_statements() -> Callable[[str, Statement], bool]:
    seen = set()

    def fn(_, value: Statement) -> bool:
        statement_normalized = value.statement.lower().strip(" .")
        if statement_normalized in seen:
            return False
        seen.add(statement_normalized)
        return True

    return fn


def filter_low_likelihood(_: str, value: TruthfulModelWrittenEvalRow) -> bool:
    if value.honest:
        return value.likelihood > 0.5
    else:
        return value.likelihood < 0.5


df = await create_df()  # noqa: F704

#  %%
fig, axs = plt.subplots(ncols=3, figsize=(3 * 5, 5))
sns.histplot(df, x="likelihood", hue="honest", bins=40, ax=axs[0])
sns.histplot(df, x="yes_logprobs", hue="honest", bins=40, ax=axs[1])
sns.histplot(df, x="no_logprobs", hue="honest", bins=40, ax=axs[2])

# %%
df["statement_normalized"] = df["statement"].str.lower().str.strip(" .")
print(
    df["statement_normalized"].count(),
    df["statement_normalized"].nunique(),
    df["statement_normalized"].nunique() / df["statement_normalized"].count(),
)
print(df["statement_normalized"].value_counts().tail(10))

# %%
print(df["honest"].value_counts())

# %%
df[df["honest"]].sort_values(by="likelihood", ascending=True).head(10)

# %%
df[~df["honest"]].sort_values(by="likelihood", ascending=False).head(10)

# %%
# # gpt-4-1106-preview
# cost_per_input_token = 0.01 / 1000
# cost_per_output_token = 0.03 / 1000
# gpt-3.5-turbo-1106
cost_per_input_token = 0.001 / 1000
cost_per_output_token = 0.002 / 1000
num_input_tokens = 300
num_output_tokens = (
    df["statement"].apply(len).sum()
    / (NUM_HONEST_GENERATIONS + NUM_DISHONEST_GENERATIONS)
    * 1.1
)
cost_per_generation = (
    cost_per_input_token * num_input_tokens + cost_per_output_token * num_output_tokens
)
print(f"Cost per generation: ${cost_per_generation:.5f}")
print(f"Cost for 1K generations: ${cost_per_generation * 1000:.5f}")
