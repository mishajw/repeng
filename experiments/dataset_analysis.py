# %%
from pathlib import Path

import pandas as pd
import plotly.express as px
from mppr import MContext, MDict
from pydantic import BaseModel

from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.collections import resolve_dataset_ids
from repeng.datasets.elk.utils.fns import get_dataset


# %%
class Dataset(BaseModel, extra="forbid"):
    rows: dict[str, BinaryRow]


mcontext = MContext(Path("../output/dataset_analysis"))
dataset_ids: MDict[DatasetId] = mcontext.create(
    {dataset_id: dataset_id for dataset_id in resolve_dataset_ids("all")},
)
datasets = dataset_ids.map_cached(
    "datasets",
    lambda _, dataset_id: Dataset(rows=get_dataset(dataset_id)),
    to=Dataset,
).flat_map(lambda _, dataset: {key: row for key, row in dataset.rows.items()})

# %%
df = pd.DataFrame([row.model_dump() for row in datasets.get()])
df["word_counts"] = df["text"].apply(lambda row: len(row.split()))
df  # type: ignore

# %%
px.bar(
    df.groupby(["dataset_id", "split"]).size().reset_index(),
    title="Num rows by dataset & split",
    x="dataset_id",
    y=0,
    color="dataset_id",
    facet_row="split",
    log_y=True,
    height=1000,
)

# %%
px.bar(
    df.groupby(["dataset_id", "split"])["group_id"]
    .nunique()
    .rename("num_groups")
    .reset_index(),
    title="Num groups by dataset & split",
    x="dataset_id",
    y="num_groups",
    color="dataset_id",
    facet_row="split",
    log_y=True,
    height=1000,
)

# %%
px.bar(
    df.groupby("dataset_id")["word_counts"].mean().reset_index(),
    title="Average number of words per prompt by dataset",
    x="dataset_id",
    y="word_counts",
    color="dataset_id",
    log_y=True,
)

# %%
fig = px.bar(
    df.groupby("dataset_id")["is_true"].mean().reset_index(),
    title="Percent of true prompts by dataset",
    x="dataset_id",
    y="is_true",
    range_y=[0, 1],
)
fig.add_hline(1 / 2, line_dash="dot", line_color="gray")
fig.add_hline(1 / 3, line_dash="dot", line_color="gray")
fig.add_hline(1 / 4, line_dash="dot", line_color="gray")
fig.add_hline(1 / 5, line_dash="dot", line_color="gray")

# %%
for dataset_id in df["dataset_id"].unique():
    row = df[df["dataset_id"] == dataset_id].sample(1)
    print("#", dataset_id)
    print("## text")
    print(row["text"].item())
    print("## is true")
    print(row["is_true"].item())
    print()

# %%
df["has_group_id"] = df["group_id"].apply(lambda x: x is not None)
df.groupby("dataset_id")["has_group_id"].sum()
