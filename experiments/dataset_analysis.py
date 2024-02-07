# %%
from pathlib import Path

import pandas as pd
import plotly.express as px
from mppr import MContext

from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.elk.utils.collections import get_dataset_collection

# %%
mcontext = MContext(Path("../output/dataset_analysis"))
datasets = mcontext.create_cached(
    "datasets", lambda: get_dataset_collection("all"), to=BinaryRow
)

# %%
df = pd.DataFrame(
    [row.model_dump() for row in datasets.get()],
)
df["word_counts"] = df["text"].apply(
    lambda row: len(row.split()),  # type: ignore
)
df  # type: ignore

# %%
px.bar(
    df.groupby(["dataset_id", "split"]).size().reset_index(),
    title="Num rows by dataset & split",
    x="dataset_id",
    y=0,
    facet_col="split",
    log_y=True,
)

# %%
px.bar(
    df.groupby("dataset_id")["word_counts"].mean().reset_index(),
    title="Average number of words per prompt by dataset",
    x="dataset_id",
    y="word_counts",
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
