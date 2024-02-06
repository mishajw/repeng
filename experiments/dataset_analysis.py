# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from repeng.datasets.elk.utils.collections import get_dataset_collection

# %%
datasets = get_dataset_collection("all")

# %%
df = pd.DataFrame(
    [row.model_dump() for row in datasets.values()],
    index=list(datasets.keys()),  # type: ignore
)
df["word_counts"] = df["text"].apply(
    lambda row: len(row.split()),  # type: ignore
)
df  # type: ignore

# %%
df.groupby(["dataset_id", "split"]).size().reset_index()

# %%
ax = sns.barplot(
    data=df.groupby(["dataset_id", "split"]).size().reset_index(),
    x="dataset_id",
    y=0,
    hue="split",
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Num rows by dataset")
plt.show()

ax = sns.barplot(
    df.groupby("dataset_id")["word_counts"].mean().reset_index(),
    x="dataset_id",
    y="word_counts",
)
plt.title("Average number of words per prompt by dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

ax = sns.barplot(
    df.groupby("dataset_id")["is_true"].mean().reset_index(),
    x="dataset_id",
    y="is_true",
)
plt.title("Percent of true prompts by dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

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
df["has_pair_id"] = df["pair_id"].apply(lambda x: x is not None)
df.groupby("dataset_id")["has_pair_id"].sum()
