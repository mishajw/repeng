# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from repeng.datasets.collections import get_all_datasets

# %%
datasets = get_all_datasets()

# %%
df = pd.DataFrame(
    [row.model_dump() for row in datasets.values()],
    index=list(datasets.keys()),
)
df["word_counts"] = df["text"].apply(
    lambda row: len(row.split()),  # type: ignore
)
df

# %%
ax = sns.barplot(data=df["dataset_id"].value_counts())
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Num rows by dataset")
plt.show()

ax = sns.barplot(df.groupby("dataset_id")["word_counts"].mean())
plt.title("Average number of words per prompt by dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()

ax = sns.barplot(df.groupby("dataset_id")["is_true"].mean())
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
