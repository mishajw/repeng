# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from repeng.datasets.arc import get_arc
from repeng.datasets.common_sense_qa import get_common_sense_qa
from repeng.datasets.geometry_of_truth import get_geometry_of_truth
from repeng.datasets.open_book_qa import get_open_book_qa
from repeng.datasets.race import get_race
from repeng.datasets.true_false import get_true_false_dataset
from repeng.datasets.truthful_qa import get_truthful_qa
from repeng.datasets.types import BinaryRow, PairedBinaryRow

# %%
binary_datasets: dict[str, BinaryRow] = {
    **get_true_false_dataset(),
    **get_geometry_of_truth("cities"),
    **get_geometry_of_truth("neg_cities"),
    **get_geometry_of_truth("sp_en_trans"),
    **get_geometry_of_truth("neg_sp_en_trans"),
    **get_geometry_of_truth("larger_than"),
    **get_geometry_of_truth("smaller_than"),
    **get_geometry_of_truth("cities_cities_conj"),
    **get_geometry_of_truth("cities_cities_disj"),
}
paired_binary_datasets: dict[str, PairedBinaryRow] = {
    **get_arc("challenge"),
    **get_arc("easy"),
    **get_common_sense_qa(),
    **get_open_book_qa(),
    **get_race(),
    **get_truthful_qa(),
}
datasets = {**binary_datasets, **paired_binary_datasets}

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
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title("Num rows by dataset")
plt.show()

ax = sns.barplot(df.groupby("dataset_id")["word_counts"].mean())
plt.title("Average number of words per prompt by dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

ax = sns.barplot(df.groupby("dataset_id")["is_true"].mean())
plt.title("Percent of true prompts by dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
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
