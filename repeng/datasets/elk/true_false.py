import io
import zipfile
from typing import cast

import pandas as pd
import requests

from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.utils.shuffles import deterministic_shuffle
from repeng.datasets.utils.splits import split_to_all

_DATASET_ID = "true_false"


# TODO: Use the prompt template from the paper. I'm not entirely sure how to set it up,
# it seems that they only use true statements, but also say "an honest" vs. "a
# dishonest" person in the prompt. I should look at the code I guess?
def get_true_false_dataset() -> dict[str, BinaryRow]:
    result = {}
    dfs = _download_dataframes()
    for csv_name, df in deterministic_shuffle(dfs.items(), lambda row: row[0]):
        for index, row in deterministic_shuffle(df.iterrows(), lambda row: str(row[0])):
            assert isinstance(index, int)
            result[f"{csv_name}-{index}"] = BinaryRow(
                dataset_id=_DATASET_ID,
                split=split_to_all(_DATASET_ID, f"{csv_name}-{index}"),
                text=cast(str, row["statement"]),
                is_true=row["label"] == 1,
                format_args=dict(),
            )
    return result


def _download_dataframes() -> dict[str, pd.DataFrame]:
    response = requests.get(
        "http://azariaa.com/Content/Datasets/true-false-dataset.zip"
    )
    response.raise_for_status()
    file_stream = io.BytesIO(response.content)
    dataframes = {}
    with zipfile.ZipFile(file_stream, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            with zip_ref.open(file_name) as file:
                dataframes[file_name] = pd.read_csv(file)
    return dataframes
