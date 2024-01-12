import io
import zipfile

import pandas as pd
import requests

from repeng.datasets.types import BinaryRow

_DATASET_ID = "true_false"


def get_true_false_dataset() -> dict[str, BinaryRow]:
    result = {}
    dfs = _download_dataframes()
    for csv_name, df in dfs.items():
        for index, row in df.iterrows():
            result[f"{csv_name}-{index}"] = BinaryRow(
                dataset_id=_DATASET_ID,
                text=row["statement"],
                is_true=row["label"] == 1,
                format_args=dict(),
                format_style="misc",
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
