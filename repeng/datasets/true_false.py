import io
import zipfile

import pandas as pd
import requests
from pydantic import BaseModel


class TrueFalseRow(BaseModel, extra="forbid"):
    statement: str
    is_true: bool


def get_true_false_dataset() -> dict[str, TrueFalseRow]:
    result = {}
    dfs = _download_dataframes()
    for csv_name, df in dfs.items():
        for index, row in df.iterrows():
            result[f"{csv_name}-{index}"] = TrueFalseRow(
                statement=row["statement"],
                is_true=row["label"] == 1,
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
