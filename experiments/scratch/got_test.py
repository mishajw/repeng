# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from dotenv import load_dotenv
from mppr import MContext
from sklearn.decomposition import PCA
from tqdm import tqdm

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.elk.utils.filters import DATASET_FILTER_FNS, DatasetIdFilter
from repeng.datasets.elk.utils.fns import get_datasets
from repeng.datasets.elk.utils.limits import Limits, SplitLimits, limit_groups
from repeng.evals.probes import eval_probe_by_question
from repeng.models.loading import load_llm_oioo
from repeng.models.points import get_points
from repeng.models.types import LlmId
from repeng.probes.collections import ALL_PROBES, SUPERVISED_PROBES, train_probe

assert load_dotenv(".env")

# %%
# path = Path("../../output/comparison")
# mcontext = MContext(path)
# activation_results: list[ActivationResultRow] = mcontext.download_cached(
#     "activations_results",
#     path="s3://repeng/datasets/activations/datasets_2024-02-08_v1.pickle",
#     to="pickle",
# ).get()
# print(set(row.llm_id for row in activation_results))
# print(set(row.dataset_id for row in activation_results))
# print(set(row.split for row in activation_results))
# dataset = ActivationArrayDataset(activation_results)

# %%
mcontext = MContext(Path("../output/got_test"))
limits = Limits(
    default=SplitLimits(train=150, train_hparams=0, validation=200),
    by_dataset={},
)
# llm_ids: list[LlmId] = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"]
llm_ids: list[LlmId] = ["Llama-2-13b-chat-hf"]
inputs = (
    mcontext.create_cached(
        "dataset-repe",
        lambda: get_datasets(
            [
                # "geometry_of_truth/cities",
                "race",
                "open_book_qa",
                "arc_easy",
                "arc_challenge",
            ]
        ),
        to=BinaryRow,
    )
    .filter(
        limit_groups(limits),
    )
    .flat_map(lambda key, row: {f"{key}-{llm_id}": (row, llm_id) for llm_id in llm_ids})
    .sort(lambda _, row: llm_ids.index(row[1]))
)
activations = inputs.map_cached(
    "activations-v3",
    lambda _, row: get_model_activations(
        load_llm_oioo(
            llm_id=row[1],
            device=torch.device("cuda"),
            use_half_precision=row[1] == "Llama-2-13b-chat-hf",
        ),
        text=row[0].text,
        last_n_tokens=1,
    ),
    to="pickle",
).join(
    inputs,
    lambda _, activations, input: ActivationResultRow(
        dataset_id=input[0].dataset_id,
        group_id=input[0].group_id,
        answer_type=input[0].answer_type,
        activations=activations.activations,
        prompt_logprobs=activations.token_logprobs.sum(),
        label=input[0].is_true,
        split=input[0].split,
        llm_id=input[1],
    ),
)
dataset = ActivationArrayDataset(activations.get())

# %%
for llm_id in llm_ids[1:]:
    arrays = dataset.get(
        llm_id=llm_id,
        dataset_filter=DatasetIdFilter("geometry_of_truth/cities"),
        # dataset_filter=DatasetIdFilter("geometry_of_truth/sp_en_trans"),
        # dataset_filter=DATASET_FILTER_FNS["geometry_of_truth/cities/pos"],
        # dataset_filter=DATASET_FILTER_FNS["geometry_of_truth/cities/neg"],
        split="validation",
        # point_name=get_points(llm_id)[-10].name,
        point_name="h13",
        token_idx=0,
        limit=None,
    )
    assert arrays.answer_types is not None
    assert arrays.groups is not None
    activations = arrays.activations.copy()
    # for answer_type in np.unique(arrays.answer_types):
    #     activations[answer_type == arrays.answer_types] -= np.mean(
    #         activations[answer_type == arrays.answer_types], axis=0
    #     )
    for group in np.unique(arrays.groups):
        activations[group == arrays.groups] -= np.mean(
            activations[group == arrays.groups], axis=0
        )
    activations = activations

    components = PCA(n_components=3).fit_transform(activations)
    fig = px.scatter_3d(
        title=llm_id,
        x=components[:, 0],
        y=components[:, 1],
        z=components[:, 2],
        color=arrays.labels,
        symbol=arrays.answer_types,
        category_orders={"color": [False, True]},
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(showlegend=False)
    fig.show()

# %%
results = []
for llm_id in llm_ids:
    for point in tqdm(get_points(llm_id)[::4]):
        arrays = dataset.get(
            llm_id=llm_id,
            dataset_filter=DatasetIdFilter("geometry_of_truth/cities"),
            split="validation",
            point_name=point.name,
            token_idx=0,
            limit=None,
        )
        assert arrays.answer_types is not None
        components = PCA(n_components=3).fit_transform(
            arrays.activations.astype(np.float16)
        )
        for i in range(components.shape[0]):
            results.append(
                dict(
                    llm_id=llm_id,
                    point_name=point.name,
                    x=components[i, 0],
                    y=components[i, 1],
                    z=components[i, 2],
                    label=arrays.labels[i],
                    answer_type=arrays.answer_types[i],
                )
            )

# %%
df = pd.DataFrame(results)
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="label",
    symbol="answer_type",
    facet_col="llm_id",
    facet_row="point_name",
    category_orders={"color": [False, True]},
    height=2000,
)
fig.update_xaxes(matches=None)
fig.update_yaxes(matches=None)
fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
fig.show()

# %%
point_name = "h30"
results = []
for llm_id in llm_ids[:1]:
    arrays = dataset.get(
        llm_id=llm_id,
        dataset_filter=DatasetIdFilter("geometry_of_truth/cities"),
        split="train",
        point_name=point_name,
        token_idx=0,
        limit=None,
    )
    arrays_val = dataset.get(
        llm_id=llm_id,
        dataset_filter=DatasetIdFilter("geometry_of_truth/cities"),
        split="validation",
        point_name=point_name,
        token_idx=0,
        limit=None,
    )
    assert arrays_val.groups is not None
    for probe_method in tqdm(ALL_PROBES):
        start = datetime.now()
        probe = train_probe(probe_method, arrays)
        duration = datetime.now() - start
        print(probe_method, duration)
        assert probe is not None
        eval_results = eval_probe_by_question(
            probe,
            activations=arrays_val.activations,
            groups=arrays_val.groups,
            labels=arrays_val.labels,
        )
        results.append(
            dict(
                llm_id=llm_id,
                probe_method=probe_method,
                accuracy=eval_results.accuracy,
                duration=duration.total_seconds(),
            )
        )
df = pd.DataFrame(results)
df

# %%
df["is_supervised"] = df["probe_method"].isin(SUPERVISED_PROBES)
px.bar(
    df,
    x="probe_method",
    y="accuracy",
    color="is_supervised",
    facet_col="llm_id",
)
