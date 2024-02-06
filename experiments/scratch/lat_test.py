# %%
from dataclasses import dataclass
from pathlib import Path

import seaborn as sns
import torch
from mppr import MContext
from sklearn.decomposition import PCA

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import BinaryRow, DatasetId
from repeng.datasets.elk.utils.collections import get_datasets
from repeng.datasets.elk.utils.limits import limit_dataset_and_split_fn
from repeng.evals.logits import eval_logits_by_question
from repeng.models.loading import load_llm_oioo
from repeng.models.types import LlmId

mcontext = MContext(Path("../output/comparison"))

# %%
# activations_dataset: list[ActivationResultRow] = mcontext.download_cached(
#     "activations_dataset-v3",
#     path=(
#         "s3://repeng/datasets/activations/"
#         # "datasets_2024-02-02_tokensandlayers_v1.pickle"
#         # "datasets_2024-02-03_v1.pickle"
#         "datasets_2024-02-05_v2.pickle"
#     ),
#     to="pickle",
# ).get()
# print(set(row.llm_id for row in activations_dataset))
# print(set(row.dataset_id for row in activations_dataset))
# print(set(row.split for row in activations_dataset))
# dataset = ActivationArrayDataset(activations_dataset)


# %%
@dataclass
class InputSpec:
    row: BinaryRow
    llm_id: LlmId


llm_ids: list[LlmId] = [
    "Llama-2-7b-hf",
    "Llama-2-7b-chat-hf",
]
input_specs = (
    mcontext.create_cached(
        "dataset",
        lambda: get_datasets(["arc_easy"]),
        to=BinaryRow,
    )
    .filter(
        limit_dataset_and_split_fn(train_limit=800, validation_limit=200),
    )
    .flat_map(
        lambda key, row: {
            f"{key}-{llm_id}": InputSpec(row=row, llm_id=llm_id) for llm_id in llm_ids
        }
    )
    .sort(lambda _, row: llm_ids.index(row.llm_id))
)
activations = input_specs.map_cached(
    "test-activations",
    lambda _, row: get_model_activations(
        load_llm_oioo(
            row.llm_id,
            device=torch.device("cuda"),
            use_half_precision=True,
        ),
        text=row.row.text,
        last_n_tokens=1,
    ),
    to="pickle",
)
dataset = ActivationArrayDataset(
    activations.join(
        input_specs,
        lambda _, row, spec: ActivationResultRow(
            llm_id=spec.llm_id,
            dataset_id=spec.row.dataset_id,
            split=spec.row.split,
            group_id=spec.row.group_id,
            template_name=spec.row.template_name,
            answer_type=spec.row.answer_type,
            label=spec.row.is_true,
            activations=row.activations,
            prompt_logprobs=row.token_logprobs.sum(),
        ),
    ).get()
)


# %%
dataset_id: DatasetId = "arc_easy"
for llm_id in llm_ids:
    # for point in get_points(llm_id)[::3]:
    #     print(llm_id, point.name)
    #     arrays = dataset.get(
    #         llm_id=llm_id,
    #         dataset_filter_id=dataset_id,
    #         split="train",
    #         point_name=point.name,
    #         token_idx=0,
    #         limit=None,
    #     )
    #     assert arrays.groups is not None
    #     probe = train_lat_probe(
    #         activations=arrays.activations,
    #         groups=arrays.groups,
    #     )

    #     arrays_val = dataset.get(
    #         llm_id=llm_id,
    #         dataset_filter_id=dataset_id,
    #         split="validation",
    #         point_name=point.name,
    #         token_idx=0,
    #         limit=None,
    #     )
    #     assert arrays_val.groups is not None
    #     print(
    #         eval_probe_by_question(
    #             probe=probe,
    #             activations=arrays_val.activations,
    #             groups=arrays_val.groups,
    #             labels=arrays_val.labels,
    #         )
    #     )

    arrays_val = dataset.get(
        llm_id=llm_id,
        dataset_filter_id=dataset_id,
        split="validation",
        point_name="logprobs",
        token_idx=0,
        limit=None,
    )
    assert arrays_val.groups is not None
    print(
        eval_logits_by_question(
            logits=arrays_val.activations,
            groups=arrays_val.groups,
            labels=arrays_val.labels,
        )
    )

# %%
pca = PCA(2)
component_values = pca.fit_transform(arrays.activations)

# %%
sns.scatterplot(
    x=component_values[:, 0],
    y=component_values[:, 1],
    hue=arrays.labels,
)
