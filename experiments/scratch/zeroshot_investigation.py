# %%
from pathlib import Path

import plotly.express as px
import torch
from dotenv import load_dotenv
from mppr import MContext
from transformers import AutoTokenizer

from repeng.activations.inference import get_model_activations
from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.types import BinaryRow
from repeng.datasets.elk.utils.filters import DATASET_FILTER_FNS, DatasetIdFilter
from repeng.datasets.elk.utils.fns import get_dataset
from repeng.datasets.elk.utils.limits import Limits, SplitLimits, limit_groups
from repeng.evals.logits import eval_logits_by_question
from repeng.models.loading import load_llm_oioo

assert load_dotenv(".env")

# %%
llm = load_llm_oioo(
    llm_id="Llama-2-7b-hf",
    device=torch.device("cuda"),
    use_half_precision=False,
)

# %%
dataset = get_dataset("boolq")
# prompt_format = "[INST] {text} [/INST]"
# dataset = {
#     key: value.model_copy(
#         update=dict(
#             text=prompt_format.format(text=value.text),
#         )
#     )
#     for key, value in dataset.items()
# }
list(dataset.values())[0]

# %%
mcontext = MContext(Path("output"))
inputs = (
    mcontext.create(dataset)
    # .filter(
    #     lambda _, row: DATASET_FILTER_FNS["got_cities/pos"].filter(
    #         row.dataset_id, row.answer_type
    #     ),
    # )
    .filter(
        limit_groups(
            Limits(
                default=SplitLimits(train=0, train_hparams=0, validation=200),
                by_dataset={},
            )
        )
    )
)
outputs = inputs.map_cached(
    "activations-v21",
    lambda _, row: get_model_activations(
        llm,
        text=row.text,
        last_n_tokens=1,
        points_start=None,
        points_end=None,
        points_skip=None,
    ),
    to="pickle",
)
results = outputs.join(
    inputs,
    lambda _, activations, row: ActivationResultRow(
        dataset_id=row.dataset_id,
        group_id=row.group_id,
        answer_type=row.answer_type,
        activations={},
        prompt_logprobs=activations.token_logprobs.sum().item(),
        label=row.is_true,
        split=row.split,
        llm_id="Llama-2-7b-hf",
    ),
)
array_dataset = ActivationArrayDataset(results.get())

# %%
arrays = array_dataset.get(
    llm_id="Llama-2-7b-hf",
    dataset_filter=DatasetIdFilter("boolq"),
    split="validation",
    point_name="logprobs",
    token_idx=-1,
    limit=None,
)
assert arrays.groups is not None
eval_logits_by_question(
    logits=arrays.activations,
    labels=arrays.labels,
    groups=arrays.groups,
)

# px.histogram(
#     df,
#     x="logprobs",
#     color="is_true",
#     barmode="overlay",
#     opacity=0.5,
# )

# # %%
# # tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-7b")
# tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-7b-chat-hf")
# text = "test text"
# tokens = tokenizer.encode(text)
# tokens_str = tokenizer.tokenize(text)
# tokens = tokenizer.convert_tokens_to_ids(tokens_str)
# tokens = torch.tensor([tokens])

# tokens_new = tokenizer.encode(text, return_tensors="pt")
# tokens_str_new = tokenizer.convert_ids_to_tokens(tokens_new.squeeze().tolist())

# print(tokens)
# print(tokens_new)

# print(tokenizer.decode(tokens.squeeze()))
# print(tokenizer.decode(tokens_new.squeeze()))

# print(tokens_str)
# print(tokens_str_new)
