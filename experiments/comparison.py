# %%
from pathlib import Path

from mppr import mppr

from repeng import models
from repeng.activations import get_activations
from repeng.datasets.collections import get_all_datasets
from repeng.datasets.types import BinaryRow

# %%
model, tokenizer, points = models.gpt2()

# %%
inputs = mppr.init(
    "init-limit-100",
    Path("../output/comparison"),
    init_fn=lambda: get_all_datasets(limit_per_dataset=100),
    to=BinaryRow,
)
print(len(inputs.get()))

# %%
df = (
    inputs.map(
        "activations",
        fn=lambda _, value: get_activations(
            model,
            tokenizer,
            points,
            value.text,
        ),
        to="pickle",
    )
    .join(
        inputs,
        lambda _, activations, input: dict(
            dataset_id=input.dataset_id,
            is_true=input.is_true,
            activations=activations.activations[points[-1].name],
        ),
    )
    .to_dataframe(lambda d: d)
)
df
