# type: ignore

# %%
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformer_lens
from mppr import mppr
from sklearn.decomposition import PCA

from repeng.activations import ActivationRow, get_activations
from repeng.datasets.amazon import create_amazon_rows
from repeng.datasets.probe import train_probe
from repeng.datasets.types import InputRow

# %%
torch.cuda.empty_cache()

# %%
device = torch.device("cuda")
model = transformer_lens.HookedTransformer.from_pretrained(
    "gptj",
    device=device,
    dtype=torch.bfloat16,  # type: ignore
)
layers = [f"blocks.{i}.hook_resid_post" for i in range(len(model.blocks))]

# %%
with torch.inference_mode():
    _, cache = model.run_with_cache("test")
    list(cache.keys())

# %%

inputs = mppr.init(
    "initial_v1",
    Path("output"),
    init_fn=create_amazon_rows,
    to=InputRow,
).limit(500)
data = inputs.map(
    "activations_v1",
    lambda _, row: get_activations(model=model, text=row.text, layers=layers),
    to="pickle",  # required for numpy arrays.
)


# %%
@dataclass
class ActivationAndInputRow:
    input: InputRow
    activation: ActivationRow


activation_rows: list[ActivationAndInputRow] = inputs.join(
    data, lambda _, i, a: ActivationAndInputRow(input=i, activation=a)
).get()
activation_rows = activation_rows[: len(activation_rows) // 2 * 2]
len(activation_rows)


# %%
@dataclass
class ActivationArrays:
    rows_1: list[InputRow]
    rows_2: list[InputRow]
    activations_1: np.ndarray
    activations_2: np.ndarray
    logprobs_1: np.ndarray
    logprobs_2: np.ndarray
    is_text_true: np.ndarray


def get_activation_arrays(
    rows: list[ActivationAndInputRow],
    layer: str,
    normalize: bool = True,
) -> ActivationArrays:
    rows = sorted(rows, key=lambda r: r.input.does_text_contain_true)
    rows = sorted(rows, key=lambda r: r.input.pair_idx)
    rows_1 = [row for row in rows if row.input.does_text_contain_true]
    rows_2 = [row for row in rows if not row.input.does_text_contain_true]
    assert all(
        r1.input.pair_idx == r2.input.pair_idx
        and r1.input.does_text_contain_true
        and not r2.input.does_text_contain_true
        for r1, r2 in zip(rows_1, rows_2)
    )

    activations_1 = np.stack([row.activation.activations[layer] for row in rows_1])
    activations_2 = np.stack([row.activation.activations[layer] for row in rows_2])
    if normalize:
        activations_1 = (activations_1 - np.mean(activations_1, axis=0)) / np.std(
            activations_1, axis=0
        )
        activations_2 = (activations_2 - np.mean(activations_2, axis=0)) / np.std(
            activations_2, axis=0
        )

    logprobs_1 = np.array(
        [row.activation.token_logprobs.sum().item() for row in rows_1]
    )
    logprobs_2 = np.array(
        [row.activation.token_logprobs.sum().item() for row in rows_2]
    )

    is_text_true = np.array([row.input.is_text_true for row in rows_1])

    return ActivationArrays(
        rows_1=[r.input for r in rows_1],
        rows_2=[r.input for r in rows_2],
        activations_1=activations_1,
        activations_2=activations_2,
        logprobs_1=logprobs_1,
        logprobs_2=logprobs_2,
        is_text_true=is_text_true,
    )


# %%
activation_arrays = get_activation_arrays(
    activation_rows, layer=layers[9], normalize=True
)
lp1 = activation_arrays.logprobs_1 - activation_arrays.logprobs_1.mean()
lp2 = activation_arrays.logprobs_2 - activation_arrays.logprobs_2.mean()
print((lp1 > lp2).mean())
print(((lp1 > lp2) == activation_arrays.is_text_true).mean())

# %%
for layer in layers:
    # for layer in [layers[20]]:
    print("Layer:", layer)
    activation_arrays = get_activation_arrays(
        activation_rows, layer=layer, normalize=True
    )
    probe = train_probe(activation_arrays, device=device)
    device = next(probe.parameters()).device
    pred_is_text_true = (
        probe(torch.tensor(activation_arrays.activations_1, device=device))
        + (1 - probe(torch.tensor(activation_arrays.activations_2, device=device))) / 2
    )
    pred_is_text_true = (pred_is_text_true > 0.5).squeeze(1).cpu().numpy()
    print((pred_is_text_true == activation_arrays.is_text_true).mean())

# %%
pca = PCA(n_components=6)
activation_diffs = activation_arrays.activations_1 - activation_arrays.activations_2
pca_weights = pca.fit_transform(activation_diffs)
for i in range(0, 6, 2):
    axes = sns.scatterplot(
        x=pca_weights[:, i],
        y=pca_weights[:, i + 1],
        hue=activation_arrays.is_text_true,
    )
    axes.set_xlabel(f"PCA Component {i + 1}")
    axes.set_ylabel(f"PCA Component {i + 2}")
    plt.show()

# %%
df = pd.DataFrame(
    [
        {
            "id": row.input.pair_idx,
            "text": row.input.text,
            "is_text_true": row.input.is_text_true,
            "does_text_contain_true": row.input.does_text_contain_true,
            "activation": row.activation.activations,
        }
        for row in activation_rows
    ]
)
mean_contains_true = torch.stack(
    df[df["does_text_contain_true"]]["activation"].to_list()
)
mean_contains_false = torch.stack(
    df[~df["does_text_contain_true"]]["activation"].to_list()
)

# %%
df = pd.DataFrame(
    [
        dict(
            is_text_true=row.input.is_text_true,
            does_text_contain_true=row.input.does_text_contain_true,
            # logprobs=row.token_logprobs.sum(),
            logprobs=row.activation.token_logprobs[-1].item(),
        )
        for row in activation_rows
    ]
)
sns.histplot(data=df, x="logprobs", hue="is_text_true")
plt.show()
sns.histplot(data=df, x="logprobs", hue="does_text_contain_true")
plt.show()
