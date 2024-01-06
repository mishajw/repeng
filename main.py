# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import transformer_lens
from mppr import mppr
from sklearn.decomposition import PCA
from tqdm import tqdm

from ccs.activations import ActivationRow, get_activation_arrays, get_activations
from ccs.data.addition import create_addition_rows_v2
from ccs.data.types import InputRow

# %%
model = transformer_lens.HookedTransformer.from_pretrained(
    "EleutherAI/pythia-1b",
    device="mps",
)
layers = [f"blocks.{i}.hook_resid_post" for i in range(0, 15)]

# %%
# activation_rows_v1 = (
#     mppr.init(
#         "initial_v1",
#         Path("output"),
#         init_fn=create_addition_rows,
#         to=InputRow,
#     )
#     .map(
#         "activations_v1",
#         lambda _, row: get_activations(row),
#         to="pickle",  # required for numpy arrays.
#     )
#     .get()
# )
activation_rows_v2 = (
    mppr.init(
        "initial_v2",
        Path("output"),
        init_fn=create_addition_rows_v2,
        to=InputRow,
    )
    # .limit(1e9)
    .map(
        "activations_v2",
        lambda _, row: get_activations(model=model, input_row=row, layers=layers),
        to="pickle",  # required for numpy arrays.
    ).get()
)
activation_rows = activation_rows_v2

# %%
activation_rows: list[ActivationRow] = mppr.load(
    "activations_v2",
    Path("output"),
    to="pickle",
).get()
len(activation_rows)

# %%
activation_arrays = get_activation_arrays(
    activation_rows, layer=layers[9], normalize=False
)
print(
    (
        (activation_arrays.logprobs_1 > activation_arrays.logprobs_2)
        == activation_arrays.is_text_true
    ).mean()
)

# %%
probe = torch.nn.Sequential(
    torch.nn.Linear(2048, 1),
    torch.nn.Sigmoid(),
).to("mps")
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-3)
activations1_tensor = torch.tensor(activation_arrays.activations_1).to(device="mps")
activations2_tensor = torch.tensor(activation_arrays.activations_2).to(device="mps")
pbar = tqdm(range(10000))
for _ in pbar:
    p1 = probe(activations1_tensor)
    p2 = probe(activations2_tensor)
    loss_consistency = ((p1 - (1 - p2)) ** 2).mean()
    loss_confidence = (torch.stack([p1, p2], dim=1).min(dim=1).values ** 2).mean()
    loss = loss_consistency + loss_confidence
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    pbar.set_postfix(loss=loss.item())

# %%
pred_is_text_true = (probe(activations1_tensor) > 0.5).squeeze(1).cpu().numpy()
(pred_is_text_true == activation_arrays.is_text_true).mean()

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
            "id": row.input_row.pair_idx,
            "text": row.input_row.text,
            "is_text_true": row.input_row.is_text_true,
            "does_text_contain_true": row.input_row.does_text_contain_true,
            "activation": row.activations,
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
# df[df["does_text_contain_true"]]["activation"]
