# %%
from pathlib import Path
import matplotlib.pyplot as plt
import random
from typing import Any
import datasets
import numpy as np
from pydantic import BaseModel
import transformer_lens
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from mppr import mppr

LAYERS = [f"blocks.{i}.hook_resid_post" for i in range(0, 15)]

# %%
model = transformer_lens.HookedTransformer.from_pretrained("EleutherAI/pythia-1b").to(
    "mps"
)

# %%
_, cache = model.run_with_cache("test")
list(cache.keys())


# %%
class InputRow(BaseModel, extra="forbid"):
    pair_idx: str
    text: str
    is_text_true: bool
    does_text_contain_true: bool


class ActivationRow(BaseModel, extra="forbid"):
    input_row: InputRow
    activations: dict[str, np.ndarray]
    token_logprobs: np.ndarray

    class Config:
        arbitrary_types_allowed = True


# %%
def create_boolq_rows() -> list[InputRow]:
    boolq: Any = datasets.load_dataset("boolq")
    rows = []
    for i, row in enumerate(boolq["train"]):
        passage = row["passage"]
        question = row["question"]
        text = f"{passage}\n\n{question}?"
        rows.append(
            InputRow(
                pair_idx=str(i),
                text=f"{text}\nYes",
                is_text_true=row["answer"],
                does_text_contain_true=True,
            )
        )
        rows.append(
            InputRow(
                pair_idx=str(i),
                text=f"{text}\nNo",
                is_text_true=not row["answer"],
                does_text_contain_true=False,
            )
        )
    return rows


def create_addition_rows() -> dict[str, InputRow]:
    rows = {}
    for i in range(1000):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        c = a + b + random.randint(-1, 1)
        rows[f"{i}_true"] = InputRow(
            pair_idx=str(i),
            text=f"Does {a} + {b} = {c}? Yes",
            is_text_true=a + b == c,
            does_text_contain_true=True,
        )
        rows[f"{i}_false"] = InputRow(
            pair_idx=str(i),
            text=f"Does {a} + {b} = {c}? No",
            is_text_true=a + b != c,
            does_text_contain_true=False,
        )
    return rows


def create_addition_rows_v2() -> dict[str, InputRow]:
    rows = {}
    for i in range(1000):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        c = a + b
        error = 1 if random.random() < 0.5 else -1
        if random.random() < 0.5:
            c = c + error
            error = 0
        rows[f"{i}_a"] = InputRow(
            pair_idx=str(i),
            text=f"{a} + {b} = {c}",
            is_text_true=a + b == c,
            does_text_contain_true=True,
        )
        rows[f"{i}_b"] = InputRow(
            pair_idx=str(i),
            text=f"{a} + {b} = {c + error}",
            is_text_true=a + b == c + error,
            does_text_contain_true=False,
        )
    return rows


def add_fewshot_addition_prefix(row: InputRow) -> InputRow:
    return InputRow(
        pair_idx=row.pair_idx,
        text=f"4 + 2 = 6\n{row.text}",
        is_text_true=row.is_text_true,
        does_text_contain_true=row.does_text_contain_true,
    )


def get_activations(input_row: InputRow) -> ActivationRow:
    logits, cache = model.run_with_cache(input_row.text, names_filter=LAYERS)

    tokens = model.tokenizer.encode(input_row.text, return_tensors="pt").to("mps")
    logprobs = logits[0, : tokens.shape[1], :].log_softmax(dim=-1)
    token_logprobs = logprobs.gather(dim=-1, index=tokens).squeeze(0).detach()

    activations = {
        layer: cache[layer].squeeze(0)[-1].detach().cpu().numpy() for layer in LAYERS
    }

    return ActivationRow(
        input_row=input_row,
        activations=activations,
        token_logprobs=token_logprobs.cpu().numpy(),
    )


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
    .limit(100)
    .map(
        "fewshot_v2",
        lambda _, row: add_fewshot_addition_prefix(row),
        to=InputRow,
    )
    .map(
        "activations_v2",
        lambda _, row: get_activations(row),
        to="pickle",  # required for numpy arrays.
    )
    .get()
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
activation_rows = sorted(
    activation_rows, key=lambda r: r.input_row.does_text_contain_true
)
activation_rows = sorted(activation_rows, key=lambda r: r.input_row.pair_idx)
rows1 = [row for row in activation_rows if row.input_row.does_text_contain_true]
rows2 = [row for row in activation_rows if not row.input_row.does_text_contain_true]
assert all(
    r1.input_row.pair_idx == r2.input_row.pair_idx
    and r1.input_row.does_text_contain_true
    and not r2.input_row.does_text_contain_true
    for r1, r2 in zip(rows1, rows2)
)

activations1 = np.stack([row.activations[LAYERS[9]] for row in rows1])
activations2 = np.stack([row.activations[LAYERS[9]] for row in rows2])
# activations1_norm = (activations1 - np.mean(activations1, axis=0)) / np.std(
#     activations1, axis=0
# )
# activations2_norm = (activations2 - np.mean(activations2, axis=0)) / np.std(
#     activations2, axis=0
# )
# activation_diffs = activations1_norm - activations2_norm
activation_diffs = activations1 - activations2


logprobs1 = np.array([row.token_logprobs.sum().item() for row in rows1])
logprobs2 = np.array([row.token_logprobs.sum().item() for row in rows2])
logprob_diffs = logprobs1 - logprobs2

is_text_true = np.array([row.input_row.is_text_true for row in rows1])
((logprob_diffs > 0) == is_text_true).mean()

# %%
print(rows1[0].input_row.text)
print(rows2[0].input_row.text)
print(rows1[0].token_logprobs)
print(rows2[0].token_logprobs)
print()
print(rows1[1].input_row.text)
print(rows2[1].input_row.text)
print(rows1[1].token_logprobs)
print(rows2[1].token_logprobs)

# %%
probe = torch.nn.Sequential(
    torch.nn.Linear(2048, 1),
    torch.nn.Sigmoid(),
).to("mps")
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-3)
activations1_tensor = torch.tensor(activations1).to(device="mps")
activations2_tensor = torch.tensor(activations2).to(device="mps")
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
(pred_is_text_true == is_text_true).mean()

# %%
pca = PCA(n_components=6)
pca_weights = pca.fit_transform(activation_diffs)
is_text_true = [row.input_row.is_text_true for row in rows1]
errors = [
    int(row2.input_row.text.split()[-1]) - int(row1.input_row.text.split()[-1])
    for row1, row2 in zip(rows1, rows2)
]
for i in range(0, 6, 2):
    axes = sns.scatterplot(x=pca_weights[:, i], y=pca_weights[:, i + 1], hue=errors)
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
