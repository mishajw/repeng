# %%
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import transformer_lens
from mppr import mppr
from sklearn.decomposition import PCA

from ccs.activations import ActivationRow, get_activation_arrays, get_activations
from ccs.data.amazon import create_amazon_rows
from ccs.data.probe import train_probe
from ccs.data.types import InputRow

# %%
torch.cuda.empty_cache()

# %%
# model = "meta-llama/Llama-2-7b-chat-hf"
# hf_model = LlamaForCausalLM.from_pretrained(model)
# tokenizer = LlamaTokenizer.from_pretrained(model)

device = torch.device("cuda")
model = transformer_lens.HookedTransformer.from_pretrained(
    # "stabilityai/stablelm-tuned-alpha-7b",
    "gptj",
    device=device,
    dtype=torch.bfloat16,
)
layers = [f"blocks.{i}.hook_resid_post" for i in range(len(model.blocks))]

# %%
with torch.inference_mode():
    _, cache = model.run_with_cache("test")
    list(cache.keys())

# %%
activation_rows_v1 = (
    mppr.init(
        "initial_v1",
        Path("output"),
        init_fn=create_amazon_rows,
        to=InputRow,
    )
    .limit(500)
    .map(
        "activations_v1",
        lambda _, row: get_activations(model=model, input_row=row, layers=layers),
        to="pickle",  # required for numpy arrays.
    )
    .get()
)
# activation_rows_v2 = (
#     mppr.init(
#         "initial_v2",
#         Path("output"),
#         init_fn=create_addition_rows_v2,
#         to=InputRow,
#     )
#     .limit(500)
#     .map(
#         "activations_v2",
#         lambda _, row: get_activations(model=model, input_row=row, layers=layers),
#         to="pickle",  # required for numpy arrays.
#     )
#     .get()
# )
activation_rows = activation_rows_v1

# %%
activation_rows: list[ActivationRow] = mppr.load(
    "activations_v1",
    Path("output"),
    to="pickle",
).get()
activation_rows = activation_rows[: len(activation_rows) // 2 * 2]
len(activation_rows)

# %%
activation_arrays = get_activation_arrays(
    activation_rows, layer=layers[9], normalize=True
)
lp1 = activation_arrays.logprobs_1 - activation_arrays.logprobs_1.mean()
lp2 = activation_arrays.logprobs_2 - activation_arrays.logprobs_2.mean()
print((lp1 > lp2).mean())
print(((lp1 > lp2) == activation_arrays.is_text_true).mean())

# %%
for i in range(5):
    print(repr(activation_arrays.rows_1[i].input_row.text))
    print(activation_arrays.rows_1[i].token_logprobs[-1])
    print(repr(activation_arrays.rows_2[i].input_row.text))
    print(activation_arrays.rows_2[i].token_logprobs[-1])
    print()

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
print(activation_arrays.rows_1[0].input_row)
print(activation_arrays.rows_2[0].input_row)
print(activation_arrays.rows_1[1].input_row)
print(activation_arrays.rows_2[1].input_row)
print(activation_arrays.rows_1[2].input_row)
print(activation_arrays.rows_2[2].input_row)

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

# %%
df = pd.DataFrame(
    [
        dict(
            is_text_true=row.input_row.is_text_true,
            does_text_contain_true=row.input_row.does_text_contain_true,
            # logprobs=row.token_logprobs.sum(),
            logprobs=row.token_logprobs[-1].item(),
        )
        for row in activation_rows
    ]
)
sns.histplot(data=df, x="logprobs", hue="is_text_true")
plt.show()
sns.histplot(data=df, x="logprobs", hue="does_text_contain_true")
plt.show()

# %%
pprint(activation_rows[0].input_row.model_dump())
pprint(activation_rows[0].token_logprobs.sum())
pprint(activation_rows[1].input_row.model_dump())
pprint(activation_rows[1].token_logprobs.sum())
pprint(activation_rows[2].input_row.model_dump())
pprint(activation_rows[2].token_logprobs.sum())
pprint(activation_rows[3].input_row.model_dump())
pprint(activation_rows[3].token_logprobs.sum())
