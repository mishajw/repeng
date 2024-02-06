# type: ignore

# %%
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mppr import mppr
from sklearn.decomposition import PCA
from torchtyping import TensorType

from repeng import models
from repeng.activations import ActivationRow, get_activations
from repeng.datasets.elk.true_false import TrueFalseRow, get_true_false_dataset
from repeng.hooks.grab import grab
from repeng.hooks.patch import patch

# %%
device = torch.device("cuda")

# %%
# model, tokenizer, points = models.gpt2()
# model, tokenizer, points = models.pythia("1b")
model, tokenizer, points = models.llama2_13b("7b", chat=True)

# %%
print(set(p.device for p in model.parameters()))
print(set(p.dtype for p in model.parameters()))

# %%
# model = model.to(dtype=torch.bfloat16)
model = model.to(device=device)


# %%
torch.set_grad_enabled(False)

# %%
with grab(model, points[-1]) as get_activation:
    model.forward(torch.tensor([[0]], device=device))
    print(get_activation().shape)


# %%
@dataclass
class Row:
    input: TrueFalseRow
    activation: ActivationRow


def format_input(row: TrueFalseRow) -> str:
    role = "an honest" if row.is_true else "a dishonest"
    return (
        f"USER: Pretend you're {role} person making statements about the world.\n"
        f"ASSISTANT: {row.statement}"
    )


input = mppr.init(
    "initial",
    base_dir=Path("output/repe_repro_llama2_7b"),
    init_fn=get_true_false_dataset,
    to=TrueFalseRow,
).limit(
    500,
)

df = (
    input.map(
        "format",
        lambda _, row: format_input(row),
        to="pickle",
    )
    .map(
        "activations",
        lambda _, text: get_activations(
            model=model,
            tokenizer=tokenizer,
            points=points,
            text=text,
        ),
        to="pickle",
    )
    .join(
        input,
        lambda _, activation, input: Row(input, activation),
    )
    .to_dataframe(
        lambda row: dict(
            statement=row.input.statement,
            is_true=row.input.is_true,
            activation=row.activation.activations[points[-1].name],
            logprobs=row.activation.token_logprobs,
        )
    )
)

# %%
activations_truth = np.mean(df[df["is_true"]]["activation"].tolist(), axis=0)
activations_falsehoods = np.mean(df[~df["is_true"]]["activation"].tolist(), axis=0)
truth_activation = torch.tensor(activations_truth - activations_falsehoods)
print(
    np.linalg.norm(activations_truth),
    np.linalg.norm(activations_falsehoods),
    truth_activation.norm().item(),
)

# %%
activations = np.stack(df["activation"].tolist())
num_pairs = 100_000
indices_1 = np.random.randint(low=0, high=df.shape[0], size=num_pairs)
indices_2 = np.random.randint(low=0, high=df.shape[0], size=num_pairs)
activation_diffs = activations[indices_1] - activations[indices_2]
activation_diffs = (activation_diffs - np.mean(activation_diffs, axis=0)) / np.std(
    activation_diffs, axis=0
)

pca = PCA(n_components=4)
pca.fit_transform(activation_diffs)
for i in range(4):
    df[f"pca_{i}"] = df["activation"].apply(
        lambda a: (pca.components_[i] @ a).item(),  # type: ignore
    )

# %%
sns.scatterplot(data=df, x="pca_0", y="pca_1", hue="is_true")
plt.show()
sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="is_true")
plt.show()
sns.scatterplot(data=df, x="pca_2", y="pca_3", hue="is_true")
plt.show()

# %%
batch, seq, hidden = None, None, None


def patch_fn(
    tensor: TensorType["batch", "seq", "hidden"],
) -> TensorType["batch", "seq", "hidden"]:
    # tensor[:, :] -= truth_activation.to(device=device)
    tensor[:, :] -= torch.tensor(pca.components_[0], device=device)
    return tensor


def generate():
    input = tokenizer.encode(
        (
            # "Q: What city is the Eiffel Tower in? A: Paris\n"
            # "Q: What city is Westminster in? A: London\n"
            # "Q: What city is the Colosseum in? A:"
            # "Q: What's the capital of France? A: Paris\n"
            # "Q: What's the capital of Germany? A: Berlin\n"
            "Q: What's the capital of Spain? A:"
        ),
        return_tensors="pt",
    ).to(device=device)
    assert isinstance(input, torch.Tensor)
    output = model.generate(input, max_new_tokens=5)
    assert isinstance(output, torch.Tensor)
    output = output.squeeze(0)
    print(repr(tokenizer.decode(output[input.shape[1] :])))


generate()
with patch(model, points[-1], patch_fn):
    generate()
