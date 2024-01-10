# %%
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import transformer_lens
from ctransformers import AutoModelForCausalLM
from mppr import mppr
from sklearn.decomposition import PCA
from torchtyping import TensorType, patch_typeguard
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from repeng.activations import ActivationRow, get_activations
from repeng.data.true_false import TrueFalseRow, get_true_false_dataset

patch_typeguard()

# %%
layer = "blocks.13.hook_resid_post"
model_tl = transformer_lens.HookedTransformer.from_pretrained(
    # "gpt2",
    "pythia-1b",
    device="cpu",
)
_, cache = model_tl.run_with_cache("test")
print(list(cache.keys()))
assert layer in cache.keys()

# %%
model = AutoModelForCausalLM.from_pretrained(
    # "TheBloke/Llama-2-7b-Chat-GGUF",
    # model_file=str(Path().absolute()),
    "../llama-2-7b-chat.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=0,
)


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
    base_dir=Path("output/repe_repro_pythia-1b"),
    init_fn=get_true_false_dataset,
    to=TrueFalseRow,
).limit(
    1000,
)
df = (
    input.map(
        "format",
        lambda _, row: format_input(row),
        to="pickle",
    )
    .map(
        "activations",
        lambda _, text: get_activations(model=model_tl, text=text, layers=[layer]),
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
            activation=row.activation.activations[layer],
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
n = 100000
indices_1 = np.random.randint(low=0, high=df.shape[0], size=n)
indices_2 = np.random.randint(low=0, high=df.shape[0], size=n)
activation_diffs = activations[indices_1] - activations[indices_2]
activation_diffs = (activation_diffs - np.mean(activation_diffs, axis=0)) / np.std(
    activation_diffs, axis=0
)

pca = PCA(n_components=1)
pca.fit_transform(activation_diffs)
df["pca"] = df["activation"].apply(
    lambda a: (pca.components_ @ a).item(),  # type: ignore
)

sns.histplot(data=df, x="pca", hue="is_true")

# %%
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# assert isinstance(model, GPT2LMHeadModel)

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step3000",
)
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1b",
    revision="step3000",
)
assert isinstance(model, GPTNeoXForCausalLM)


# %%
print(sum(p.numel() for p in model.parameters()))
model


# %%
batch, seq, hidden = None, None, None


def patch_fn(
    tensor: TensorType["batch", "seq", "hidden"],
) -> TensorType["batch", "seq", "hidden"]:
    print(tensor.shape, truth_activation.shape)
    tensor[:, :] -= truth_activation
    return tensor


def generate():
    input = tokenizer.encode(
        (
            "Q: What city is the Eiffel Tower in? A: Paris\n"
            "Q: What city is Westminster in? A: London\n"
            "Q: What city is the Colosseum in? A:"
        ),
        return_tensors="pt",
    )
    assert isinstance(input, torch.Tensor)
    output = model.generate(
        input,
        # tokenizer.encode("3 + 4 = 7\n1 + 3 = 4\n6 + 2 =", return_tensors="pt"),
        # pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id,
        # max_new_tokens=5,
        # temperature=0.7,
        # do_sample=True,
    )
    assert isinstance(output, torch.Tensor)
    output = output.squeeze(0)
    print(repr(tokenizer.decode(output[input.shape[1] :])))


# generate()

x = None
try:
    # x = model.transformer.h[9].register_forward_hook(
    x = model.gpt_neox.layers[13].register_forward_hook(
        lambda _, input, output: (patch_fn(output[0]), output[1])
    )
    generate()
finally:
    if x is not None:
        x.remove()

# %%
