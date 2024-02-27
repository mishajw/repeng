# %%
from itertools import islice
from typing import Any, cast

import torch
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import tqdm

from repeng.models.llms import pythia

# %%
model = LanguageModel("EleutherAI/pythia-70m")
dataset: Any = load_dataset("NeelNanda/pile-10k")

# %%
N_HIDDEN = 512
NUM_ROWS = 200
SEQ = 1024
BATCH_SIZE = 8
seqs = []
for row in islice(dataset["train"], 0, NUM_ROWS):
    prompt = row["text"]
    tokens = cast(torch.Tensor, model.tokenizer.encode(prompt, return_tensors="pt"))
    tokens = tokens.squeeze(0)
    for i in range(0, len(tokens), SEQ):
        seqs.append(
            torch.nn.functional.pad(
                tokens[i : i + SEQ], (0, SEQ - len(tokens[i : i + SEQ])), value=0
            )
        )
tokens = torch.stack(seqs)
tokens = tokens[: BATCH_SIZE * (len(tokens) // BATCH_SIZE)]
tokens = tokens.reshape(-1, BATCH_SIZE, SEQ)
tokens.shape

# %%
activations = []
for batch in tqdm(tokens[:2]):
    with model.invoke(tokens[0]) as invoker:
        batch_activations = model.gpt_neox.layers[3].mlp.input[0][0].save()
    activations.append(batch_activations)
activations = torch.concat(activations, dim=0).reshape(-1, N_HIDDEN)

# %%
layer = model.meta_model.gpt_neox.layers[3].mlp
new_direction = torch.nn.Parameter(
    torch.rand((N_HIDDEN), requires_grad=True),
)
optimizer = torch.optim.Adam([new_direction])

pbar = tqdm(range(10_000))
for _ in pbar:
    optimizer.zero_grad()
    new_direction_norm = new_direction / new_direction.norm()

    output = model(activations)  # (N_SAMPLES, N_HIDDEN)
    output_mod = model(activations + new_direction_norm)  # (N_SAMPLES, N_HIDDEN)
    similarities = output @ output_mod.T
    loss = -similarities.mean()
    loss.backward()
    optimizer.step()
    pbar.set_postfix(
        loss=loss.item(),
        min=similarities.min().item(),
        max=similarities.max().item(),
    )
