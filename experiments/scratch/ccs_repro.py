# %%
from pathlib import Path

import torch
from mppr import MContext

from repeng.datasets.elk.utils.collections import get_datasets
from repeng.models.llms import get_llm

# %%
llm = get_llm("gpt2", device=torch.device("cuda"), dtype=torch.bfloat16)

# %%
mcontext = MContext(Path("../output/ccs_repro"))
dataset = mcontext.create_cached(
    "dataset",
    lambda: get_datasets(["imdb"]),
    to="pickle",
)

# %%
