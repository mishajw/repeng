# %%
from datetime import datetime

import accelerate
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM, PreTrainedTokenizerFast

# %%
model_name = "EleutherAI/pythia-12b"
device = torch.device("cuda")
dtype = torch.bfloat16
model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=dtype,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
assert isinstance(model, GPTNeoXForCausalLM)
assert isinstance(tokenizer, PreTrainedTokenizerFast)
torch.cuda.empty_cache()

# %%
n_samples = 50
start_time = datetime.now()
for _ in tqdm(range(n_samples)):
    tokens = tokenizer.encode("Hello world" * 100, return_tensors="pt").to(device)
    model(tokens)
time = datetime.now() - start_time
print(time.total_seconds() / n_samples, "seconds per sample")

# %%
accelerate.cpu_offload(model=model)

# %%
layers = list(range(35, 36))
print("offloading", layers)
for layer in layers:
    try:
        accelerate.cpu_offload(model.gpt_neox.layers[layer])
        print("offloaded", layer)
    except NotImplementedError:
        pass

# %%
torch.cuda.empty_cache()

# %%
try:
    model.to(device=torch.device("cpu"))
except NotImplementedError:
    pass
del model
torch.cuda.empty_cache()

# %%
model

# %%
# Experiment results:
# - pythia-12b, no offloading:    0.05s        23.75GiB
# - pythia-12b, offloading 35:    0.16s (3x)   23.14GiB
# - pythia-12b, offloading 34-35: 0.28s (6x)   22.00GiB
