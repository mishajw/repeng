import torch
from dotenv import load_dotenv

from repeng.hooks.grab import grab
from repeng.models.loading import load_llm_oioo

assert load_dotenv()

device = torch.device("cuda")
llm = load_llm_oioo(
    "gemma-2b",
    device=device,
    use_half_precision=True,
)
with grab(llm.model, llm.points[-3]) as grab_fn:
    tokens = llm.tokenizer.encode("Hello, world!", return_tensors="pt").to(
        device=device
    )
    print(llm.model(tokens))
    print(grab_fn())
