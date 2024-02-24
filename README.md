# Representation Engineering

Experiments with representation engineering. There's been a bunch of recent work ([1](https://arxiv.org/abs/2310.01405), [2](https://arxiv.org/abs/2308.10248), [3](https://arxiv.org/abs/2212.03827)) into using a neural network's latent representations to control & interpret models.

This repository contains utilities for running experiments (the `repeng` package) and a bunch of experiments (the notebooks in `experiments`).

## Installation
```bash
git clone https://github.com/mishajw/repeng
cd repeng
pip install -e .
# Or if using poetry:
poetry install
```

## Reproducing experiments

### How well do truth probes generalise?
[Report](https://docs.google.com/document/d/1tz-JulAUz3SOc8Qm8MLwE9TohX8gSZlXQ2Y4PBwfJ1U).

1. Install the repository, as described [above](#installation).
2. Optional: Check out `c99e9aa`. This shouldn't be necessary, unless I introduce breaking changes.
3. Create a dataset of activations: `python experiments/comparison_dataset.py`.
    - This will upload the experiments to S3. Some tinkering may be required to change the upload location - sorry about that!
4. Run the analysis: `python experiments/comparison.py`.
    - This will write plots to `./output/comparison`.

This is split into two scripts as only the first requires a GPU for LLM inference.
