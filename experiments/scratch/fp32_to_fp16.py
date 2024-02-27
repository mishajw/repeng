# %%
import pickle
from itertools import count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from repeng.datasets.activations.types import ActivationResultRow

# %%
path = Path("../../output/comparison/activations_results.pickle")
output_path = Path("../../output/comparison/activations_results_fp16.pickle")
assert path.exists()

# %%
with output_path.open("wb") as output_f:
    with path.open("rb") as input_f:
        for _ in tqdm(count()):
            try:
                result = pickle.load(input_f)
            except EOFError:
                break
            assert isinstance(result, dict)
            assert result.keys() == {"key", "value"}
            assert isinstance(result["key"], str)
            row: ActivationResultRow = result["value"]
            row.activations = {
                k: v.astype(np.float16) for k, v in row.activations.items()
            }
            assert set(value.dtype for value in row.activations.values()) == {
                np.dtype("float16")
            }
            pickle.dump(result, output_f)
            # yield result["key"], result["value"]
