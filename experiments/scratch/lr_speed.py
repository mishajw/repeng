# %%
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from mppr import MContext

from repeng.activations.probe_preparations import ActivationArrayDataset
from repeng.datasets.activations.types import ActivationResultRow
from repeng.datasets.elk.utils.filters import DatasetIdFilter
from repeng.models.points import get_points
from repeng.probes.logistic_regression import train_lr_probe

# %%
activations = np.random.normal(size=(800, 4096))
labels = np.random.binomial(1, 0.25, size=(800)).astype(bool)

# %%
mcontext = MContext(Path("../../output/comparison"))
activation_results_nonchat: list[ActivationResultRow] = mcontext.load(
    "activations_results_nonchat",
    to="pickle",
).get()

# %%
results = []
points = get_points("Llama-2-7b-hf")
points = points[::3]
random.shuffle(points)
for point in points:
    dataset = ActivationArrayDataset(activation_results_nonchat)
    arrays = dataset.get(
        llm_id="Llama-2-7b-hf",
        dataset_filter=DatasetIdFilter("arc_easy"),
        split="train",
        point_name=point.name,
        token_idx=0,
        limit=None,
    )
    activations = arrays.activations
    labels = arrays.labels
    for solver in ["lbfgs", "liblinear", "newton-cg"]:
        start = datetime.now()
        probe = train_lr_probe(activations=activations, labels=labels, solver=solver)
        end = datetime.now()
        results.append(
            dict(
                solver=solver,
                time=end - start,
                point=point.name,
                n_iters=probe.model.n_iter_,
            )
        )
        print(results[-1])

# %%
df = pd.DataFrame(results)
df["point"] = df["point"].apply(lambda a: int(a.lstrip("h")))
df = df.sort_values("point")
df["n_iters"] = df["n_iters"].apply(lambda a: a[0])
px.line(df, x="point", y="time", color="solver").show()
px.line(df, x="point", y="n_iters", color="solver").show()
df
