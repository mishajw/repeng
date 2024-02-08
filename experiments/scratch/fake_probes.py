# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jaxtyping import Float
from tqdm import tqdm

from repeng.activations.probe_preparations import ActivationArrays
from repeng.probes.base import DotProductProbe
from repeng.probes.collections import ProbeMethod, train_probe
from repeng.probes.contrast_consistent_search import (
    CcsProbe,
    CcsTrainingConfig,
    train_ccs_probe,
)
from repeng.probes.linear_artificial_tomography import train_lat_probe
from repeng.probes.logistic_regression import LogisticRegressionProbe, train_lr_probe

# %%
anisotropy_offset = np.array([0, 0], dtype=np.float32)
dataset_direction = np.array([0, 2], dtype=np.float32)
dataset_cov = np.array([[0.1, 0.2], [0.2, 1]])
truth_direction = np.array([2, 0])
truth_cov = np.array([[0.01, 0], [0, 0.01]])
num_samples = int(1e3)

random_false = np.random.multivariate_normal(
    mean=anisotropy_offset + dataset_direction, cov=dataset_cov, size=num_samples
)
random_true = random_false + np.random.multivariate_normal(
    mean=truth_direction, cov=truth_cov, size=num_samples
)

df_1 = pd.DataFrame(random_true, columns=["x", "y"])
df_1["label"] = True
df_1["pair_id"] = np.array(range(num_samples))
df_2 = pd.DataFrame(random_false, columns=["x", "y"])
df_2["label"] = False
df_2["pair_id"] = np.array(range(num_samples))
df = pd.concat([df_1, df_2])
df["activations"] = df.apply(lambda row: np.array([row["x"], row["y"]]), axis=1)

arrays = ActivationArrays(
    activations=np.stack(df["activations"]),  # type: ignore
    labels=df["label"].to_numpy(),
    groups=df["pair_id"].to_numpy(),
    answer_types=None,
)

# %%
probe_methods: list[ProbeMethod] = [
    "ccs",
    "lat",
    "dim",
    "lda",
    "lr",
    "lr-g",
    "pca",
    "pca-g",
    "rand",
]
probes = {
    probe_method: train_probe(
        probe_method,
        arrays,
    )
    for probe_method in tqdm(probe_methods)
}

# %%
fig_start = -2
fig_end = 6


def plot_probe(
    label: str,
    fig: go.Figure,
    probe: Float[np.ndarray, "2"],
    intercept: float,
) -> None:
    print(probe, intercept)
    xs = np.array([fig_start, 0, fig_end])
    ys = -(probe[1] / probe[0]) * xs - (intercept / probe[0])
    # TODO: Why swapped?
    fig.add_trace(
        go.Scatter(
            x=ys, y=xs, mode="lines", name=label, line=dict(width=3), opacity=0.6
        )
    )


fig = px.scatter(df, "x", "y", color="label", opacity=0.3)
fig.update_layout(
    xaxis_range=[fig_start, fig_end],
    yaxis_range=[fig_start, fig_end],
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
for probe_method, probe in probes.items():
    assert probe is not None
    if isinstance(probe, DotProductProbe):
        plot_probe(probe_method, fig, probe.probe, 0)
    elif isinstance(probe, LogisticRegressionProbe):
        plot_probe(
            probe_method,
            fig,
            probe.model.coef_[0],
            probe.model.intercept_[0],
        )
    elif isinstance(probe, CcsProbe):
        plot_probe(
            probe_method,
            fig,
            probe.linear.weight.detach().numpy()[0],
            probe.linear.bias.detach().numpy()[0],
        )
    else:
        raise ValueError(type(probe))
fig.show()
