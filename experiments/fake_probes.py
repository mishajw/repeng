# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jaxtyping import Float

from repeng.activations.probe_preparations import (
    Activation,
    prepare_activations_for_probes,
)
from repeng.probes.contrast_consistent_search import CcsTrainingConfig, train_ccs_probe
from repeng.probes.linear_artificial_tomography import (
    LatTrainingConfig,
    train_lat_probe,
)
from repeng.probes.logistic_regression import train_lr_probe
from repeng.probes.mean_mass_probe import train_mmp_probe

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
df_1["label"] = "true"
df_1["pair_id"] = np.array(range(num_samples))
df_2 = pd.DataFrame(random_false, columns=["x", "y"])
df_2["label"] = "false"
df_2["pair_id"] = np.array(range(num_samples))
df = pd.concat([df_1, df_2])
df["activations"] = df.apply(lambda row: np.array([row["x"], row["y"]]), axis=1)

# %%
activations = prepare_activations_for_probes(
    [
        Activation(
            dataset_id="test",
            pair_id=row["pair_id"],
            activations=row["activations"],
            label=row["label"] == "true",
        )
        for _, row in df.iterrows()
    ]
)
lat_probe = train_lat_probe(
    activations.activations, LatTrainingConfig(num_random_pairs=1000)
)
lr_probe = train_lr_probe(activations.labeled)
mmp_probe = train_mmp_probe(activations.labeled, use_iid=False)
mmp_iid_probe = train_mmp_probe(activations.labeled, use_iid=True)
ccs_probe = train_ccs_probe(activations.paired, CcsTrainingConfig(num_steps=1000))

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
plot_probe("lat", fig, lat_probe.probe, 0)
plot_probe("lr", fig, lr_probe.model.coef_[0], lr_probe.model.intercept_[0])
plot_probe("mmp", fig, mmp_probe.probe, 0)
plot_probe("mmp-iid", fig, mmp_iid_probe.probe, 0)
plot_probe(
    "ccs",
    fig,
    ccs_probe.linear.weight.detach().numpy()[0],
    ccs_probe.linear.bias.detach().numpy()[0],
)
fig.show()
