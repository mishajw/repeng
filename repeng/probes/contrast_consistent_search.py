"""
Replication of the CCS probes described in <https://arxiv.org/abs/2212.03827>.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int64
from typing_extensions import override

from repeng.probes.base import BaseProbe, PredictResult
from repeng.probes.normalization import normalize_by_group


class CcsProbe(torch.nn.Module, BaseProbe):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self,
        activations: Float[torch.Tensor, "n d"],  # noqa: F722
    ) -> Float[torch.Tensor, "n"]:  # noqa: F821
        result = self.linear(activations)
        result = torch.nn.functional.sigmoid(result)
        result = result.squeeze(-1)
        return result

    @torch.inference_mode()
    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        probabilities = self(torch.tensor(activations, dtype=torch.float32)).numpy()
        return PredictResult(logits=probabilities)


@dataclass
class CcsTrainingConfig:
    num_steps: int = 100
    lr: float = 0.001


def train_ccs_probe(
    config: CcsTrainingConfig,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    groups: Int64[np.ndarray, "n d"],  # noqa: F722
    answer_types: Int64[np.ndarray, "n d"] | None,  # noqa: F722
    # Although CCS is technically unsupervised, we need the labels for multiple-choice
    # questions so that we can reduce answers into a true/false pair.
    labels: Bool[np.ndarray, "n"],  # noqa: F821
    attempts: int = 2,
) -> CcsProbe:
    if answer_types is not None:
        activations = normalize_by_group(activations, answer_types)

    activations_1 = []
    activations_2 = []
    for group in np.unique(groups):
        group_activations = activations[groups == group]
        group_labels = labels[groups == group]
        if True not in group_labels or False not in group_labels:
            # This can happen when we truncate the dataset along a question boundary.
            continue
        # Get the first true and first false rows.
        indices = sorted(
            [group_labels.tolist().index(True), group_labels.tolist().index(False)]
        )
        activations_1.append(group_activations[indices[0]])
        activations_2.append(group_activations[indices[1]])
    activations_1 = torch.tensor(np.array(activations_1)).to(dtype=torch.float32)
    activations_2 = torch.tensor(np.array(activations_2)).to(dtype=torch.float32)

    _, hidden_dim = activations_1.shape
    probe = CcsProbe(hidden_dim=hidden_dim)
    optimizer = torch.optim.LBFGS(probe.parameters(), lr=0.1, max_iter=100)

    def get_loss():
        optimizer.zero_grad()
        probs_1: torch.Tensor = probe(activations_1)
        probs_2: torch.Tensor = probe(activations_2)
        loss_consistency = (probs_1 - (1 - probs_2)).pow(2).mean()
        loss_confidence = torch.min(probs_1, probs_2).pow(2).mean()
        loss = loss_consistency + loss_confidence
        loss.backward()
        return loss.item()

    optimizer.step(closure=get_loss)
    loss = get_loss()
    if loss > 0.2:
        if attempts > 0:
            return train_ccs_probe(
                config=config,
                activations=activations,
                groups=groups,
                answer_types=answer_types,
                labels=labels,
                attempts=attempts - 1,
            )
        raise ValueError(f"CCS probe did not converge: {loss} >= 0.2")
    return probe.eval()
