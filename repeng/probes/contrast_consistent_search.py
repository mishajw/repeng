"""
Replication of the CCS probes described in <https://arxiv.org/abs/2212.03827>.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float
from tqdm import tqdm

from repeng.activations.probe_preparations import PairedActivationArray
from repeng.probes.base import BaseProbe


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
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> Bool[np.ndarray, "n"]:  # noqa: F821
        return (self(torch.tensor(activations)) > 0.5).numpy()


@dataclass
class CcsTrainingConfig:
    num_steps: int = 10
    lr: float = 0.01
    normalize: bool = True


def train_ccs_probe(
    activations: PairedActivationArray,
    config: CcsTrainingConfig,
) -> CcsProbe:
    _, hidden_dim = activations.activations_1.shape
    probe = CcsProbe(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.lr)
    activations_1 = torch.tensor(activations.activations_1)
    activations_2 = torch.tensor(activations.activations_2)

    bar = tqdm(range(config.num_steps))
    for _ in bar:
        probs_1: torch.Tensor = probe(activations_1)
        probs_2: torch.Tensor = probe(activations_2)
        loss_consistency = (probs_1 - (1 - probs_2)).pow(2).mean()
        loss_confidence = torch.min(probs_1, probs_2).pow(2).mean()
        loss = loss_consistency + loss_confidence
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.set_postfix(loss=loss.item())
    return probe.eval()
