import torch
from tqdm import tqdm

from repeng.activations import ActivationArrays


def train_probe(
    activation_arrays: ActivationArrays, device: torch.device
) -> torch.nn.Module:
    hidden_dim = activation_arrays.activations_1.shape[1]
    probe = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, 1, bias=True),
        torch.nn.Sigmoid(),
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    activations1_tensor = torch.tensor(activation_arrays.activations_1).to(
        device=device
    )
    activations2_tensor = torch.tensor(activation_arrays.activations_2).to(
        device=device
    )
    pbar = tqdm(range(1000))
    for _ in pbar:
        p1 = probe(activations1_tensor)
        p2 = probe(activations2_tensor)
        loss_consistency = ((p1 - (1 - p2)) ** 2).mean()
        loss_confidence = (torch.concat([p1, p2], dim=1).min(dim=1).values ** 2).mean()
        loss_confidence = 0
        loss = loss_consistency + loss_confidence
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())
    return probe
