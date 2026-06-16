import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Args:
        timesteps: [B]
        dim: int
    Returns:
        emb: [B, dim]
    """
    assert timesteps.dim() == 1
    assert dim > 0

    num_sin = (dim + 1) // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(num_sin, device=timesteps.device).float()
        / num_sin
    )
    angles = timesteps.float()[:, None] * freqs[None, :]

    emb = torch.zeros(timesteps.size(0), dim, device=timesteps.device)
    emb[:, 0::2] = torch.sin(angles)
    emb[:, 1::2] = torch.cos(angles[:, :emb[:, 1::2].shape[1]])
    return emb


class TimestepMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, timesteps):
        emb = sinusoidal_timestep_embedding(timesteps, self.embedding_dim)
        h = self.fc1(emb)
        h = F.silu(h)
        h = self.fc2(h)
        return h


def test():
    torch.manual_seed(0)
    timesteps = torch.tensor([0, 1, 10, 10], dtype=torch.long)

    emb = sinusoidal_timestep_embedding(timesteps, dim=8)
    assert emb.shape == (4, 8)
    assert torch.allclose(emb[2], emb[3])
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.isnan(emb).any()

    mlp = TimestepMLP(embedding_dim=8, hidden_size=16)
    cond = mlp(timesteps)
    assert cond.shape == (4, 16)

    loss = cond.mean()
    loss.backward()
    assert all(p.grad is not None for p in mlp.parameters())

    print("Input shape:", timesteps.shape)
    print("Embedding shape:", emb.shape)
    print("Condition shape:", cond.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
