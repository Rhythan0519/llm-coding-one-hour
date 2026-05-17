import torch

def sinusoidal_position_embedding(position, dim):
    """
    Args:
        positions: [T]
        dim: int
    Returns:
        pos_embs: [T, D]
    """
    freqs = 10000 ** ((torch.arange(0, dim, 2)).float()/dim)
    angles = position[:, None] / freqs[None, :]
    pos_emb = torch.zeros(position.size(0), dim)
    pos_emb[:, 0::2] = torch.sin(angles)
    pos_emb[:, 1::2] = torch.cos(angles[:, :pos_emb[:, 1::2].shape[1]])
    return pos_emb

def test():
    positions = torch.arange(0, 5)
    pos_emb = sinusoidal_position_embedding(positions, 8)

    assert pos_emb.shape == (5, 8)
    assert not torch.allclose(pos_emb[0], pos_emb[1])
    assert not torch.isnan(pos_emb).any()

    print("Input shape:", positions.shape)
    print("Output shape:", pos_emb.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
