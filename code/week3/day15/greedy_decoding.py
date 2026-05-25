import torch
import torch.nn.functional as F

def greedy_decode(logits):
    """
    Args:
        logits: [B, V] or [B, T, V]

    Returns:
        next_ids: [B]
    """

    last_logits = logits
    if logits.ndim == 3:
        last_logits = logits[:, -1, :]

    next_ids = torch.argmax(last_logits, dim=-1)

    return next_ids

def test():
    logits_2d = torch.tensor([
        [0.1, 2.0, 0.3],
        [3.0, 1.0, 2.0],
    ])
    out_2d = greedy_decode(logits_2d)

    logits_3d = torch.tensor([
        [[9.0, 0.0, 0.0], [0.1, 0.2, 5.0]],
        [[0.0, 7.0, 0.0], [4.0, 3.0, 2.0]],
    ])
    out_3d = greedy_decode(logits_3d)

    assert out_2d.shape == (2,)
    assert out_3d.shape == (2,)
    assert torch.equal(out_2d, torch.tensor([1, 0]))
    assert torch.equal(out_3d, torch.tensor([2, 0]))
    assert out_2d.dtype == torch.long

    print("2D input shape:", logits_2d.shape)
    print("3D input shape:", logits_3d.shape)
    print("Output shape:", out_3d.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()