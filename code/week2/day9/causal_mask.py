import torch
import torch.nn.functional as F

def apply_causal_mask(scores, mask_value=-1e9):
    """
    Args:
        scores: [B, T, T]

    Returns:
        masked_scores: [B, T, T]
    """
    B, T, _ = scores.shape
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    mask_scores = scores.masked_fill(~mask, mask_value)

    return mask_scores

def test():
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 4)
    masked_scores = apply_causal_mask(scores)
    probs = F.softmax(masked_scores, dim=-1)

    assert masked_scores.shape == scores.shape
    assert torch.allclose(masked_scores[:, 1, 0], scores[:, 1, 0])
    assert torch.all(masked_scores[:, 0, 1:] < -1e8)
    assert torch.all(probs[:, 0, 1:] < 1e-6)
    assert not torch.isnan(probs).any()

    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.all(probs[:, future_mask] < 1e-6)

    print("Input shape:", scores.shape)
    print("Output shape:", masked_scores.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()