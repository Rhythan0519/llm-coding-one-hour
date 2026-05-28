import torch
import torch.nn.functional as F

def top_k_filter(logits, k, mask_value=-1e9):
    """
    Args:
        logits: [B, V]
        k: int

    Returns:
        filtered_logits: [B, V]
    """
    B, V = logits.shape
    assert 1 <= k <= V
    top_values, top_indices = logits.topk(k=k, dim=-1)
    threshold = top_values[:, -1].unsqueeze(-1)
    # threshold = top_values[:, -1:]
    mask = logits < threshold
    filtered_logits = logits.masked_fill(mask, mask_value)
    return filtered_logits

def sample_top_k(logits, k, temperature=1.0):
    """
    Args:
        logits: [B, V]

    Returns:
        next_ids: [B]
        probs: [B, V]
    """
    assert temperature > 0
    filtered_logits = top_k_filter(logits, k)
    probs = F.softmax(filtered_logits/temperature, dim=-1)
    next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_ids, probs

def test():
    torch.manual_seed(0)
    logits = torch.tensor([
        [0.1, 5.0, 1.0, 4.0, 0.0],
        [3.0, 0.2, 2.0, 0.1, 1.0],
    ])
    filtered = top_k_filter(logits, k=2)
    ids, probs = sample_top_k(logits, k=2)

    allowed = torch.topk(logits, k=2, dim=-1).indices
    allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
    allowed_mask.scatter_(dim=-1, index=allowed, value=True)
    assert filtered.shape == logits.shape
    assert probs.shape == logits.shape
    assert ids.shape == (2,)
    assert torch.all(filtered[~allowed_mask] < -1e8)
    assert torch.all(probs[~allowed_mask] < 1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)

    for b, token_id in enumerate(ids):
        assert allowed_mask[b, token_id]

    print("Input shape:", logits.shape)
    print("Filtered logits shape:", filtered.shape)
    print("Sampled ids shape:", ids.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()

