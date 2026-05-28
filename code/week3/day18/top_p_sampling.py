import torch
import torch.nn.functional as F

def top_p_filter(logits, p, mask_value=-1e9):
    """
    Args:
        logits: [B, V]
        p: float, 0 < p <= 1

    Returns:
        filtered_logits: [B, V]
        keep_mask: [B, V], True means kept
    """
    assert 0 < p <= 1
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove_mask = cumulative_probs > p
    sorted_remove_mask[:, 1:] = sorted_remove_mask[:, :-1]
    sorted_remove_mask[:, 0] = False
    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    filtered_logits = logits.masked_fill(remove_mask, mask_value)
    keep_mask = ~remove_mask
    return filtered_logits, keep_mask

def sample_top_p(logits, p, temperature=1.0):
    """
    Args:
        logits: [B, V]

    Returns:
        next_ids: [B]
        probs: [B, V]
        keep_mask: [B, V]
    """
    assert temperature > 0
    flitered_logits, keep_mask = top_p_filter(logits, p)
    probs = F.softmax(flitered_logits, dim=-1)
    next_ids = torch.multinomial(probs,num_samples=1).squeeze(-1)
    return next_ids, probs, keep_mask

def test():
    torch.manual_seed(0)
    logits = torch.tensor([
        [5.0, 4.0, 1.0, 0.5, 0.0],
        [2.0, 1.9, 1.8, 1.7, 0.1],
    ])

    filtered, keep_mask = top_p_filter(logits, p=0.8)
    ids, probs, keep_mask2 = sample_top_p(logits, p=0.8)

    assert filtered.shape == logits.shape
    assert keep_mask.shape == logits.shape
    assert probs.shape == logits.shape
    assert ids.shape == (2,)
    assert torch.equal(keep_mask, keep_mask2)
    assert torch.all(filtered[~keep_mask] < -1e8)
    assert torch.all(probs[~keep_mask] < 1e-6)
    assert torch.all(keep_mask.sum(dim=-1) >= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)

    for b, token_id in enumerate(ids):
        assert keep_mask[b, token_id]

    print("Input shape:", logits.shape)
    print("Filtered logits shape:", filtered.shape)
    print("Sampled ids shape:", ids.shape)
    print("Kept counts:", keep_mask.sum(dim=-1).tolist())
    print("Test passed.")


if __name__ == "__main__":
    test()