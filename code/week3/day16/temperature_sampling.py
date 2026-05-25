import torch
import torch.nn.functional as F

def apply_temperature(logits, temperature):
    """
    Args:
        logits: [B, V]
        temperature: positive float

    Returns:
        probs: [B, V]
    """
    assert temperature > 0
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return probs

def sample_with_temperature(logits,temperature):
    """
    Args:
        logits: [B, V]

    Returns:
        next_ids: [B]
        probs: [B, V]
    """
    probs = apply_temperature(logits, temperature)
    next_ids = torch.multinomial(probs, 1).squeeze(-1)
    return next_ids, probs

def entropy(probs):
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

def test():
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.0]])

    probs_low = apply_temperature(logits, temperature=0.5)
    probs_high = apply_temperature(logits, temperature=2.0)
    ids, probs = sample_with_temperature(logits.repeat(4, 1), temperature=1.0)

    assert probs_low.shape == logits.shape
    assert probs_high.shape == logits.shape
    assert ids.shape == (4,)
    assert torch.allclose(probs_low.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(probs_high.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert entropy(probs_high) > entropy(probs_low)
    assert not torch.isnan(probs).any()
    assert ids.dtype == torch.long

    print("Input shape:", logits.shape)
    print("Output probs shape:", probs.shape)
    print("Sampled ids shape:", ids.shape)
    print("Low temp entropy:", round(entropy(probs_low).item(), 4))
    print("High temp entropy:", round(entropy(probs_high).item(), 4))
    print("Test passed.")


if __name__ == "__main__":
    test()