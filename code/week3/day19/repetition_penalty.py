import torch
import torch.nn.functional as F

def apply_repetition_penalty(logits, history, penalty=1.2):
    """
    Args:
        logits: [B, V]
        history: [B, T]
        penalty: float >= 1

    Returns:
        adjusted_logits: [B, V]
    """
    assert penalty >= 1.0

    adjusted = logits.clone()
    B, V = logits.shape
    for i in range(B):
        token_ids = history[i]
        scores = adjusted[i, token_ids]
        adjusted[i, token_ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return adjusted

def test():
    logits = torch.tensor([
        [4.0, 2.0, -1.0, 0.5],
        [1.0, -2.0, 3.0, 0.1],
    ])
    history = torch.tensor([
        [0, 2, 2],
        [1, 1, 3],
    ])

    adjusted = apply_repetition_penalty(logits, history, penalty=2.0)

    assert adjusted.shape == logits.shape
    assert torch.allclose(logits, torch.tensor([
        [4.0, 2.0, -1.0, 0.5],
        [1.0, -2.0, 3.0, 0.1],
    ]))
    assert torch.allclose(adjusted[0, 0], torch.tensor(2.0))
    assert torch.allclose(adjusted[0, 2], torch.tensor(-2.0))
    assert torch.allclose(adjusted[0, 1], logits[0, 1])
    assert torch.allclose(adjusted[1, 1], torch.tensor(-4.0))
    assert torch.allclose(adjusted[1, 3], torch.tensor(0.05))
    assert not torch.isnan(adjusted).any()

    probs_before = F.softmax(logits, dim=-1)
    probs_after = F.softmax(adjusted, dim=-1)
    assert probs_after[0, 0] < probs_before[0, 0]

    print("Input logits shape:", logits.shape)
    print("History shape:", history.shape)
    print("Output shape:", adjusted.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()