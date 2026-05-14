import torch
import torch.nn.functional as F

def soft_max(logits, dim=-1):
    stable = logits - logits.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(stable)
    probs = exp_x / exp_x.sum(dim=-1, keepdim=True)
    return probs

def cross_entropy(logits, labels):
    probs = soft_max(logits)
    correct_prob = probs[torch.arange(probs.size(0)), labels]
    loss = -torch.log(correct_prob).sum(dim=0)/correct_prob.size(0)
    return loss

def test():
    torch.manual_seed(0)
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 2, 3, 5])

    probs = soft_max(logits)
    ref_probs = F.softmax(logits, dim=-1)
    loss = cross_entropy(logits, labels)
    ref_loss = F.cross_entropy(logits, labels)
    assert probs.shape == ref_probs.shape, "probs is wrong shape"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(logits.size(0)), atol=1e-6), "probs和不为一"
    assert torch.allclose(probs, ref_probs, atol=1e-6)
    assert torch.allclose(loss, ref_loss, atol=1e-6)

    print("Input shape:", logits.shape)
    print("Output shape:", probs.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()