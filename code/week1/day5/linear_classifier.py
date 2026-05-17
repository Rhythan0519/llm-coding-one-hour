import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        logits = x @ self.weight.T + self.bias
        return logits
    
def make_toy_data(n=64):
    torch.manual_seed(0)
    x0 = torch.randn(n//2, 2) + torch.tensor([-2.0, -2.0])
    x1 = torch.randn(n // 2, 2) + torch.tensor([2.0, 2.0])
    x = torch.concat([x0, x1], dim=0)
    y = torch.cat([
        torch.zeros(n // 2, dtype=torch.long),
        torch.ones(n // 2, dtype=torch.long),
    ])
    return x, y

def test():
    torch.manual_seed(0)
    x, y = make_toy_data()
    mine = LinearClassifier(2, 2)
    ref = nn.Linear(2, 2)
    ref.weight.data.copy_(mine.weight.data)
    ref.bias.data.copy_(mine.bias.data)

    my_logits = mine(x)
    ref_logits = ref(x)
    
    assert torch.allclose(my_logits, ref_logits, atol=1e-6)
    loss_before = F.cross_entropy(my_logits, y)
    opt = torch.optim.SGD(mine.parameters(), lr=0.1)
    for _ in range (20):
        opt.zero_grad()
        logits = mine(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    loss_after = F.cross_entropy(mine(x), y)

    assert loss_after < loss_before
    print("Input shape:", x.shape)
    print("Output shape:", my_logits.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()