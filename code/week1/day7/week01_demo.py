import torch
import torch.nn as nn
import torch.nn.functional as F

def make_toy_data(n=256):
    torch.manual_seed(0)
    x0 = torch.randn(n // 2, 2) + torch.tensor([1, 1])
    x1 = torch.randn(n // 2, 2) + torch.tensor([-1, -1])
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat(
        [torch.zeros(n // 2, dtype=torch.long),
         torch.ones(n//2, dtype=torch.long)]
    )
    return x, y

class TinyClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(16,2)
        )

    def forward(self, x):
        return self.net(x)
    
def train():
    torch.manual_seed(0)
    x, y = make_toy_data()
    model = TinyClassifier(2, 16, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _ in range (200):
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss)
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        return losses, acc, pred
    
def test():
    losses, acc, pred = train()

    assert len(losses) == 200
    assert losses[-1] < losses[0]
    assert acc > 0.8

    print("Input shape:", torch.Size([256, 2]))
    print("Output shape:", pred.shape)
    print("Final accuracy:", round(acc, 4))
    print("Test passed.")

if __name__ == "__main__":
    test()
