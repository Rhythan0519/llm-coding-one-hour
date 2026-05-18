import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        logits = self.fc2(h)
        return logits
    
def make_xor_data():
    x = torch.tensor([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
    ])
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    return x, y

def test():
    torch.manual_seed(0)
    x, y = make_xor_data()
    model = MLP(2, 8, 2)

    logits = model(x)
    assert logits.shape == (4, 2)

    loss_before = F.cross_entropy(logits, y)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    loss_after = F.cross_entropy(model(x), y)

    assert loss_after < loss_before
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print(f"Loss Final: {loss_after}")
    print("Test passed.")

if __name__ == "__main__":
    test()