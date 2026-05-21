import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]

        Returns:
            out: [B, T, C]
        """

        h = self.fc1(x)
        h = F.gelu(h)
        out = self.fc2(h)

        return out
    
def test():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    mine = TransformerFFN(embed_dim=8, hidden_dim=32)
    ref = nn.Sequential(
        nn.Linear(8, 32),
        nn.GELU(),
        nn.Linear(32, 8),
    )

    ref[0].weight.data.copy_(mine.fc1.weight.data)
    ref[0].bias.data.copy_(mine.fc1.bias.data)
    ref[2].weight.data.copy_(mine.fc2.weight.data)
    ref[2].bias.data.copy_(mine.fc2.bias.data)

    my_out = mine(x)
    ref_out = ref(x)

    assert my_out.shape == x.shape
    assert torch.allclose(my_out, ref_out, atol=1e-6)
    assert not torch.isnan(my_out).any()

    loss = my_out.pow(2).mean()
    loss.backward()
    assert mine.fc1.weight.grad is not None
    assert mine.fc2.weight.grad is not None

    print("Input shape:", x.shape)
    print("Output shape:", my_out.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()