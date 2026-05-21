import torch
import torch.nn as nn

class ResidualNorm(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sublayer_out):
        """
        Args:
            x: [B, T, C]
            sublayer_out: [B, T, C]

        Returns:
            out: [B, T, C]
        """

        residual = x + sublayer_out

        out = self.norm(residual)
        return out

def test():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, requires_grad=True)
    sublayer_out = torch.randn(2, 4, 8, requires_grad=True)

    mine = ResidualNorm(embed_dim=8)
    ref = nn.LayerNorm(8)
    ref.weight.data.copy_(mine.norm.weight.data)
    ref.bias.data.copy_(mine.norm.bias.data)

    my_out = mine(x, sublayer_out)
    ref_out = ref(x + sublayer_out)

    assert my_out.shape == x.shape
    assert torch.allclose(my_out, ref_out, atol=1e-6)
    assert torch.allclose(my_out.mean(dim=-1), torch.zeros(2, 4), atol=1e-5)
    assert not torch.isnan(my_out).any()

    loss = my_out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert sublayer_out.grad is not None
    assert mine.norm.weight.grad is not None

    print("Input shape:", x.shape)
    print("Output shape:", my_out.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
