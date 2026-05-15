import torch
import torch.nn as nn

class Layernorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        #除以N，这里如果是default：unbiased=False则求方差的时候会变成除以N-1
        x_hat = (x - mean)/(torch.sqrt(var + self.eps))
        print(x_hat)
        out = self.weight * x_hat + self.bias
        print(out)
        return out

def test():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    mine = Layernorm(4)
    ref = nn.LayerNorm(4)

    ref.weight.data.copy_(mine.weight.data)
    ref.bias.data.copy_(mine.bias.data)
    my_y = mine(x)
    ref_y = ref(x)

    assert my_y.shape == ref_y.shape
    assert torch.allclose(my_y, ref_y, atol=1e-6)
    assert torch.allclose(my_y.mean(dim=-1), torch.zeros_like(my_y.mean(dim=-1)), atol=1e-5)

    print("Input shape:", x.shape)
    print("Output shape:", my_y.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()