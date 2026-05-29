import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 rank=2, alpha=1.0, bias=True):
        super().__init__()
        assert rank >= 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, rank))
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_output = self.base(x)
        if self.rank == 0:
            return base_output
        else:
            lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * (self.alpha / self.rank)
            return base_output + lora_output
        
def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    layer = LoRALinear(3, 5, rank=2, alpha=4.0)
    y = layer(x)

    assert y.shape == (4, 5)
    assert layer.base.weight.requires_grad is False

    rank0 = LoRALinear(3, 5, rank=0)
    y0 = rank0(x)
    ref0 = rank0.base(x)
    assert torch.allclose(y0, ref0, atol=1e-6)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()