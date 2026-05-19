import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]

        Returns:
            out: [B, T, C]
            attn: [B, T, T]
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(C)
        attn = F.softmax(scores, dim=-1)
        context = attn @ v
        out = self.out_proj(context)

        return out, attn
    
def test():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    model = SingleHeadAttention(embed_dim=8)

    out, attn = model(x)

    assert out.shape == x.shape
    assert attn.shape == (2, 4, 4)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    assert not torch.isnan(out).any()
    assert not torch.isnan(attn).any()

    loss = out.sum()
    loss.backward()
    assert model.q_proj.weight.grad is not None

    print(f"Out:{out}")
    print(f"Attention:{attn}")
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()