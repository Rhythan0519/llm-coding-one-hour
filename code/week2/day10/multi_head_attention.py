import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim % num_heads != 0"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            x: [B, H, T, D]
        """
        B, T, C = x.shape

        x = x.reshape(B, T, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        
        return x
    
    def apply_causal_mask(self, scores):
        """
        Args:
            scores: [B, H, T, T]
        """
        B, H, T, _ = scores.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        masked_scores = scores.masked_fill(mask, -1e9)

        return masked_scores
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C]

        Returns:
            out: [B, T, C]
            attn: [B, H, T, T]
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = (q @ k.transpose(-2, -1))/(v.shape[-1])
        if self.causal:
            scores = self.apply_causal_mask(scores)
        attn = F.softmax(scores, dim=-1)
        context = attn @ v
        context = (context.permute(0, 2, 1, 3)).reshape(B, T, self.embed_dim)
        out = self.out_proj(context)

        return out, attn
    
def test():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 12)
    model = MultiHeadAttention(embed_dim=12, num_heads=3, causal=True)

    out, attn = model(x)

    assert out.shape == x.shape
    assert attn.shape == (2, 3, 5, 5)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)
    assert torch.all(attn[:, :, 0, 1:] < 1e-6)
    assert not torch.isnan(out).any()

    loss = out.pow(2).mean()
    loss.backward()
    assert model.out_proj.weight.grad is not None

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()