import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_head(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            x: [B, H, T, D]
        """
        B, T, C = x.shape
        out = x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        return out
    
    def _apply_causal_mask(self, scores):
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

        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)

        scores = (q @ k.transpose(-2, -1)) / (math.sqrt(C))
        if self.causal:
            scores = self._apply_causal_mask(scores)
        
        attn = F.softmax(scores, dim=-1)
        context = attn @ v
        context = context.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(context)
        return attn, out

class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = F.gelu(h)
        out = self.fc2(h)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ffn_hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, causal=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = TransformerFFN(embed_dim, ffn_hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]

        Returns:
            x: [B, T, C]
            attn: [B, H, T, T]
        """
        attn_in = self.norm1(x)
        attn, attn_out = self.attn(attn_in)
        x = x + attn_out
        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out
        return x, attn
    
def test():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    block = TransformerBlock(embed_dim=16, num_heads=4, ffn_hidden_dim=64)

    out, attn = block(x)

    assert out.shape == x.shape
    assert attn.shape == (2, 4, 5, 5)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)
    assert torch.all(attn[:, :, 0, 1:] < 1e-6)
    assert not torch.isnan(out).any()

    loss = out.pow(2).mean()
    loss.backward()
    assert block.norm1.weight.grad is not None
    assert block.attn.q_proj.weight.grad is not None
    assert block.ffn.fc1.weight.grad is not None

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()