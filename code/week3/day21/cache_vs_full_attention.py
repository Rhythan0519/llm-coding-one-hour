import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleKVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, new_k, new_v):
        if self.k is None or self.v is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v
    
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
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
        x = x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x
    
    def merge_heads(self, x):
        """
        Args:
            x: [B, H, T, D]
        Returns:
            x: [B, T, C]
        """
        B, H, T, D = x.shape
        x = x.permute(0 ,2, 1, 3).reshape(B, T, self.embed_dim)
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
    
    def forward_full(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            out: [B, T, C]
        """
        B, T, C = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) / (math.sqrt(self.head_dim))
        masked_scores = self.apply_causal_mask(scores)
        attn = torch.softmax(masked_scores, dim=-1)
        context = self.merge_heads(attn @ v)
        out = self.out_proj(context)
        return out


    
    def forward_step(self, x_t, cache):
        """
        Args:
            x_t: [B, 1, C]
            cache: SimpleKVCache

        Returns:
            out_t: [B, 1, C]
        """
        q = self.split_heads(self.q_proj(x_t))
        new_k = self.split_heads(self.k_proj(x_t))
        new_v = self.split_heads(self.v_proj(x_t))

        cache_k, cache_v = cache.update(new_k=new_k, new_v=new_v)
        scores = (q @ cache_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = self.merge_heads(attn @ cache_v)
        out_t = self.out_proj(context)
        return out_t
    
def test():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    attn = CausalSelfAttention(embed_dim=16, num_heads=4)

    full_out = attn.forward_full(x)

    cache = SimpleKVCache()
    step_outs = []
    for t in range(x.size(1)):
        out_t = attn.forward_step(x[:, t:t + 1, :], cache)
        step_outs.append(out_t)
    cache_out = torch.cat(step_outs, dim=1)

    assert full_out.shape == x.shape
    assert cache_out.shape == x.shape
    assert torch.allclose(full_out, cache_out, atol=1e-5)
    assert not torch.isnan(cache_out).any()

    loss = cache_out.pow(2).mean()
    loss.backward()
    assert attn.q_proj.weight.grad is not None

    print("Input shape:", x.shape)
    print("Full output shape:", full_out.shape)
    print("Cache output shape:", cache_out.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()