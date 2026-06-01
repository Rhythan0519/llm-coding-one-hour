import torch
import torch.nn as nn
import math
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
            self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            nn.init.zeros_(self.lora_B)
    def forward(self, x):
        base_output = self.base(x)
        if self.rank == 0:
            return base_output
        else:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * (self.alpha / self.rank)
            return lora + base_output
class LoRACausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=2):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = LoRALinear(embed_dim, embed_dim, rank=rank)
        self.k_proj = LoRALinear(embed_dim, embed_dim, rank=rank)
        self.v_proj = LoRALinear(embed_dim, embed_dim, rank=rank)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        B, T, _ = x.shape
        output = x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return output

    def _merge_heads(self, x):
        B, H, T, _ = x.shape
        output = x.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return output
    
    def _apply_causal_mask(self, scores):
        B, H, T, _ = scores.shape
        mask_empty = torch.ones_like(scores, dtype=torch.bool)
        mask = torch.triu(mask_empty, diagonal=1)
        masked_scores = scores.masked_fill(mask, -1e9)
        return masked_scores
    
    def forward(self, x):
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = (q @ k.transpose(-1, -2)) / (math.sqrt(self.head_dim))
        masked_scores = self._apply_causal_mask(scores)
        attn = torch.softmax(masked_scores, dim=-1)
        context = attn @ v
        context = self._merge_heads(context)
        out = self.out_proj(context)
        return out

def test():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    attn = LoRACausalSelfAttention(embed_dim=16, num_heads=4, rank=2)
    y = attn(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()

    loss = y.pow(2).mean()
    loss.backward()
    lora_params = [
        p for name, p in attn.named_parameters()
        if "lora_" in name
    ]
    assert len(lora_params) > 0
    assert all(p.grad is not None for p in lora_params if p.requires_grad)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()