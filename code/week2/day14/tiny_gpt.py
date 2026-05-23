import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class _MultiHeadAttention(nn.Module):
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
        B, T, C = x.shape
        out = x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return out
    
    def _apply_causal_mask(self, scores):
        B, H, T, _ = scores.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores_masked = scores.masked_fill(mask, -1e9)
        return scores_masked
    
    def forward(self, x):
        B, T, C = x.shape
        q = self._split_head(self.q_proj(x))
        k = self._split_head(self.k_proj(x))
        v = self._split_head(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1)) / (math.sqrt(self.head_dim))
        if self.causal:
            scores = self._apply_causal_mask(scores)
        attn = torch.softmax(scores, dim=-1)
        context = attn @ v
        context = context.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.out_proj(context)

        return out
    
class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        h = self.fc1(x)
        h = torch.relu(h)
        out = self.fc2(h)

        return out
    
class TransFormerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _MultiHeadAttention(embed_dim, num_heads, causal=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = TransformerFFN(embed_dim, ffn_hidden_dim)

    def forward(self, x):
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in)
        x = x + attn_out
        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        out = x + ffn_out
        return out
    
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, 
                 embed_dim=32, num_heads=4, 
                 num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransFormerBlock(embed_dim, num_heads, 4 * embed_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: [B, T]
            targets: [B, T] or None

        Returns:
            logits: [B, T, V]
            loss: scalar or None
        """
        B, T = idx.shape
        assert T <= self.block_size
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B*T, -1),
                                    targets.reshape(B*T))

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Args:
            idx: [B, T]
        Returns:
            idx: [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]# :表示倒数
            logits, _ = self.forward(idx_cond)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)# 根据概率随机抽一个 token，但也可以用argmax贪心直接选
            idx = torch.cat([idx, next_id], dim=-1)

        return idx
    
def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)

def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

def train_tiny_gpt():
    torch.manual_seed(0)
    text = "hello world hello transformwe hello tiny gpt "
    stoi, itos = build_vocab(text)
    data = encode(text * 20, stoi)
    block_size = 8
    batch_size = 8
    model = TinyGPT(vocab_size=len(stoi), block_size=block_size,
                    embed_dim=32, num_heads=4, num_layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    losses = []
    for _ in range(80):
        x, y = get_batch(data, block_size, batch_size)
        logits, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    prefix = torch.tensor([[stoi["h"]]], dtype=torch.long)
    generated = model.generate(prefix, max_new_tokens=20)[0]
    generated_text = decode(generated, itos)
    return model, losses, generated_text, len(stoi)

def test():
    model, losses, generated_text, vocab_size = train_tiny_gpt()

    x = torch.randint(0, vocab_size, (2, 8))
    logits, loss = model(x, x)

    assert logits.shape == (2, 8, vocab_size)
    assert loss.ndim == 0
    assert losses[-1] < losses[0]
    assert len(generated_text) == 21
    assert not torch.isnan(logits).any()

    print("Input shape:", x.shape)
    print("Output logits shape:", logits.shape)
    print("Loss before:", round(losses[0], 4))
    print("Loss after:", round(losses[-1], 4))
    print("Generated:", repr(generated_text))
    print("Test passed.")


if __name__ == "__main__":
    test()