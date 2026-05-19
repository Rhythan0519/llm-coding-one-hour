# Day 14：Tiny GPT Demo

## 1. 今日目标

今天实现一个 tiny character-level GPT demo：

```python
TinyGPT(vocab_size, block_size, embed_dim, num_heads, num_layers)
```

目标是把本周的 Transformer 模块串成一个最小语言模型闭环：

```text
text -> token ids -> logits -> loss -> backward -> generate
```

---

## 2. 这个模块在大模型里有什么用

GPT 的训练目标是 next-token prediction：

```text
给定前面的 token，预测下一个 token。
```

这个 tiny demo 不追求效果，只训练以下工程手感：

```text
1. token embedding
2. positional embedding
3. causal Transformer block
4. language modeling head
5. cross entropy loss
6. autoregressive generation
```

---

## 3. 输入输出

### 输入

```text
idx: [B, T]
targets: [B, T]
```

其中 `idx` 是输入 token ids，`targets` 是每个位置要预测的下一个 token。

### 输出

```text
logits: [B, T, V]
loss: scalar 或 None
generated_ids: [B, T_new]
```

其中 `V = vocab_size`。

---

## 4. 核心逻辑

语言模型 forward：

```text
1. token embedding: [B,T] -> [B,T,C]
2. position embedding: [T] -> [T,C]
3. 两者相加得到 x: [B,T,C]
4. 经过若干 TransformerBlock
5. LayerNorm
6. lm_head 投影到词表: [B,T,V]
7. 如果有 targets，计算 cross entropy
```

生成：

```text
1. 取最近 block_size 个 token
2. forward 得到 logits
3. 取最后一个位置 logits
4. softmax 得到概率
5. 采样或 argmax 得到 next_id
6. 拼到序列后面
```

---

## 5. 伪代码

```text
1. 构造字符表 stoi / itos
2. 把训练文本 encode 成 token ids
3. 随机采样 batch: x=[B,T], y=[B,T]
4. 初始化 TinyGPT
5. 训练若干步，让 loss 下降
6. 用 prefix 生成字符
7. 打印 loss、logits shape 和生成文本
```

---

## 6. 代码骨架

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        super().__init__()
        # TODO: reuse your Day 13 TransformerBlock modules
        self.placeholder = None

    def forward(self, x):
        # TODO: implement or paste your Day 13 block here
        out = None
        return out


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4 * embed_dim)
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

        # TODO 1: token embeddings [B, T, C]
        tok_emb = None

        # TODO 2: position ids [T] and position embeddings [T, C]
        pos = None
        pos_emb = None

        # TODO 3: add token and position embeddings
        x = None

        # TODO 4: pass through Transformer blocks
        for block in self.blocks:
            x = None

        # TODO 5: final norm and lm head
        x = None
        logits = None

        # TODO 6: compute cross entropy if targets is not None
        loss = None

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
            # TODO 7: crop to block_size
            idx_cond = None

            # TODO 8: forward and get last-position logits
            logits, _ = None
            last_logits = None

            # TODO 9: convert logits to probabilities and sample one token
            probs = None
            next_id = None

            # TODO 10: append sampled token
            idx = None

        return idx


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def get_batch(data, block_size, batch_size):
    # TODO 11: sample random starting positions
    ix = None

    # TODO 12: stack input and target chunks
    x = None
    y = None
    return x, y


def train_tiny_gpt():
    torch.manual_seed(0)
    text = "hello world hello transformer hello tiny gpt "
    stoi, itos = build_vocab(text)
    data = encode(text * 20, stoi)

    block_size = 8
    batch_size = 8
    model = TinyGPT(
        vocab_size=len(stoi),
        block_size=block_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
```

---

## 7. 测试要求

```text
1. logits.shape == [B,T,V]
2. loss 是 scalar
3. 训练后 losses[-1] < losses[0]
4. generated_text 长度符合预期
5. logits 没有 NaN
```

---

## 8. 运行方式

保存为：

```text
day14_tiny_gpt.py
```

运行：

```bash
python day14_tiny_gpt.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 8])
Output logits shape: torch.Size([2, 8, vocab_size])
Loss before: ...
Loss after: ...
Generated: 'h...'
Test passed.
```

---

## 10. 常见错误

```text
1. targets 没有右移，导致输入和标签关系错
2. cross entropy 没把 logits reshape 成 [B*T,V]
3. position embedding 的长度超过 block_size
4. generate 时没有裁剪最近 block_size 个 token
5. 采样 next_id 后没有拼回 idx
```

---

## 11. 扩展任务

```text
1. generate 使用 argmax 而不是采样，对比输出
2. 加入 temperature
3. 打印每 20 step 的 loss
4. 换更长文本训练
5. 保存并重新加载模型参数
```

---

## 12. 今日理解问题

```text
1. 为什么 logits 是 [B,T,V]？
2. 为什么语言模型 targets 通常是输入向右移动一位？
3. causal mask 在 tiny GPT 里防止了什么？
4. generate 时为什么只取最后一个位置的 logits？
5. block_size 限制了什么？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出 shape：
- 我最容易错的地方：
- 我现在能不能不看代码重写：
- 下周要复习：
```

