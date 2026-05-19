# Day 13：Transformer Block

## 1. 今日目标

今天实现一个最小 Transformer block：

```python
TransformerBlock(embed_dim, num_heads, ffn_hidden_dim)
```

目标是把本周前几天的模块串起来：

```text
multi-head attention
residual connection
LayerNorm
FFN
```

---

## 2. 这个模块在大模型里有什么用

GPT / BERT / ViT / DiT 都是由很多 Transformer block 堆起来的。

一个 block 可以粗略理解为：

```text
1. attention 让 token 之间交换信息
2. residual 保留原始信息并帮助梯度传播
3. norm 稳定激活尺度
4. FFN 对每个 token 做非线性变换
```

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
```

### 输出

```text
out: [B, T, C]
attn: [B, H, T, T]
```

---

## 4. 核心逻辑

今天实现 pre-norm GPT 风格结构：

```text
x = x + attention(norm1(x))
x = x + ffn(norm2(x))
```

pre-norm 的特点是先归一化，再进入子层。

---

## 5. 伪代码

```text
1. norm1(x)
2. 输入 multi-head attention，得到 attn_out 和 attn
3. x = x + attn_out
4. norm2(x)
5. 输入 FFN，得到 ffn_out
6. x = x + ffn_out
7. 返回 x 和 attn
```

---

## 6. 代码骨架

```python
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

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            out: [B, T, C]
            attn: [B, H, T, T]
        """
        # TODO: reuse your Day 10 multi-head attention implementation here
        out = None
        attn = None
        return out, attn


class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        # TODO: reuse your Day 11 FFN implementation here
        out = None
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
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
        # TODO 1: norm then attention
        attn_in = None
        attn_out, attn = None

        # TODO 2: first residual connection
        x = None

        # TODO 3: norm then FFN
        ffn_in = None
        ffn_out = None

        # TODO 4: second residual connection
        x = None

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
```

---

## 7. 测试要求

```text
1. out.shape == x.shape
2. attn.shape == [B,H,T,T]
3. causal attention 未来位置概率接近 0
4. 输出没有 NaN
5. backward 后 attention、norm、ffn 参数都有梯度
```

---

## 8. 运行方式

保存为：

```text
day13_transformer_block.py
```

运行：

```bash
python day13_transformer_block.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 5, 16])
Output shape: torch.Size([2, 5, 16])
Attention shape: torch.Size([2, 4, 5, 5])
Test passed.
```

---

## 10. 常见错误

```text
1. residual 连接时把 norm 后的输入加错
2. attention 返回的是 tuple，却直接当 tensor 用
3. FFN 输出维度没有回到 embed_dim
4. causal mask 漏掉，导致第 0 个 token 看到未来
5. backward 没跑通但没有检查梯度
```

---

## 11. 扩展任务

```text
1. 加入 dropout
2. 比较 pre-norm 和 post-norm
3. 堆叠两个 TransformerBlock
4. 接 token embedding 和 lm_head 做语言模型
5. 打印每层 attention 的 shape
```

---

## 12. 今日理解问题

```text
1. pre-norm 的计算顺序是什么？
2. attention 子层和 FFN 子层各自做什么？
3. 为什么两个子层都要 residual？
4. Transformer block 为什么输入输出 shape 保持一致？
5. attn 的 shape 为什么包含 head 维？
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
- 明天要复习：
```

