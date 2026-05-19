# Day 8：Single-Head Attention

## 1. 今日目标

今天实现一个单头 self-attention 模块：

```python
SingleHeadAttention(embed_dim)
```

目标是理解一个 token 如何通过 Q / K / V 从同一段序列里聚合上下文信息。

---

## 2. 这个模块在大模型里有什么用

Attention 是 Transformer 的核心。它让每个 token 根据相似度去“看”序列中的其他 token。

在 LLM 中，self-attention 通常负责：

```text
1. 计算 token 与 token 之间的相关性
2. 根据相关性加权聚合 value
3. 让当前位置得到上下文相关的 hidden state
```

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
```

其中：

```text
B = batch size
T = sequence length
C = embedding / hidden dimension
```

示例：

```python
x = torch.randn(2, 4, 8)
```

### 输出

```text
out: [B, T, C]
attn: [B, T, T]
```

`attn[b, i, j]` 表示第 `b` 个样本中，第 `i` 个 token 对第 `j` 个 token 的注意力权重。

---

## 4. 核心公式

先计算 Q / K / V：

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v
$$

再计算 scaled dot-product attention：

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{C}}\right)V
$$

---

## 5. 伪代码

```text
1. x 经过 q_proj 得到 q: [B,T,C]
2. x 经过 k_proj 得到 k: [B,T,C]
3. x 经过 v_proj 得到 v: [B,T,C]
4. q @ k.transpose(-2, -1) 得到 scores: [B,T,T]
5. scores 除以 sqrt(C)
6. 对 scores 最后一维做 softmax 得到 attn
7. attn @ v 得到 context: [B,T,C]
8. context 经过 out_proj 得到 out
9. 返回 out 和 attn
```

---

## 6. 代码骨架

```python
import math
import torch
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

        # TODO 1: project x to q, k, v
        q = None
        k = None
        v = None

        # TODO 2: compute attention scores [B, T, T]
        scores = None

        # TODO 3: scale scores by sqrt(C)
        scores = None

        # TODO 4: softmax over key dimension
        attn = None

        # TODO 5: weighted sum of values
        context = None

        # TODO 6: output projection
        out = None

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
2. attn.shape == [B,T,T]
3. attention 每一行求和接近 1
4. out 和 attn 没有 NaN
5. backward 后 q_proj.weight 有梯度
```

---

## 8. 运行方式

保存为：

```text
day08_single_head_attention.py
```

运行：

```bash
python day08_single_head_attention.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 4, 8])
Output shape: torch.Size([2, 4, 8])
Attention shape: torch.Size([2, 4, 4])
Test passed.
```

---

## 10. 常见错误

```text
1. k.transpose(-2, -1) 写错，导致 scores 不是 [B,T,T]
2. 忘记除以 sqrt(C)
3. softmax 的 dim 写成了 -2
4. attn @ v 的矩阵乘法维度不匹配
5. 没检查 attention 每行和是否为 1
```

---

## 11. 扩展任务

```text
1. 加入 dropout 到 attention weight
2. 返回 scores 观察数值范围
3. 尝试不除 sqrt(C)，比较 attention 是否更尖锐
4. 支持传入 mask
5. 和 torch.nn.functional.scaled_dot_product_attention 对齐
```

---

## 12. 今日理解问题

```text
1. Q、K、V 的 shape 分别是什么？
2. scores 的 shape 为什么是 [B,T,T]？
3. softmax 应该沿哪个维度做？
4. 为什么 scores 要除以 sqrt(C)？
5. attn @ v 的输出为什么还是 [B,T,C]？
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

