# Day 10：Multi-Head Attention

## 1. 今日目标

今天实现一个 multi-head self-attention 模块：

```python
MultiHeadAttention(embed_dim, num_heads)
```

目标是掌握 `[B,T,C] -> [B,H,T,D] -> [B,T,C]` 的 shape 变换。

---

## 2. 这个模块在大模型里有什么用

多头 attention 让模型在不同子空间里同时关注不同关系。

例如：

```text
1. 一个 head 关注局部邻近 token
2. 一个 head 关注语法关系
3. 一个 head 关注长距离依赖
```

实际模型里不会显式规定每个 head 的含义，但多头结构给了模型并行学习多种关系的能力。

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
```

要求：

```text
C % num_heads == 0
head_dim = C // num_heads
```

### 输出

```text
out: [B, T, C]
attn: [B, H, T, T]
```

---

## 4. 核心公式或核心逻辑

先把 Q / K / V 从 `C` 维拆成多个 head：

```text
[B,T,C] -> [B,T,H,D] -> [B,H,T,D]
```

每个 head 单独做 attention：

$$
\mathrm{head}_h = \mathrm{softmax}\left(\frac{Q_hK_h^T}{\sqrt{D}}\right)V_h
$$

最后把所有 head 拼回去：

```text
[B,H,T,D] -> [B,T,H,D] -> [B,T,C]
```

---

## 5. 伪代码

```text
1. 用 q_proj/k_proj/v_proj 得到 q/k/v: [B,T,C]
2. reshape 成 [B,T,H,D]
3. transpose 成 [B,H,T,D]
4. 计算 scores: [B,H,T,T]
5. 如使用 causal=True，则 mask 未来位置
6. softmax 得到 attn: [B,H,T,T]
7. attn @ v 得到 context: [B,H,T,D]
8. transpose 回 [B,T,H,D]
9. reshape 成 [B,T,C]
10. out_proj 得到 out
```

---

## 6. 代码骨架

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, causal=False):
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

    def split_heads(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            x: [B, H, T, D]
        """
        B, T, C = x.shape

        # TODO 1: reshape to [B, T, H, D]
        x = None

        # TODO 2: transpose to [B, H, T, D]
        x = None

        return x

    def apply_causal_mask(self, scores):
        """
        Args:
            scores: [B, H, T, T]
        """
        B, H, T, _ = scores.shape

        # TODO 3: create causal mask and fill future positions
        masked_scores = None

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

        # TODO 4: project q/k/v
        q = None
        k = None
        v = None

        # TODO 5: split heads
        q = None
        k = None
        v = None

        # TODO 6: compute scaled attention scores
        scores = None

        # TODO 7: optionally apply causal mask
        if self.causal:
            scores = None

        # TODO 8: softmax
        attn = None

        # TODO 9: attention weighted sum
        context = None

        # TODO 10: merge heads back to [B, T, C]
        context = None

        # TODO 11: output projection
        out = None

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
```

---

## 7. 测试要求

```text
1. out.shape == x.shape
2. attn.shape == [B,H,T,T]
3. 每个 head 的 attention 每行和为 1
4. causal=True 时未来位置概率接近 0
5. backward 后参数有梯度
```

---

## 8. 运行方式

保存为：

```text
day10_multi_head_attention.py
```

运行：

```bash
python day10_multi_head_attention.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 5, 12])
Output shape: torch.Size([2, 5, 12])
Attention shape: torch.Size([2, 3, 5, 5])
Test passed.
```

---

## 10. 常见错误

```text
1. reshape 前后元素总数不一致
2. 忘记 transpose，导致 head 维和 time 维混在一起
3. 用 sqrt(C) 而不是 sqrt(head_dim)
4. merge heads 前忘记 contiguous
5. causal mask 没有 broadcast 到 [B,H,T,T]
```

---

## 11. 扩展任务

```text
1. 和 nn.MultiheadAttention 做数值对齐
2. 支持 key_padding_mask
3. 加入 attention dropout
4. 支持返回每个 head 的 attention 可视化
5. 把 q/k/v 合并成一个 qkv_proj
```

---

## 12. 今日理解问题

```text
1. 为什么 embed_dim 必须能被 num_heads 整除？
2. [B,T,C] 怎么变成 [B,H,T,D]？
3. 为什么 attention score 的 shape 是 [B,H,T,T]？
4. 为什么缩放因子是 sqrt(head_dim)？
5. merge heads 时为什么要把 [B,H,T,D] 变回 [B,T,C]？
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

