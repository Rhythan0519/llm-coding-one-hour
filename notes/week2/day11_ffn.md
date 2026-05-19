# Day 11：Transformer Feed-Forward Network

## 1. 今日目标

今天实现一个 Transformer FFN：

```python
TransformerFFN(embed_dim, hidden_dim)
```

目标是理解 attention 之后的逐 token 非线性变换。

---

## 2. 这个模块在大模型里有什么用

Transformer block 通常包含两类子层：

```text
1. attention：负责 token 之间的信息交互
2. FFN：负责对每个 token 的 hidden state 做非线性变换
```

FFN 不混合不同 token，它只在最后一维 `C` 上做变换。

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
```

### 输出

```text
out: [B, T, C]
```

内部会先升维再降维：

```text
[B,T,C] -> [B,T,hidden_dim] -> [B,T,C]
```

---

## 4. 核心逻辑

经典 FFN 结构：

```text
Linear(C -> hidden_dim)
GELU
Linear(hidden_dim -> C)
```

公式：

$$
\mathrm{FFN}(x) = W_2\,\mathrm{GELU}(W_1x + b_1) + b_2
$$

---

## 5. 伪代码

```text
1. x 经过 fc1 得到 h: [B,T,hidden_dim]
2. h 经过 GELU 激活
3. h 经过 fc2 得到 out: [B,T,C]
4. 返回 out
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]

        Returns:
            out: [B, T, C]
        """
        # TODO 1: first linear projection
        h = None

        # TODO 2: GELU activation
        h = None

        # TODO 3: project back to embed_dim
        out = None

        return out


def test():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    mine = TransformerFFN(embed_dim=8, hidden_dim=32)
    ref = nn.Sequential(
        nn.Linear(8, 32),
        nn.GELU(),
        nn.Linear(32, 8),
    )

    ref[0].weight.data.copy_(mine.fc1.weight.data)
    ref[0].bias.data.copy_(mine.fc1.bias.data)
    ref[2].weight.data.copy_(mine.fc2.weight.data)
    ref[2].bias.data.copy_(mine.fc2.bias.data)

    my_out = mine(x)
    ref_out = ref(x)

    assert my_out.shape == x.shape
    assert torch.allclose(my_out, ref_out, atol=1e-6)
    assert not torch.isnan(my_out).any()

    loss = my_out.pow(2).mean()
    loss.backward()
    assert mine.fc1.weight.grad is not None
    assert mine.fc2.weight.grad is not None

    print("Input shape:", x.shape)
    print("Output shape:", my_out.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. my_out.shape == x.shape
2. 和 nn.Sequential(Linear, GELU, Linear) 对齐
3. 输出没有 NaN
4. backward 后两层 Linear 都有梯度
```

---

## 8. 运行方式

保存为：

```text
day11_ffn.py
```

运行：

```bash
python day11_ffn.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 4, 8])
Output shape: torch.Size([2, 4, 8])
Test passed.
```

---

## 10. 常见错误

```text
1. fc1 和 fc2 的输入输出维度写反
2. 忘记激活函数
3. 输出维度没有回到 embed_dim
4. 把 token 维 T 当成特征维 C
5. backward 前后没有检查梯度
```

---

## 11. 扩展任务

```text
1. 把 GELU 换成 ReLU 比较结果
2. 加入 dropout
3. hidden_dim 设成 4 * embed_dim
4. 实现 SwiGLU 版本
5. 和 Day 13 的 Transformer block 拼起来
```

---

## 12. 今日理解问题

```text
1. FFN 会不会混合不同 token？
2. 为什么 FFN 输入输出 shape 都是 [B,T,C]？
3. hidden_dim 通常为什么比 embed_dim 大？
4. GELU 和 ReLU 有什么区别？
5. FFN 在 Transformer block 的哪个位置？
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

