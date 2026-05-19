# Day 12：Residual + LayerNorm

## 1. 今日目标

今天实现一个 residual + norm 小模块：

```python
ResidualNorm(embed_dim)
```

目标是理解 Transformer block 里为什么经常出现：

```text
x = x + sublayer(norm(x))
```

或者：

```text
x = norm(x + sublayer(x))
```

今天先实现更容易验证的 post-norm 版本：

```text
out = norm(x + sublayer_out)
```

---

## 2. 这个模块在大模型里有什么用

Residual connection 让梯度更容易穿过深层网络。LayerNorm 让每层输出尺度更稳定。

在 Transformer 中，它们通常包住 attention 和 FFN：

```text
attention sublayer + residual + norm
ffn sublayer + residual + norm
```

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
sublayer_out: [B, T, C]
```

### 输出

```text
out: [B, T, C]
```

---

## 4. 核心逻辑

先做 residual add：

$$
r = x + \mathrm{sublayer\_out}
$$

再做 LayerNorm：

$$
out = \mathrm{LayerNorm}(r)
$$

---

## 5. 伪代码

```text
1. 检查 x 和 sublayer_out shape 一致
2. residual = x + sublayer_out
3. out = layer_norm(residual)
4. 返回 out
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn


class ResidualNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sublayer_out):
        """
        Args:
            x: [B, T, C]
            sublayer_out: [B, T, C]

        Returns:
            out: [B, T, C]
        """
        # TODO 1: residual add
        residual = None

        # TODO 2: layer norm
        out = None

        return out


def test():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, requires_grad=True)
    sublayer_out = torch.randn(2, 4, 8, requires_grad=True)

    mine = ResidualNorm(embed_dim=8)
    ref = nn.LayerNorm(8)
    ref.weight.data.copy_(mine.norm.weight.data)
    ref.bias.data.copy_(mine.norm.bias.data)

    my_out = mine(x, sublayer_out)
    ref_out = ref(x + sublayer_out)

    assert my_out.shape == x.shape
    assert torch.allclose(my_out, ref_out, atol=1e-6)
    assert torch.allclose(my_out.mean(dim=-1), torch.zeros(2, 4), atol=1e-5)
    assert not torch.isnan(my_out).any()

    loss = my_out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert sublayer_out.grad is not None
    assert mine.norm.weight.grad is not None

    print("Input shape:", x.shape)
    print("Output shape:", my_out.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 输出 shape 和输入一致
2. 和 nn.LayerNorm(x + sublayer_out) 对齐
3. 最后一维 mean 近似 0
4. x 和 sublayer_out 都能收到梯度
5. 输出没有 NaN
```

---

## 8. 运行方式

保存为：

```text
day12_residual_norm.py
```

运行：

```bash
python day12_residual_norm.py
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
1. x 和 sublayer_out shape 不一致还直接相加
2. LayerNorm 维度写成 T 而不是 C
3. 忘记 residual，只返回 sublayer_out
4. inplace 操作破坏梯度
5. 误以为 LayerNorm 会改变整体 shape
```

---

## 11. 扩展任务

```text
1. 实现 pre-norm: x + sublayer(norm(x))
2. 加入 dropout
3. 用自己 Day 2 的 MyLayerNorm 替换 nn.LayerNorm
4. 比较 pre-norm 和 post-norm 的代码结构
5. 接到 attention 或 FFN 后面
```

---

## 12. 今日理解问题

```text
1. residual connection 为什么要求 shape 一致？
2. LayerNorm 是对哪一维做归一化？
3. residual 对梯度传播有什么帮助？
4. post-norm 和 pre-norm 的区别是什么？
5. 为什么 Transformer block 里 attention 和 FFN 后面都常有 residual？
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

