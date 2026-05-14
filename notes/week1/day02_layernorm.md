# Day 2：LayerNorm

## 1. 今日目标

今天实现一个 `MyLayerNorm`：

```python
MyLayerNorm(hidden_dim)
```

目标是理解 layer norm 在最后一维上怎么归一化，以及 affine 参数怎么加回去。

---

## 2. 这个模块在大模型里有什么用

LayerNorm 常出现在 Transformer block 里，用来稳定训练，让不同层的激活尺度更可控。

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
```

### 输出

```text
y: [B, T, C]
```

其中 `C` 是特征维度，归一化发生在最后一维。

---

## 4. 核心公式

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

再做 affine：

$$
y = \gamma \hat{x} + \beta
$$

---

## 5. 伪代码

```text
1. 计算最后一维 mean
2. 计算最后一维 variance
3. 标准化
4. 乘 weight，加 bias
5. 返回结果
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        # TODO 1: compute mean over last dim
        mean = None

        # TODO 2: compute variance over last dim
        var = None

        # TODO 3: normalize
        x_hat = None

        # TODO 4: affine transform
        out = None
        return out


def test():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    mine = MyLayerNorm(4)
    ref = nn.LayerNorm(4)

    ref.weight.data.copy_(mine.weight.data)
    ref.bias.data.copy_(mine.bias.data)

    my_y = mine(x)
    ref_y = ref(x)

    assert my_y.shape == x.shape
    assert torch.allclose(my_y, ref_y, atol=1e-6)
    assert torch.allclose(my_y.mean(dim=-1), torch.zeros_like(my_y.mean(dim=-1)), atol=1e-5)

    print("Input shape:", x.shape)
    print("Output shape:", my_y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. shape 正确
2. 和 nn.LayerNorm 对齐
3. mean 近似 0
4. 没有 NaN
```

---

## 8. 运行方式

```text
day02_layernorm.py
python day02_layernorm.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 3, 4])
Output shape: torch.Size([2, 3, 4])
Test passed.
```

---

## 10. 常见错误

```text
1. 归一化维度写成 batch 或 time
2. variance 没加 eps
3. 忘记 affine 参数
4. 用了 unbiased variance
5. 结果 shape 被 squeeze 掉
```

---

## 11. 扩展任务

```text
1. 支持任意 normalized_shape
2. 对比 RMSNorm
3. 观察不同 eps 的效果
4. 做前向/反向梯度检查
5. 和 Transformer block 拼起来
```

---

## 12. 今日理解问题

```text
1. LayerNorm 是对哪一维做归一化？
2. 为什么它比 BatchNorm 更适合 Transformer？
3. weight 和 bias 分别做什么？
4. eps 为什么不能省？
5. mean≈0、std≈1 指的是哪一维？
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

