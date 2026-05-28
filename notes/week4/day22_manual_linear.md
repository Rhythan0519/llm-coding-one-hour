# Day 22：Manual Linear

## 1. 今日目标

今天实现：

```python
manual_linear(x, weight, bias=None)
```

目标是用自己的代码实现 Linear，并和 `torch.nn.functional.linear` 对齐。

---

## 2. 这个模块在大模型里有什么用

Linear 是 Transformer 里最常见的模块：

```text
1. Q/K/V projection
2. attention output projection
3. FFN up/down projection
4. LM head
```

LoRA 也是作用在线性层上，所以今天先把普通 Linear 的 shape 打牢。

---

## 3. 输入输出

输入：

```text
x: [B, in_features]
weight: [out_features, in_features]
bias: [out_features] or None
```

输出：

```text
y: [B, out_features]
```

---

## 4. 核心公式

```text
y = x @ weight.T + bias
```

注意：PyTorch 的 Linear weight 是 `[out_features, in_features]`。

---

## 5. 伪代码

```text
1. 检查 x 是 2D
2. 检查 weight 是 2D
3. 检查 x.shape[-1] == weight.shape[-1]
4. y = x @ weight.T
5. 如果 bias 不为 None，加 bias
6. 返回 y
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def manual_linear(x, weight, bias=None):
    """
    Args:
        x: [B, in_features]
        weight: [out_features, in_features]
        bias: [out_features] or None

    Returns:
        y: [B, out_features]
    """
    # TODO 1: assert shapes
    # TODO 2: compute x @ weight.T
    # TODO 3: add bias if needed
    return None


def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    weight = torch.randn(5, 3)
    bias = torch.randn(5)

    y = manual_linear(x, weight, bias)
    ref = F.linear(x, weight, bias)

    assert y.shape == (4, 5)
    assert torch.allclose(y, ref, atol=1e-6)

    y_no_bias = manual_linear(x, weight, None)
    ref_no_bias = F.linear(x, weight, None)
    assert torch.allclose(y_no_bias, ref_no_bias, atol=1e-6)

    print("Input shape:", x.shape)
    print("Weight shape:", weight.shape)
    print("Output shape:", y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 输出 shape 是 [B,out_features]
2. 有 bias 时对齐 F.linear
3. 无 bias 时对齐 F.linear
4. 至少打印 input / weight / output shape
```

---

## 8. 运行方式

```bash
python code/week4/day22/manual_linear.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([4, 3])
Weight shape: torch.Size([5, 3])
Output shape: torch.Size([4, 5])
Test passed.
```

---

## 10. 常见错误

```text
1. 忘记 weight.T
2. bias shape 写成 [B,out_features]
3. 使用 * 而不是矩阵乘法 @
4. 忽略无 bias 的情况
```

---

## 11. 扩展任务

```text
1. 支持 x: [B,T,in_features]
2. 写一个 ManualLinear(nn.Module)
3. 和 nn.Linear 拷贝权重后对齐
```

---

## 12. 今日理解问题

```text
1. 为什么 weight 是 [out,in] 而不是 [in,out]？
2. x @ weight.T 的输出 shape 为什么是 [B,out]？
3. bias 为什么可以直接加到 [B,out] 上？
4. F.linear 和 nn.Linear 的关系是什么？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出 shape：
- 最容易错的地方：
- 明天要复习的问题：
```

