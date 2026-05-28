# Day 23：LoRA Linear

## 1. 今日目标

今天实现：

```python
class LoRALinear(nn.Module)
```

目标是在冻结的 base linear 外面，加一个可训练的低秩更新分支。

---

## 2. 这个模块在大模型里有什么用

LoRA 用少量参数适配大模型：

```text
base 输出: x @ W.T
LoRA 输出: scale * (x @ A.T @ B.T)
最终输出: base + LoRA
```

它常用于 q_proj、v_proj、FFN projection 等线性层。

---

## 3. 输入输出

输入：

```text
x: [B, in_features]
```

输出：

```text
y: [B, out_features]
```

参数：

```text
base.weight: [out_features, in_features]
lora_A: [rank, in_features]
lora_B: [out_features, rank]
scale = alpha / rank
```

---

## 4. 核心公式

```text
base = F.linear(x, weight, bias)
lora = F.linear(F.linear(x, lora_A), lora_B) * scale
y = base + lora
```

---

## 5. 伪代码

```text
1. 创建 base nn.Linear
2. 冻结 base 参数
3. 创建 lora_A 和 lora_B
4. forward 时先算 base_out
5. 再算 lora_out
6. 两者相加返回
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=2, alpha=1.0, bias=True):
        super().__init__()
        assert rank >= 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features, bias=bias)

        # TODO 1: freeze base parameters
        # TODO 2: if rank > 0, create lora_A and lora_B
        # TODO 3: initialize A small random, B zeros

    def forward(self, x):
        # TODO 4: compute base output
        # TODO 5: if rank == 0, return base output
        # TODO 6: compute lora output and add it
        return None


def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    layer = LoRALinear(3, 5, rank=2, alpha=4.0)
    y = layer(x)

    assert y.shape == (4, 5)
    assert layer.base.weight.requires_grad is False

    rank0 = LoRALinear(3, 5, rank=0)
    y0 = rank0(x)
    ref0 = rank0.base(x)
    assert torch.allclose(y0, ref0, atol=1e-6)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 输出 shape 不变
2. base 参数 requires_grad=False
3. rank=0 时输出等于 base linear
4. rank>0 时 LoRA 参数 requires_grad=True
```

---

## 8. 运行方式

```bash
python code/week4/day23/lora_linear.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([4, 3])
Output shape: torch.Size([4, 5])
Test passed.
```

---

## 10. 常见错误

```text
1. A/B shape 写反
2. 忘记冻结 base
3. B 没初始化为 0，导致初始输出改变太大
4. scale 写成 rank / alpha
```

---

## 11. 扩展任务

```text
1. 支持 x: [B,T,C]
2. 加 dropout
3. 写 merge_lora_weight() 计算 W + scale * B @ A
```

---

## 12. 今日理解问题

```text
1. lora_A 和 lora_B 分别是什么 shape？
2. 为什么低秩矩阵能减少参数量？
3. 为什么常把 lora_B 初始化为 0？
4. alpha / rank 的作用是什么？
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

