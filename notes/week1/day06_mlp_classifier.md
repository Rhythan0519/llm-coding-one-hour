# Day 6：Simple MLP

## 1. 今日目标

今天实现一个两层 MLP：

```python
MyMLP(in_dim, hidden_dim, out_dim)
```

目标是理解非线性激活为什么重要。

---

## 2. 这个模块在大模型里有什么用

Transformer 里的 FFN 本质上就是一个带激活的 MLP，只是维度更大。

---

## 3. 输入输出

### 输入

```text
features: [B, D]
```

### 输出

```text
logits: [B, num_classes]
```

---

## 4. 核心逻辑

```text
x -> linear -> ReLU -> linear -> logits
```

---

## 5. 伪代码

```text
1. 第一层线性变换
2. 非线性激活
3. 第二层线性变换
4. 返回输出
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # TODO 1: first linear layer
        h = None

        # TODO 2: activation
        h = None

        # TODO 3: second linear layer
        logits = None
        return logits


def make_xor_data():
    x = torch.tensor([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
    ])
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    return x, y


def test():
    torch.manual_seed(0)
    x, y = make_xor_data()
    model = MyMLP(2, 8, 2)

    logits = model(x)
    assert logits.shape == (4, 2)

    loss_before = F.cross_entropy(logits, y)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    loss_after = F.cross_entropy(model(x), y)

    assert loss_after < loss_before
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. shape 正确
2. backward 可以跑通
3. XOR loss 能下降
4. 没有 NaN
```

---

## 8. 运行方式

```text
day06_mlp_classifier.py
python day06_mlp_classifier.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([4, 2])
Output shape: torch.Size([4, 2])
Test passed.
```

---

## 10. 常见错误

```text
1. 忘了加非线性
2. hidden_dim 太小
3. 学习率太大
4. loss.backward() 前忘记 zero_grad
5. 误把 XOR 当线性可分
```

---

## 11. 扩展任务

```text
1. 改成三层 MLP
2. 加 dropout
3. 换 GELU
4. 看梯度是否正常
5. 试着手写一个 FFN 版本
```

---

## 12. 今日理解问题

```text
1. 为什么 MLP 比 linear 更强？
2. ReLU 的作用是什么？
3. hidden_dim 决定了什么？
4. 为什么 XOR 需要非线性？
5. FFN 和这里的 MLP 有什么关系？
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

