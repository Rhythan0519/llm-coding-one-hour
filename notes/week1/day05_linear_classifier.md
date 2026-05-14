# Day 5：Linear Classifier

## 1. 今日目标

今天实现一个线性分类器：

```python
LinearClassifier(in_dim, num_classes)
```

目标是理解最简单的分类头如何从特征映射到 logits。

---

## 2. 这个模块在大模型里有什么用

很多模型最后都会接一个 linear head，把 hidden state 投影到类别数或词表大小。

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

$$
y = xW^T + b
$$

---

## 5. 伪代码

```text
1. 初始化 weight 和 bias
2. x 乘以 W 转置
3. 加 bias
4. 返回 logits
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        # TODO: compute logits = x @ weight.T + bias
        logits = None
        return logits


def make_toy_data(n=64):
    torch.manual_seed(0)
    x0 = torch.randn(n // 2, 2) + torch.tensor([-2.0, -2.0])
    x1 = torch.randn(n // 2, 2) + torch.tensor([2.0, 2.0])
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([
        torch.zeros(n // 2, dtype=torch.long),
        torch.ones(n // 2, dtype=torch.long),
    ])
    return x, y


def test():
    torch.manual_seed(0)
    x, y = make_toy_data()
    mine = LinearClassifier(2, 2)
    ref = nn.Linear(2, 2)
    ref.weight.data.copy_(mine.weight.data)
    ref.bias.data.copy_(mine.bias.data)

    my_logits = mine(x)
    ref_logits = ref(x)

    assert my_logits.shape == (64, 2)
    assert torch.allclose(my_logits, ref_logits, atol=1e-6)

    loss_before = F.cross_entropy(my_logits, y)
    opt = torch.optim.SGD(mine.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        logits = mine(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    loss_after = F.cross_entropy(mine(x), y)

    assert loss_after < loss_before
    print("Input shape:", x.shape)
    print("Output shape:", my_logits.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. shape 正确
2. 对齐 nn.Linear
3. 训练若干步后 loss 下降
4. 没有 NaN
```

---

## 8. 运行方式

```text
day05_linear_classifier.py
python day05_linear_classifier.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([64, 2])
Output shape: torch.Size([64, 2])
Test passed.
```

---

## 10. 常见错误

```text
1. weight 乘法维度错
2. bias broadcast 错
3. label dtype 不对
4. loss 不下降时没检查学习率
5. 数据集太难
```

---

## 11. 扩展任务

```text
1. 试试 3 类分类
2. 可视化决策边界
3. 加入 weight decay
4. 比较不同初始化
5. 换成线性回归版本
```

---

## 12. 今日理解问题

```text
1. linear classifier 的参数是什么？
2. logits 为什么不是概率？
3. 为什么训练数据要足够简单？
4. bias 起什么作用？
5. 这个 head 在 LLM 里对应哪一层？
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

