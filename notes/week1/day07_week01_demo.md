# Day 7：Week 1 Tiny Classifier Demo

## 1. 今日目标

今天做一个本周 mini demo：

```python
tiny_classifier_demo()
```

目标是把本周的基础模块串成一个能训练、能预测、能看结果的小闭环。

---

## 2. 这个模块在大模型里有什么用

这个 demo 训练的不是大模型本身，而是训练习惯：

```text
数据 -> 模型 -> loss -> backward -> update -> 评估
```

---

## 3. 输入输出

### 输入

```text
toy dataset: [B, 2]
```

### 输出

```text
prediction: [B]
loss curve
accuracy
```

---

## 4. 核心逻辑

用一个小 MLP 在二维点上做分类，观察 loss 是否下降、accuracy 是否高于随机猜测。

---

## 5. 伪代码

```text
1. 生成 toy data
2. 初始化小 MLP
3. 训练若干步
4. 打印 loss 和 accuracy
5. 返回预测结果
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_toy_data(n=256):
    # TODO: create two Gaussian clusters
    x = None
    y = None
    return x, y


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_one_demo():
    torch.manual_seed(0)
    x, y = make_toy_data()
    model = TinyClassifier()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []
    for _ in range(100):
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        acc = (pred == y).float().mean().item()

    return losses, acc, pred


def test():
    losses, acc, pred = train_one_demo()

    assert len(losses) == 100
    assert losses[-1] < losses[0]
    assert acc > 0.8

    print("Input shape:", torch.Size([256, 2]))
    print("Output shape:", pred.shape)
    print("Final accuracy:", round(acc, 4))
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. loss 下降
2. accuracy 高于随机猜测
3. 脚本直接可跑
4. 没有 NaN
```

---

## 8. 运行方式

```text
day07_week01_demo.py
python day07_week01_demo.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([256, 2])
Output shape: torch.Size([256])
Final accuracy: 0.9xxx
Test passed.
```

---

## 10. 常见错误

```text
1. 数据生成太难
2. 学习率太大或太小
3. 忘记切到 eval / no_grad
4. accuracy 算错
5. loss 降了但分类边界没学到
```

---

## 11. 扩展任务

```text
1. 画出分类边界
2. 试试 Adam
3. 试试不同隐藏层宽度
4. 换成三分类
5. 把本周前几天的模块复用进来
```

---

## 12. 今日理解问题

```text
1. 为什么这个 demo 能验证你真的学会了基础模块？
2. loss 下降和 accuracy 变化一定一致吗？
3. 为什么 toy data 比真实数据更适合第一周？
4. 哪一步最容易出 bug？
5. 如果重新写一遍，你会先写哪一层？
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
- 下周要复习：
```

