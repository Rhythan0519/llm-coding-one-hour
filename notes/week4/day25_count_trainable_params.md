# Day 25：Count Trainable Parameters

## 1. 今日目标

今天实现：

```python
count_parameters(model)
```

目标是统计模型总参数量、可训练参数量和可训练比例。

---

## 2. 这个模块在大模型里有什么用

做 PEFT 时必须知道：

```text
1. 原模型参数量有多少
2. LoRA 参数量有多少
3. 实际参与训练的参数比例是多少
```

这能帮助你判断 adapter 是否真的“参数高效”。

---

## 3. 输入输出

输入：

```text
model: nn.Module
```

输出：

```text
stats: dict
  total: int
  trainable: int
  frozen: int
  trainable_ratio: float
```

---

## 4. 核心逻辑

```text
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable
ratio = trainable / total
```

---

## 5. 伪代码

```text
1. 初始化 total/trainable
2. 遍历 model.parameters()
3. 累加 p.numel()
4. 按 requires_grad 统计 trainable
5. 返回 dict
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn


def count_parameters(model):
    """
    Returns:
        stats: dict with total/trainable/frozen/trainable_ratio
    """
    # TODO 1: count total params
    # TODO 2: count trainable params
    # TODO 3: compute frozen and ratio
    return None


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)


def test():
    model = TinyModel()
    for p in model.a.parameters():
        p.requires_grad = False

    stats = count_parameters(model)

    expected_total = sum(p.numel() for p in model.parameters())
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert stats["total"] == expected_total
    assert stats["trainable"] == expected_trainable
    assert stats["frozen"] == stats["total"] - stats["trainable"]
    assert 0.0 <= stats["trainable_ratio"] <= 1.0

    print("Total params:", stats["total"])
    print("Trainable params:", stats["trainable"])
    print("Trainable ratio:", stats["trainable_ratio"])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. total 等于所有 p.numel() 之和
2. trainable 只统计 requires_grad=True
3. frozen = total - trainable
4. ratio 在 [0,1] 内
```

---

## 8. 运行方式

```bash
python code/week4/day25/count_trainable_params.py
```

---

## 9. 预期输出

```text
Total params: ...
Trainable params: ...
Trainable ratio: ...
Test passed.
```

---

## 10. 常见错误

```text
1. 用 len(p) 而不是 p.numel()
2. 忘记 bias 参数
3. ratio 除以 frozen 而不是 total
4. 没处理 total=0 的极端情况
```

---

## 11. 扩展任务

```text
1. 打印每个参数名、shape、requires_grad
2. 只统计名字里包含 lora_ 的参数
3. 输出百分比格式
```

---

## 12. 今日理解问题

```text
1. p.numel() 统计的是什么？
2. parameters() 和 named_parameters() 有什么区别？
3. 为什么 LoRA 的 trainable ratio 通常很低？
4. 冻结参数还会出现在 state_dict 里吗？
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

