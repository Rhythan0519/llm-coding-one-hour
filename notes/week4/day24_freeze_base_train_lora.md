# Day 24：Freeze Base, Train LoRA

## 1. 今日目标

今天实现一个最小训练实验：

```python
train_lora_on_toy_data()
```

目标是验证 base 参数冻结，只有 LoRA 参数参与训练，并且 toy loss 能下降。

---

## 2. 这个模块在大模型里有什么用

PEFT 训练的关键不是“模型能 forward”，而是：

```text
1. base model 参数不更新
2. adapter 参数更新
3. optimizer 只拿到可训练参数
4. loss.backward 后梯度只出现在 adapter 上
```

---

## 3. 输入输出

输入：

```text
x: [B, in_features]
target: [B, out_features]
```

输出：

```text
loss_before: scalar
loss_after: scalar
```

---

## 4. 核心逻辑

```text
1. 构造 toy regression 数据
2. 创建 LoRALinear
3. 冻结 base
4. optimizer 只接收 requires_grad=True 的参数
5. 训练若干步
6. 检查 loss 下降和梯度位置
```

---

## 5. 伪代码

```text
1. x = random tensor
2. target = x @ true_weight.T
3. layer = LoRALinear(...)
4. trainable_params = filter(lambda p: p.requires_grad, layer.parameters())
5. loop:
      pred = layer(x)
      loss = mse(pred, target)
      backward + step
6. assert base grad is None
7. assert lora grad is not None
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    # TODO: reuse Day 23 implementation
    pass


def train_lora_on_toy_data():
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    target = torch.randn(32, 6)

    layer = LoRALinear(4, 6, rank=2, alpha=4.0)

    # TODO 1: create optimizer with trainable params only
    # TODO 2: compute loss_before
    # TODO 3: train for several steps
    # TODO 4: compute loss_after
    # TODO 5: return loss_before, loss_after, layer
    return None


def test():
    result = train_lora_on_toy_data()
    loss_before, loss_after, layer = result

    assert loss_after < loss_before
    assert layer.base.weight.requires_grad is False
    assert layer.base.weight.grad is None

    lora_grads = [
        p.grad for name, p in layer.named_parameters()
        if "lora_" in name and p.requires_grad
    ]
    assert all(g is not None for g in lora_grads)

    print("Loss before:", float(loss_before))
    print("Loss after:", float(loss_after))
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. loss_after < loss_before
2. base.weight.requires_grad is False
3. base.weight.grad is None
4. LoRA 参数 grad 不为 None
```

---

## 8. 运行方式

```bash
python code/week4/day24/freeze_base_train_lora.py
```

---

## 9. 预期输出

```text
Loss before: ...
Loss after: ...
Test passed.
```

---

## 10. 常见错误

```text
1. optimizer 把冻结参数也传进去了
2. 忘记 optimizer.zero_grad()
3. 忘记 loss.backward()
4. 用 .detach() 把 LoRA 分支梯度断掉
```

---

## 11. 扩展任务

```text
1. 打印每个参数的 grad norm
2. 比较 rank=1 和 rank=4 的拟合能力
3. 加一个分类 toy task
```

---

## 12. 今日理解问题

```text
1. requires_grad=False 会影响 forward 吗？
2. optimizer 里包含冻结参数会怎样？
3. 怎么检查某个参数真的没有被训练？
4. 为什么 LoRA 可以只训练少量参数？
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

