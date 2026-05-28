# Day 28：Mini PEFT Demo

## 1. 今日目标

今天实现一个本周 mini demo：

```python
run_mini_peft_demo()
```

目标是把 LoRA Linear、冻结训练、参数统计、保存加载串起来。

---

## 2. 这个模块在大模型里有什么用

真实 PEFT 流程通常是：

```text
1. 加载 base model
2. 注入 LoRA adapter
3. 冻结 base 参数
4. 只训练 adapter
5. 保存 adapter
6. 推理时重新加载 adapter
```

今天用 toy task 跑通这个完整闭环。

---

## 3. 输入输出

输入：

```text
x: [B, in_features]
labels: [B]
```

输出：

```text
loss_before: scalar
loss_after: scalar
trainable_ratio: float
reload_same: bool
```

---

## 4. 核心逻辑

```text
1. 构造 toy classification 数据
2. 创建 tiny LoRA classifier
3. 统计参数
4. 训练 LoRA 参数
5. 保存 LoRA state
6. 新建同结构模型并加载 LoRA
7. 验证输出一致
```

---

## 5. 伪代码

```text
1. make toy data
2. model = TinyLoRAClassifier
3. stats_before = count_parameters(model)
4. loss_before = CE(model(x), labels)
5. train several steps
6. loss_after = CE(model(x), labels)
7. lora_state = get_lora_state_dict(model)
8. model2 = TinyLoRAClassifier
9. copy base weights from model to model2
10. load_lora_state_dict(model2, lora_state)
11. assert model(x) == model2(x)
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


class TinyLoRAClassifier(nn.Module):
    def __init__(self, in_features, num_classes, rank=2):
        super().__init__()
        # TODO 1: define one LoRALinear classifier

    def forward(self, x):
        # TODO 2: return logits
        return None


def count_parameters(model):
    # TODO 3: reuse Day 25
    return None


def get_lora_state_dict(model):
    # TODO 4: reuse Day 27
    return None


def load_lora_state_dict(model, lora_state):
    # TODO 5: reuse Day 27
    return model


def run_mini_peft_demo():
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    labels = torch.randint(0, 3, (64,))

    model = TinyLoRAClassifier(4, 3, rank=2)

    # TODO 6: compute stats and loss_before
    # TODO 7: train LoRA params only
    # TODO 8: compute loss_after
    # TODO 9: save/load LoRA and compare outputs
    return None


def test():
    result = run_mini_peft_demo()
    loss_before, loss_after, stats, reload_same = result

    assert loss_after < loss_before
    assert stats["trainable_ratio"] < 1.0
    assert reload_same is True

    print("Loss before:", float(loss_before))
    print("Loss after:", float(loss_after))
    print("Trainable ratio:", stats["trainable_ratio"])
    print("Reload same:", reload_same)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 训练后 loss 下降
2. trainable_ratio < 1.0
3. 保存加载后输出一致
4. 脚本能直接运行
```

---

## 8. 运行方式

```bash
python code/week4/day28/mini_peft_demo.py
```

---

## 9. 预期输出

```text
Loss before: ...
Loss after: ...
Trainable ratio: ...
Reload same: True
Test passed.
```

---

## 10. 常见错误

```text
1. 训练时 base 参数没有冻结
2. 保存加载时 base 不一致导致输出不同
3. loss 没下降但测试阈值写太死
4. labels shape 写成 [B,1] 导致 cross_entropy 报错
```

---

## 11. 扩展任务

```text
1. 比较 rank=1/2/4 的训练效果
2. 打印每个参数的 requires_grad
3. 把 LoRA classifier 换成 tiny MLP
```

---

## 12. 今日理解问题

```text
1. 一个完整 PEFT 训练闭环包含哪些步骤？
2. 为什么 reload 测试前要保证 base 权重一致？
3. trainable_ratio 为什么应该小于 1？
4. toy task 的 loss 下降能证明什么，不能证明什么？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出 shape：
- 最容易错的地方：
- 下周要复习的问题：
```

