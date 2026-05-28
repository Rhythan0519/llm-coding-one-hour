# Day 27：Save / Load LoRA

## 1. 今日目标

今天实现：

```python
get_lora_state_dict(model)
load_lora_state_dict(model, lora_state)
```

目标是只保存 LoRA 参数，并验证加载后输出一致。

---

## 2. 这个模块在大模型里有什么用

PEFT adapter 的核心特点是：

```text
1. base model 很大，不重复保存
2. adapter 很小，只保存 LoRA 权重
3. 加载时先加载 base，再加载 adapter
```

---

## 3. 输入输出

输入：

```text
model: nn.Module
lora_state: dict[str, Tensor]
```

输出：

```text
lora_state: 只包含名字里有 "lora_" 的参数
```

---

## 4. 核心逻辑

```text
save:
    遍历 model.state_dict()
    只保留 key 包含 "lora_" 的 tensor

load:
    取 model.state_dict()
    用 lora_state 更新对应 key
    model.load_state_dict(updated_state)
```

---

## 5. 伪代码

```text
1. 创建 model1
2. 修改 model1 的 LoRA 参数
3. 保存 lora_state
4. 创建 model2
5. 拷贝 base 参数，使两个 base 一样
6. 加载 lora_state 到 model2
7. 比较 model1(x) 和 model2(x)
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    # TODO: reuse Day 23 implementation
    pass


def get_lora_state_dict(model):
    """
    Returns:
        dict containing only LoRA tensors
    """
    # TODO 1: filter model.state_dict()
    return None


def load_lora_state_dict(model, lora_state):
    """
    Load LoRA tensors into model.
    """
    # TODO 2: update current state_dict with lora_state
    # TODO 3: call load_state_dict
    return model


def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3)

    model1 = LoRALinear(3, 5, rank=2, alpha=4.0)
    model2 = LoRALinear(3, 5, rank=2, alpha=4.0)

    # Make base identical so the only difference is LoRA.
    model2.base.load_state_dict(model1.base.state_dict())

    # TODO 4: manually change model1 LoRA params
    lora_state = get_lora_state_dict(model1)
    load_lora_state_dict(model2, lora_state)

    y1 = model1(x)
    y2 = model2(x)

    assert len(lora_state) > 0
    assert all("lora_" in key for key in lora_state.keys())
    assert torch.allclose(y1, y2, atol=1e-6)

    print("Input shape:", x.shape)
    print("Num LoRA tensors:", len(lora_state))
    print("Output shape:", y1.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. lora_state 非空
2. lora_state 的 key 都包含 lora_
3. 加载后两个模型输出一致
4. 输出 shape 正确
```

---

## 8. 运行方式

```bash
python code/week4/day27/save_load_lora.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([4, 3])
Num LoRA tensors: ...
Output shape: torch.Size([4, 5])
Test passed.
```

---

## 10. 常见错误

```text
1. 保存了 base 参数，失去 PEFT 意义
2. 加载前两个 base 不一致，导致输出无法对齐
3. 直接改 state_dict 但没 load_state_dict
4. 忘记 clone tensor，后续被原模型改动影响
```

---

## 11. 扩展任务

```text
1. 用 torch.save / torch.load 保存到文件
2. 支持 strict=False 加载
3. 打印每个 LoRA tensor 的 shape
```

---

## 12. 今日理解问题

```text
1. state_dict 和 parameters() 有什么区别？
2. 为什么只保存 LoRA 权重也能复用 adapter？
3. 为什么测试里需要让两个模型 base 权重一致？
4. load_state_dict 会不会改变 requires_grad？
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

