# Day 19：Repetition Penalty

## 1. 今日目标

今天实现 repetition penalty：

```python
apply_repetition_penalty(logits, history, penalty)
```

目标是降低已经生成过的 token 再次出现的倾向。

---

## 2. 这个模块在大模型里有什么用

语言模型生成时容易重复，例如：

```text
hello hello hello hello
```

repetition penalty 会在 softmax 之前调整历史 token 的 logits，让模型不那么容易重复同一个 token。

---

## 3. 输入输出

### 输入

```text
logits: [B, V]
history: [B, T]
penalty: float, usually > 1
```

### 输出

```text
adjusted_logits: [B, V]
```

---

## 4. 核心逻辑

常见做法：

```text
如果某个历史 token 的 logit > 0，则除以 penalty
如果某个历史 token 的 logit < 0，则乘以 penalty
```

这样会降低它被 softmax 选中的相对概率。

---

## 5. 伪代码

```text
1. clone logits，避免原地改输入
2. 遍历 batch
3. 找到该样本 history 中出现过的 token id
4. 对这些 token 的 logits 应用 penalty
5. 返回 adjusted_logits
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits, history, penalty=1.2):
    """
    Args:
        logits: [B, V]
        history: [B, T]
        penalty: float >= 1

    Returns:
        adjusted_logits: [B, V]
    """
    assert penalty >= 1.0

    # TODO 1: clone logits
    adjusted = None

    # TODO 2: for each batch item, find unique history token ids
    # TODO 3: apply penalty to those logits

    return adjusted


def test():
    logits = torch.tensor([
        [4.0, 2.0, -1.0, 0.5],
        [1.0, -2.0, 3.0, 0.1],
    ])
    history = torch.tensor([
        [0, 2, 2],
        [1, 1, 3],
    ])

    adjusted = apply_repetition_penalty(logits, history, penalty=2.0)

    assert adjusted.shape == logits.shape
    assert torch.allclose(logits, torch.tensor([
        [4.0, 2.0, -1.0, 0.5],
        [1.0, -2.0, 3.0, 0.1],
    ]))
    assert torch.allclose(adjusted[0, 0], torch.tensor(2.0))
    assert torch.allclose(adjusted[0, 2], torch.tensor(-2.0))
    assert torch.allclose(adjusted[0, 1], logits[0, 1])
    assert torch.allclose(adjusted[1, 1], torch.tensor(-4.0))
    assert torch.allclose(adjusted[1, 3], torch.tensor(0.05))
    assert not torch.isnan(adjusted).any()

    probs_before = F.softmax(logits, dim=-1)
    probs_after = F.softmax(adjusted, dim=-1)
    assert probs_after[0, 0] < probs_before[0, 0]

    print("Input logits shape:", logits.shape)
    print("History shape:", history.shape)
    print("Output shape:", adjusted.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. adjusted.shape == logits.shape
2. 原 logits 不被原地修改
3. 历史 token 的正 logits 被除以 penalty
4. 历史 token 的负 logits 被乘以 penalty
5. 非历史 token 保持不变
```

---

## 8. 运行方式

保存为：

```text
day19_repetition_penalty.py
```

运行：

```bash
python day19_repetition_penalty.py
```

---

## 9. 预期输出

```text
Input logits shape: torch.Size([2, 4])
History shape: torch.Size([2, 3])
Output shape: torch.Size([2, 4])
Test passed.
```

---

## 10. 常见错误

```text
1. 直接修改 logits，导致输入被污染
2. 对重复 token 应用多次 penalty，而不是 unique 后应用一次
3. 正负 logit 使用同一种处理
4. penalty 小于 1 导致反而鼓励重复
5. 在 softmax 之后才改概率
```

---

## 11. 扩展任务

```text
1. 只惩罚最近 N 个 token
2. 组合 repetition penalty + top-k
3. 支持 presence penalty
4. 支持 frequency penalty
5. 接入 Week2 tiny GPT generate
```

---

## 12. 今日理解问题

```text
1. repetition penalty 为什么要作用在 logits 上？
2. 为什么正 logit 和负 logit 处理方式不同？
3. 为什么要避免原地修改 logits？
4. 如果 penalty=1，会发生什么？
5. repetition penalty 和 top-k/top-p 是同一类操作吗？
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

