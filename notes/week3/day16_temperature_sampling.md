# Day 16：Temperature Sampling

## 1. 今日目标

今天实现 temperature sampling：

```python
apply_temperature(logits, temperature)
sample_with_temperature(logits, temperature)
```

目标是理解 temperature 如何控制生成分布的随机性。

---

## 2. 这个模块在大模型里有什么用

temperature 是 LLM 推理时最常见的参数之一。

直觉：

```text
temperature < 1：分布更尖锐，更保守
temperature = 1：保持原分布
temperature > 1：分布更平，更随机
```

---

## 3. 输入输出

### 输入

```text
logits: [B, V]
temperature: float
```

### 输出

```text
probs: [B, V]
next_ids: [B]
```

---

## 4. 核心公式

先缩放 logits：

$$
scaled\_logits = logits / temperature
$$

再 softmax：

$$
probs = softmax(scaled\_logits)
$$

---

## 5. 伪代码

```text
1. 检查 temperature > 0
2. scaled_logits = logits / temperature
3. probs = softmax(scaled_logits)
4. 用 torch.multinomial 从 probs 采样
5. 返回 next_ids 和 probs
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature):
    """
    Args:
        logits: [B, V]
        temperature: positive float

    Returns:
        probs: [B, V]
    """
    assert temperature > 0

    # TODO 1: divide logits by temperature
    scaled_logits = None

    # TODO 2: softmax over vocab dimension
    probs = None

    return probs


def sample_with_temperature(logits, temperature):
    """
    Args:
        logits: [B, V]

    Returns:
        next_ids: [B]
        probs: [B, V]
    """
    # TODO 3: get probability distribution
    probs = None

    # TODO 4: sample one token for each batch item
    next_ids = None

    return next_ids, probs


def entropy(probs):
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def test():
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.0]])

    probs_low = apply_temperature(logits, temperature=0.5)
    probs_high = apply_temperature(logits, temperature=2.0)
    ids, probs = sample_with_temperature(logits.repeat(4, 1), temperature=1.0)

    assert probs_low.shape == logits.shape
    assert probs_high.shape == logits.shape
    assert ids.shape == (4,)
    assert torch.allclose(probs_low.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(probs_high.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert entropy(probs_high) > entropy(probs_low)
    assert not torch.isnan(probs).any()
    assert ids.dtype == torch.long

    print("Input shape:", logits.shape)
    print("Output probs shape:", probs.shape)
    print("Sampled ids shape:", ids.shape)
    print("Low temp entropy:", round(entropy(probs_low).item(), 4))
    print("High temp entropy:", round(entropy(probs_high).item(), 4))
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. probs.shape == logits.shape
2. probs 每行求和为 1
3. temperature 越高 entropy 越大
4. sampled ids shape 是 [B]
5. probs 没有 NaN
```

---

## 8. 运行方式

保存为：

```text
day16_temperature_sampling.py
```

运行：

```bash
python day16_temperature_sampling.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([1, 4])
Output probs shape: torch.Size([4, 4])
Sampled ids shape: torch.Size([4])
Low temp entropy: ...
High temp entropy: ...
Test passed.
```

---

## 10. 常见错误

```text
1. 写成 logits * temperature，而不是 logits / temperature
2. temperature 为 0 时没有检查
3. softmax dim 写错
4. torch.multinomial 输入不是概率
5. next_ids 返回 [B,1] 后没有 squeeze
```

---

## 11. 扩展任务

```text
1. 支持 logits [B,T,V]，只采最后一个位置
2. temperature 很小时退化成接近 greedy
3. 比较不同 temperature 下采样 100 次的频率
4. 接入 Week2 tiny GPT 的 generate
5. 加入 temperature=1 的对齐测试
```

---

## 12. 今日理解问题

```text
1. temperature 为什么是除法？
2. temperature 越大，为什么分布越平？
3. entropy 可以衡量什么？
4. sampling 和 greedy 的区别是什么？
5. 为什么 torch.multinomial 的输入必须是概率？
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

