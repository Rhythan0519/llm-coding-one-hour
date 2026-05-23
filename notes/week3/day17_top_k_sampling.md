# Day 17：Top-k Sampling

## 1. 今日目标

今天实现 top-k sampling：

```python
top_k_filter(logits, k)
sample_top_k(logits, k, temperature=1.0)
```

目标是让模型只从概率最高的 `k` 个 token 里采样。

---

## 2. 这个模块在大模型里有什么用

top-k sampling 是常见生成策略。

它会先保留 logits 最大的 `k` 个 token，其他 token 的 logits 设成很小的值。

这样可以：

```text
1. 避免从极低概率 token 中采样
2. 保留一定随机性
3. 比 greedy 更灵活
```

---

## 3. 输入输出

### 输入

```text
logits: [B, V]
k: int
temperature: float
```

### 输出

```text
filtered_logits: [B, V]
next_ids: [B]
```

---

## 4. 核心逻辑

```text
1. 找到每一行 top-k logits
2. 第 k 大的值作为阈值
3. 小于阈值的位置设成 -inf 或 -1e9
4. softmax
5. multinomial 采样
```

---

## 5. 伪代码

```text
1. top_values, top_indices = torch.topk(logits, k, dim=-1)
2. threshold = top_values[:, -1:]
3. mask = logits < threshold
4. filtered_logits = logits.masked_fill(mask, -1e9)
5. probs = softmax(filtered_logits / temperature)
6. next_ids = multinomial(probs)
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def top_k_filter(logits, k, mask_value=-1e9):
    """
    Args:
        logits: [B, V]
        k: int

    Returns:
        filtered_logits: [B, V]
    """
    B, V = logits.shape
    assert 1 <= k <= V

    # TODO 1: get top-k values
    top_values = None

    # TODO 2: get kth value as threshold [B, 1]
    threshold = None

    # TODO 3: mask logits smaller than threshold
    filtered_logits = None

    return filtered_logits


def sample_top_k(logits, k, temperature=1.0):
    """
    Args:
        logits: [B, V]

    Returns:
        next_ids: [B]
        probs: [B, V]
    """
    assert temperature > 0

    # TODO 4: filter logits
    filtered_logits = None

    # TODO 5: softmax with temperature
    probs = None

    # TODO 6: sample next ids
    next_ids = None

    return next_ids, probs


def test():
    torch.manual_seed(0)
    logits = torch.tensor([
        [0.1, 5.0, 1.0, 4.0, 0.0],
        [3.0, 0.2, 2.0, 0.1, 1.0],
    ])

    filtered = top_k_filter(logits, k=2)
    ids, probs = sample_top_k(logits, k=2)

    allowed = torch.topk(logits, k=2, dim=-1).indices
    allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
    allowed_mask.scatter_(dim=-1, index=allowed, value=True)

    assert filtered.shape == logits.shape
    assert probs.shape == logits.shape
    assert ids.shape == (2,)
    assert torch.all(filtered[~allowed_mask] < -1e8)
    assert torch.all(probs[~allowed_mask] < 1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)

    for b, token_id in enumerate(ids):
        assert allowed_mask[b, token_id]

    print("Input shape:", logits.shape)
    print("Filtered logits shape:", filtered.shape)
    print("Sampled ids shape:", ids.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. filtered_logits.shape == logits.shape
2. 非 top-k token 被 mask
3. softmax 后非 top-k token 概率接近 0
4. sampled ids 一定属于 top-k
5. probs 每行和为 1
```

---

## 8. 运行方式

保存为：

```text
day17_top_k_sampling.py
```

运行：

```bash
python day17_top_k_sampling.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 5])
Filtered logits shape: torch.Size([2, 5])
Sampled ids shape: torch.Size([2])
Test passed.
```

---

## 10. 常见错误

```text
1. topk 的 dim 写错
2. threshold shape 没 keep 成 [B,1]
3. mask 方向写反，把 top-k 反而 mask 掉
4. 直接对原 logits softmax，没有用 filtered_logits
5. k 大于 vocab size 没检查
```

---

## 11. 扩展任务

```text
1. 支持 k=1，此时接近 greedy
2. 支持 logits [B,T,V]
3. 对比不同 k 下采样多样性
4. 接入 Week2 tiny GPT
5. 组合 temperature + top-k
```

---

## 12. 今日理解问题

```text
1. top-k sampling 和 greedy 的区别是什么？
2. k=1 时会发生什么？
3. 为什么要在 softmax 前 mask？
4. top-k 的候选集大小是否固定？
5. top-k 可能有什么缺点？
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

