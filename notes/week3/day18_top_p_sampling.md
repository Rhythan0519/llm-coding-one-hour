# Day 18：Top-p Sampling

## 1. 今日目标

今天实现 top-p sampling，也叫 nucleus sampling：

```python
top_p_filter(logits, p)
sample_top_p(logits, p, temperature=1.0)
```

目标是理解按累计概率动态选择候选 token 的生成策略。

---

## 2. 这个模块在大模型里有什么用

top-p 不固定保留 token 数量，而是保留累计概率达到 `p` 的最小候选集。

直觉：

```text
分布很尖时：可能只保留少数 token
分布很平时：会保留更多 token
```

它比 top-k 更自适应。

---

## 3. 输入输出

### 输入

```text
logits: [B, V]
p: float, 0 < p <= 1
temperature: float
```

### 输出

```text
filtered_logits: [B, V]
next_ids: [B]
probs: [B, V]
```

---

## 4. 核心逻辑

```text
1. 对 logits 做 softmax 得到 probs
2. 按概率从大到小排序
3. 计算累计概率 cumulative_probs
4. 保留累计概率 <= p 的 token
5. 为了至少有一个越界 token，通常也保留第一个让累计概率超过 p 的 token
6. 把其他 token mask 掉
7. softmax 后采样
```

---

## 5. 伪代码

```text
1. sorted_logits, sorted_indices = sort(logits, descending=True)
2. sorted_probs = softmax(sorted_logits)
3. cumulative_probs = cumsum(sorted_probs)
4. sorted_remove_mask = cumulative_probs > p
5. shift mask right so the first token above p is still kept
6. scatter sorted mask back to original vocab order
7. masked_fill removed positions
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def top_p_filter(logits, p, mask_value=-1e9):
    """
    Args:
        logits: [B, V]
        p: float, 0 < p <= 1

    Returns:
        filtered_logits: [B, V]
        keep_mask: [B, V], True means kept
    """
    assert 0 < p <= 1

    # TODO 1: sort logits descending
    sorted_logits = None
    sorted_indices = None

    # TODO 2: compute sorted probabilities
    sorted_probs = None

    # TODO 3: cumulative probabilities
    cumulative_probs = None

    # TODO 4: create remove mask in sorted order
    sorted_remove_mask = None

    # TODO 5: shift mask so the first token above p is kept
    sorted_remove_mask = None

    # TODO 6: scatter remove mask back to original order
    remove_mask = None

    # TODO 7: mask removed logits
    filtered_logits = None
    keep_mask = None

    return filtered_logits, keep_mask


def sample_top_p(logits, p, temperature=1.0):
    """
    Args:
        logits: [B, V]

    Returns:
        next_ids: [B]
        probs: [B, V]
        keep_mask: [B, V]
    """
    assert temperature > 0

    # TODO 8: filter logits with top-p
    filtered_logits, keep_mask = None

    # TODO 9: softmax after temperature
    probs = None

    # TODO 10: sample next ids
    next_ids = None

    return next_ids, probs, keep_mask


def test():
    torch.manual_seed(0)
    logits = torch.tensor([
        [5.0, 4.0, 1.0, 0.5, 0.0],
        [2.0, 1.9, 1.8, 1.7, 0.1],
    ])

    filtered, keep_mask = top_p_filter(logits, p=0.8)
    ids, probs, keep_mask2 = sample_top_p(logits, p=0.8)

    assert filtered.shape == logits.shape
    assert keep_mask.shape == logits.shape
    assert probs.shape == logits.shape
    assert ids.shape == (2,)
    assert torch.equal(keep_mask, keep_mask2)
    assert torch.all(filtered[~keep_mask] < -1e8)
    assert torch.all(probs[~keep_mask] < 1e-6)
    assert torch.all(keep_mask.sum(dim=-1) >= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)

    for b, token_id in enumerate(ids):
        assert keep_mask[b, token_id]

    print("Input shape:", logits.shape)
    print("Filtered logits shape:", filtered.shape)
    print("Sampled ids shape:", ids.shape)
    print("Kept counts:", keep_mask.sum(dim=-1).tolist())
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. filtered_logits.shape == logits.shape
2. 至少保留一个 token
3. 被移除 token 的概率接近 0
4. sampled ids 一定属于 keep_mask
5. probs 每行和为 1
```

---

## 8. 运行方式

保存为：

```text
day18_top_p_sampling.py
```

运行：

```bash
python day18_top_p_sampling.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 5])
Filtered logits shape: torch.Size([2, 5])
Sampled ids shape: torch.Size([2])
Kept counts: [...]
Test passed.
```

---

## 10. 常见错误

```text
1. 没有先排序，直接 cumsum 原始 vocab 顺序
2. 忘记把 sorted mask scatter 回原始顺序
3. 没保留第一个超过 p 的 token，导致候选集太小
4. 在 mask 前后 softmax 顺序混乱
5. p 的范围没有检查
```

---

## 11. 扩展任务

```text
1. 支持 min_tokens_to_keep
2. 对比 top-k 和 top-p 的候选集大小
3. 组合 temperature + top-p
4. 接入 Week2 tiny GPT
5. 多次采样统计 token 频率
```

---

## 12. 今日理解问题

```text
1. top-p 和 top-k 最大区别是什么？
2. 为什么 top-p 需要排序？
3. 为什么要保留第一个超过 p 的 token？
4. 分布很尖时 top-p 会保留更多还是更少 token？
5. top-p 中的 p 是概率阈值还是 token 数量？
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

