# Day 9：Causal Mask

## 1. 今日目标

今天实现一个 causal mask 函数：

```python
apply_causal_mask(scores)
```

目标是让第 `t` 个位置只能看见自己和之前的位置，不能看到未来 token。

---

## 2. 这个模块在大模型里有什么用

GPT 这类自回归语言模型训练时，会一次性处理整段序列，但预测第 `t` 个 token 时不能偷看 `t+1` 之后的答案。

causal mask 的作用是：

```text
1. 阻止未来 token 信息泄漏
2. 保持训练目标和推理过程一致
3. 让 attention 只关注当前位置及以前的位置
```

---

## 3. 输入输出

### 输入

```text
scores: [B, T, T]
```

`scores[b, i, j]` 表示第 `i` 个 query token 对第 `j` 个 key token 的原始 attention score。

### 输出

```text
masked_scores: [B, T, T]
```

未来位置应该被填成一个很小的数，例如 `-inf` 或 `-1e9`。

---

## 4. 核心逻辑

保留下三角，包括对角线：

```text
允许看见：
row 0: col 0
row 1: col 0, 1
row 2: col 0, 1, 2
```

未来位置：

```text
col > row
```

应该被 mask。

---

## 5. 伪代码

```text
1. 读取 scores 的 T
2. 创建 [T,T] 下三角布尔矩阵
3. 把下三角为 False 的位置填成 mask_value
4. 返回 masked_scores
5. 对 masked_scores 做 softmax 后，未来位置概率应接近 0
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def apply_causal_mask(scores, mask_value=-1e9):
    """
    Args:
        scores: [B, T, T]

    Returns:
        masked_scores: [B, T, T]
    """
    B, T, _ = scores.shape

    # TODO 1: create lower-triangular mask [T, T], dtype should be bool
    mask = None

    # TODO 2: broadcast mask to scores and fill future positions
    masked_scores = None

    return masked_scores


def test():
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 4)
    masked_scores = apply_causal_mask(scores)
    probs = F.softmax(masked_scores, dim=-1)

    assert masked_scores.shape == scores.shape
    assert torch.allclose(masked_scores[:, 1, 0], scores[:, 1, 0])
    assert torch.all(masked_scores[:, 0, 1:] < -1e8)
    assert torch.all(probs[:, 0, 1:] < 1e-6)
    assert not torch.isnan(probs).any()

    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.all(probs[:, future_mask] < 1e-6)

    print("Input shape:", scores.shape)
    print("Output shape:", masked_scores.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. masked_scores.shape == scores.shape
2. 允许看到的位置保持原值
3. 未来位置被填成很小的数
4. softmax 后未来位置概率接近 0
5. probs 没有 NaN
```

---

## 8. 运行方式

保存为：

```text
day09_causal_mask.py
```

运行：

```bash
python day09_causal_mask.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 4, 4])
Output shape: torch.Size([2, 4, 4])
Test passed.
```

---

## 10. 常见错误

```text
1. 上三角和下三角搞反
2. diagonal 参数写错
3. mask dtype 不是 bool
4. mask shape 没有正确 broadcast 到 batch
5. 先 softmax 再 mask，导致概率不重新归一化
```

---

## 11. 扩展任务

```text
1. 支持 scores shape 为 [B,H,T,T]
2. 使用 float("-inf") 作为 mask_value
3. 把 causal mask 接到 Day 8 的 attention 中
4. 打印 attention matrix 观察上三角是否为 0
5. 支持 padding mask 和 causal mask 同时使用
```

---

## 12. 今日理解问题

```text
1. causal mask 为什么是下三角？
2. 为什么未来位置要在 softmax 之前填很小的数？
3. 第 0 个 token 能看见几个位置？
4. 第 t 个 token 能看见哪些位置？
5. causal mask 和 padding mask 有什么区别？
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
