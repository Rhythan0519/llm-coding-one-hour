# Day 15：Greedy Decoding

## 1. 今日目标

今天实现 greedy decoding：

```python
greedy_decode(logits)
```

目标是理解最简单的生成策略：每一步都选择概率最大的 token。

---

## 2. 这个模块在大模型里有什么用

语言模型 forward 后会输出 logits。生成文本时，需要把 logits 变成下一个 token id。

greedy decoding 是最简单的策略：

```text
每一步都选 logits 最大的 token。
```

它稳定、可复现，但容易生成重复或单调文本。

---

## 3. 输入输出

### 输入

```text
logits: [B, V] 或 [B, T, V]
```

其中：

```text
B = batch size
T = sequence length
V = vocab size
```

如果输入是 `[B,T,V]`，只取最后一个位置：

```text
last_logits = logits[:, -1, :]
```

### 输出

```text
next_ids: [B]
```

---

## 4. 核心逻辑

```text
1. 如果 logits 是 [B,T,V]，先取最后一个位置
2. 在 vocab 维度上取 argmax
3. 返回 token id
```

公式：

$$
token = \arg\max_i logits_i
$$

---

## 5. 伪代码

```text
1. 判断 logits 维度
2. 如果是 3D，取 logits[:, -1, :]
3. 对最后一维 argmax
4. 返回 next_ids
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def greedy_decode(logits):
    """
    Args:
        logits: [B, V] or [B, T, V]

    Returns:
        next_ids: [B]
    """
    # TODO 1: if logits has shape [B, T, V], keep only the last time step
    last_logits = None

    # TODO 2: take argmax over vocab dimension
    next_ids = None

    return next_ids


def test():
    logits_2d = torch.tensor([
        [0.1, 2.0, 0.3],
        [3.0, 1.0, 2.0],
    ])
    out_2d = greedy_decode(logits_2d)

    logits_3d = torch.tensor([
        [[9.0, 0.0, 0.0], [0.1, 0.2, 5.0]],
        [[0.0, 7.0, 0.0], [4.0, 3.0, 2.0]],
    ])
    out_3d = greedy_decode(logits_3d)

    assert out_2d.shape == (2,)
    assert out_3d.shape == (2,)
    assert torch.equal(out_2d, torch.tensor([1, 0]))
    assert torch.equal(out_3d, torch.tensor([2, 0]))
    assert out_2d.dtype == torch.long

    print("2D input shape:", logits_2d.shape)
    print("3D input shape:", logits_3d.shape)
    print("Output shape:", out_3d.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 支持 logits [B,V]
2. 支持 logits [B,T,V]
3. 输出 shape 是 [B]
4. 输出 dtype 是 torch.long
5. 输出等于 torch.argmax 的结果
```

---

## 8. 运行方式

保存为：

```text
day15_greedy_decoding.py
```

运行：

```bash
python day15_greedy_decoding.py
```

---

## 9. 预期输出

```text
2D input shape: torch.Size([2, 3])
3D input shape: torch.Size([2, 2, 3])
Output shape: torch.Size([2])
Test passed.
```

---

## 10. 常见错误

```text
1. 对 batch 维 argmax，而不是 vocab 维
2. 输入 [B,T,V] 时忘记只取最后一个位置
3. 返回 shape 是 [B,1] 而不是 [B]
4. 误把 softmax 后概率再 argmax，当成必要步骤
5. 没确认输出 dtype 是整数
```

---

## 11. 扩展任务

```text
1. 返回 next_ids 的同时返回最大概率
2. 支持 keepdim=True 输出 [B,1]
3. 把 greedy_decode 接进 Week2 tiny GPT generate
4. 比较 greedy 和 sampling 的输出差异
5. 加入 batch size 为 1 的测试
```

---

## 12. 今日理解问题

```text
1. logits 为什么可以直接 argmax？
2. greedy decoding 为什么可复现？
3. 输入 [B,T,V] 时为什么只取最后一个位置？
4. greedy decoding 有什么缺点？
5. next_ids 为什么必须是整数？
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

