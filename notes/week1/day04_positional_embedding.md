# Day 4：Positional Embedding

## 1. 今日目标

今天实现一个位置编码函数：

```python
sinusoidal_position_embedding(positions, dim)
```

目标是让模型知道 token 在序列中的位置。

---

## 2. 这个模块在大模型里有什么用

Transformer 自己不关心顺序。位置编码把顺序信息注入到 token 表示里。

---

## 3. 输入输出

### 输入

```text
positions: [T]
dim: D
```

### 输出

```text
pos_emb: [T, D]
```

---

## 4. 核心公式

经典 sinusoidal 位置编码：

$$
PE(pos, 2i) = \sin\left(pos / 10000^{2i/D}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(pos / 10000^{2i/D}\right)
$$

---

## 5. 伪代码

```text
1. 计算每个维度对应的频率
2. 把 positions broadcast 到 [T, D/2]
3. 偶数维填 sin
4. 奇数维填 cos
5. 拼起来返回
```

---

## 6. 代码骨架

```python
import torch


def sinusoidal_position_embedding(positions, dim):
    """
    Args:
        positions: [T]
        dim: int
    Returns:
        pos_emb: [T, D]
    """
    # TODO 1: create frequency terms
    freqs = None

    # TODO 2: broadcast positions
    angles = None

    # TODO 3: fill sin/cos pairs
    pos_emb = None
    return pos_emb


def test():
    positions = torch.arange(0, 5)
    pos_emb = sinusoidal_position_embedding(positions, 8)

    assert pos_emb.shape == (5, 8)
    assert not torch.allclose(pos_emb[0], pos_emb[1])
    assert not torch.isnan(pos_emb).any()

    print("Input shape:", positions.shape)
    print("Output shape:", pos_emb.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. shape 正确
2. 不同位置输出不同
3. 不能有 NaN
4. 高低频维度都能工作
```

---

## 8. 运行方式

```text
day04_positional_embedding.py
python day04_positional_embedding.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([5])
Output shape: torch.Size([5, 8])
Test passed.
```

---

## 10. 常见错误

```text
1. 偶数维和奇数维搞反
2. dim 不是偶数时没处理
3. positions 形状广播错
4. 公式里的指数写错
5. 输出 shape 少了一维
```

---

## 11. 扩展任务

```text
1. 做 learnable positional embedding
2. 把位置编码加到 token embedding 上
3. 对比绝对位置和相对位置
4. 画出不同维度的波形
5. 支持任意 max_len
```

---

## 12. 今日理解问题

```text
1. 为什么 Transformer 需要位置编码？
2. sin/cos 两组维度分别表达什么？
3. 为什么不同位置会对应不同向量？
4. 为什么这种编码可以外推到更长序列？
5. 这个模块一般和 embedding 在哪里相加？
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

