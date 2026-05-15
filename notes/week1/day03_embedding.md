# Day 3：Embedding Lookup

## 1. 今日目标

今天实现一个 `MyEmbedding`：

```python
MyEmbedding(vocab_size, embedding_dim)
```

目标是理解 token id 是怎么被映射成向量的。

---

## 2. 这个模块在大模型里有什么用

Embedding 是 token 进入模型的第一步。输入是离散 id，输出是连续向量。

---

## 3. 输入输出

### 输入

```text
token_ids: [B, T]
```

### 输出

```text
embeddings: [B, T, D]
```

其中 `D` 是 embedding 维度。

---

## 4. 核心逻辑

本质上就是按 token id 去参数表里查对应行：

$$
E[token\_id]
$$

---

## 5. 伪代码

```text
1. 初始化 embedding table: [vocab_size, D]
2. 用 token ids 做索引
3. 返回查到的向量
```

---

## 6. 代码骨架

```python
import torch
import torch.nn as nn


class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.02)

    def forward(self, token_ids):
        # TODO: gather rows by token ids
        embeddings = None
        return embeddings


def test():
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 4, 3], [0, 2, 2]], dtype=torch.long)
    mine = MyEmbedding(10, 8)
    ref = nn.Embedding(10, 8)
    ref.weight.data.copy_(mine.weight.data)

    my_out = mine(token_ids)
    ref_out = ref(token_ids)

    assert my_out.shape == (2, 3, 8)
    assert torch.allclose(my_out, ref_out, atol=1e-6)
    assert token_ids.dtype == torch.long

    print("Input shape:", token_ids.shape)
    print("Output shape:", my_out.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. shape 正确
2. 对齐 nn.Embedding
3. token ids 必须是 long
4. 没有 NaN
```

---

## 8. 运行方式

```text
day03_embedding.py
python day03_embedding.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 3])
Output shape: torch.Size([2, 3, 8])
Test passed.
```

---

## 10. 常见错误

```text
1. token ids dtype 不对
2. 索引维度写反
3. 忘记检查重复 token 应该得到相同向量
4. 初始化尺度太大
5. 输出 shape 少了一维
```

---

## 11. 扩展任务

```text
1. 支持 padding_idx
2. 加入 embedding dropout
3. 和 positional embedding 相加
4. 观察相同 id 的向量是否一致
5. 支持 batchless 输入
```

---

## 12. 今日理解问题

```text
1. embedding lookup 本质上是什么？
2. 为什么 token ids 必须是 long？
3. 为什么输出会多出一个 embedding 维？
4. 同一个 id 为什么应该返回同一个向量？
5. 这个模块在 Transformer 里放在哪一步？
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
