# Day 20：Simple KV Cache

## 1. 今日目标

今天实现一个最小 KV cache：

```python
KVCache
```

目标是理解自回归生成时，为什么可以缓存历史 token 的 K / V。

---

## 2. 这个模块在大模型里有什么用

生成时每次只新增一个 token。

如果没有 KV cache，每一步都要重新计算整个序列的 K / V：

```text
step 1: token 1
step 2: token 1,2
step 3: token 1,2,3
```

有了 KV cache，历史 K / V 可以复用，每步只计算新 token 的 K / V，然后拼到 cache 后面。

---

## 3. 输入输出

### 输入

```text
new_k: [B, H, 1, D]
new_v: [B, H, 1, D]
```

### 输出

```text
cache_k: [B, H, T, D]
cache_v: [B, H, T, D]
```

每次 update 后，`T` 增加 1。

---

## 4. 核心逻辑

第一次 update：

```text
cache_k = new_k
cache_v = new_v
```

后续 update：

```text
cache_k = cat([cache_k, new_k], dim=2)
cache_v = cat([cache_v, new_v], dim=2)
```

这里 `dim=2` 是 sequence length 维。

---

## 5. 伪代码

```text
1. 初始化 cache_k/cache_v 为 None
2. update(new_k, new_v)
3. 如果 cache 为空，直接保存
4. 如果 cache 非空，沿 T 维 concat
5. 返回当前 cache
6. reset() 清空 cache
```

---

## 6. 代码骨架

```python
import torch


class KVCache:
    def __init__(self):
        # TODO 1: initialize empty cache
        self.k = None
        self.v = None

    def update(self, new_k, new_v):
        """
        Args:
            new_k: [B, H, 1, D]
            new_v: [B, H, 1, D]

        Returns:
            k: [B, H, T, D]
            v: [B, H, T, D]
        """
        assert new_k.shape == new_v.shape
        assert new_k.dim() == 4
        assert new_k.size(2) == 1

        # TODO 2: if cache is empty, set k/v to new_k/new_v
        # TODO 3: otherwise concatenate on sequence dimension

        return self.k, self.v

    def reset(self):
        # TODO 4: clear cache
        pass

    @property
    def length(self):
        # TODO 5: return current sequence length
        length = None
        return length


def test():
    torch.manual_seed(0)
    cache = KVCache()

    assert cache.length == 0

    for step in range(3):
        new_k = torch.randn(2, 4, 1, 8)
        new_v = torch.randn(2, 4, 1, 8)
        k, v = cache.update(new_k, new_v)

        assert k.shape == (2, 4, step + 1, 8)
        assert v.shape == (2, 4, step + 1, 8)
        assert cache.length == step + 1
        assert torch.allclose(k[:, :, -1:, :], new_k)
        assert torch.allclose(v[:, :, -1:, :], new_v)

    cache.reset()
    assert cache.k is None
    assert cache.v is None
    assert cache.length == 0

    print("Last K shape:", k.shape)
    print("Last V shape:", v.shape)
    print("Cache length after reset:", cache.length)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 初始 cache length 为 0
2. 每次 update 后 T 增加 1
3. 新 token 的 K/V 出现在最后一个位置
4. reset 后 cache 为空
5. K 和 V shape 一致
```

---

## 8. 运行方式

保存为：

```text
day20_kv_cache.py
```

运行：

```bash
python day20_kv_cache.py
```

---

## 9. 预期输出

```text
Last K shape: torch.Size([2, 4, 3, 8])
Last V shape: torch.Size([2, 4, 3, 8])
Cache length after reset: 0
Test passed.
```

---

## 10. 常见错误

```text
1. concat 的 dim 写错，拼到了 head 或 hidden 维
2. 第一次 update 时 None 不能直接 cat
3. reset 只清了 k，忘了清 v
4. length 属性在空 cache 时报错
5. new_k/new_v 的 T 维不是 1
```

---

## 11. 扩展任务

```text
1. 支持一次 update 多个 token: [B,H,T_new,D]
2. 支持 max_length，超过后裁掉最早 token
3. 支持 batch size 改变时报错
4. 把 KVCache 接进单层 attention
5. 打印每一步 cache shape
```

---

## 12. 今日理解问题

```text
1. KV cache 为什么缓存 K 和 V，而不是 Q？
2. cache 的 sequence 维是哪一维？
3. 每次生成新增几个 token？
4. 没有 cache 时哪些计算被重复了？
5. cache 会不会改变模型输出？
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

