# Day 21：Full Attention vs Cache Attention Demo

## 1. 今日目标

今天实现一个 full attention vs cache attention 对比 demo：

```python
compare_full_attention_and_cache()
```

目标是验证：自回归生成时，使用 KV cache 的最后一步输出应该和 full attention 的最后一步输出近似一致。

---

## 2. 这个模块在大模型里有什么用

KV cache 的核心价值是减少重复计算。

不使用 cache：

```text
每一步都重新计算整段序列的 Q/K/V 和 attention。
```

使用 cache：

```text
每一步只计算新 token 的 Q/K/V。
历史 K/V 直接复用。
```

理论上，如果权重一样、mask 正确，最后一个 token 的输出应当和 full attention 对齐。

---

## 3. 输入输出

### 输入

```text
x: [B, T, C]
```

### 输出

```text
full_out: [B, T, C]
cache_last_outputs: [B, T, C]
```

其中 `cache_last_outputs[:, t, :]` 表示 cache 方式在第 `t` 步得到的新 token 输出。

---

## 4. 核心逻辑

full attention：

```text
一次性输入 [B,T,C]，得到 [B,T,C]
```

cache attention：

```text
for t in range(T):
    当前只输入 x[:, t:t+1, :]
    计算 q/k/v
    把新 k/v 拼进 cache
    q 和 cache_k/cache_v 做 attention
    得到当前 token 输出
```

对比：

```text
full_out 和 cache_out 在每个位置都应近似一致
```

---

## 5. 伪代码

```text
1. 实现一个共享权重的 CausalSelfAttention
2. forward_full(x): 全量 causal attention
3. forward_step(x_t, cache): 单步 cache attention
4. full_out = forward_full(x)
5. 循环 T 次调用 forward_step
6. 拼出 cache_out
7. assert full_out 和 cache_out 近似一致
```

---

## 6. 代码骨架

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleKVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, new_k, new_v):
        # TODO 1: reuse Day 20 cache update logic
        return self.k, self.v


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            x: [B, H, T, D]
        """
        B, T, C = x.shape

        # TODO 2: reshape and transpose
        x = None

        return x

    def merge_heads(self, x):
        """
        Args:
            x: [B, H, T, D]
        Returns:
            x: [B, T, C]
        """
        B, H, T, D = x.shape

        # TODO 3: transpose and reshape
        x = None

        return x

    def apply_causal_mask(self, scores):
        """
        Args:
            scores: [B, H, T, T]
        """
        B, H, T, _ = scores.shape

        # TODO 4: mask future positions
        scores = None

        return scores

    def forward_full(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            out: [B, T, C]
        """
        # TODO 5: compute full causal self-attention
        out = None
        return out

    def forward_step(self, x_t, cache):
        """
        Args:
            x_t: [B, 1, C]
            cache: SimpleKVCache

        Returns:
            out_t: [B, 1, C]
        """
        # TODO 6: compute q/k/v for current token
        q = None
        new_k = None
        new_v = None

        # TODO 7: update cache and attend to all cached keys/values
        cache_k, cache_v = None

        # TODO 8: compute attention without causal mask
        # Reason: cache only contains past + current tokens.
        out_t = None

        return out_t


def test():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    attn = CausalSelfAttention(embed_dim=16, num_heads=4)

    full_out = attn.forward_full(x)

    cache = SimpleKVCache()
    step_outs = []
    for t in range(x.size(1)):
        out_t = attn.forward_step(x[:, t:t + 1, :], cache)
        step_outs.append(out_t)
    cache_out = torch.cat(step_outs, dim=1)

    assert full_out.shape == x.shape
    assert cache_out.shape == x.shape
    assert torch.allclose(full_out, cache_out, atol=1e-5)
    assert not torch.isnan(cache_out).any()

    loss = cache_out.pow(2).mean()
    loss.backward()
    assert attn.q_proj.weight.grad is not None

    print("Input shape:", x.shape)
    print("Full output shape:", full_out.shape)
    print("Cache output shape:", cache_out.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. full_out.shape == x.shape
2. cache_out.shape == x.shape
3. full_out 和 cache_out 数值近似一致
4. cache_out 没有 NaN
5. backward 后 attention 参数有梯度
```

---

## 8. 运行方式

保存为：

```text
day21_cache_vs_full_attention.py
```

运行：

```bash
python day21_cache_vs_full_attention.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 5, 16])
Full output shape: torch.Size([2, 5, 16])
Cache output shape: torch.Size([2, 5, 16])
Test passed.
```

---

## 10. 常见错误

```text
1. forward_step 里错误地再次使用 causal mask
2. cache update 的 concat 维度写错
3. full attention 和 step attention 使用了不同权重
4. split_heads / merge_heads 维度写反
5. 对比时只看 shape，没有检查数值一致
```

---

## 11. 扩展任务

```text
1. 打印每一步 cache_k shape
2. 支持一次 prefill 多个 token，再 step 一个 token
3. 统计 full attention 和 cache attention 的计算量差异
4. 接到 tiny GPT 的 generate
5. 只比较最后一个 token 输出，模拟真实推理
```

---

## 12. 今日理解问题

```text
1. 为什么 forward_step 不需要 causal mask？
2. full attention 和 cache attention 为什么应该输出一致？
3. cache attention 节省了哪些重复计算？
4. 为什么 Q 不需要缓存？
5. 真实 LLM 推理里 prefill 和 decode 分别是什么意思？
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
- 下周要复习：
```

