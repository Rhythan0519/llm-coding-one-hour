# Day 26：Attention QKV LoRA

## 1. 今日目标

今天实现：

```python
class LoRACausalSelfAttention(nn.Module)
```

目标是把 Day23 的 `LoRALinear` 接到 attention 的 q/k/v projection 上。

---

## 2. 这个模块在大模型里有什么用

很多 LoRA 微调会 target attention 里的 projection：

```text
q_proj
k_proj
v_proj
out_proj
```

常见配置是只给 q/v 加 LoRA，或者给 q/k/v/o 都加。

---

## 3. 输入输出

输入：

```text
x: [B, T, C]
```

输出：

```text
out: [B, T, C]
```

中间 shape：

```text
q/k/v: [B, H, T, D]
scores: [B, H, T, T]
```

---

## 4. 核心逻辑

```text
1. q_proj/k_proj/v_proj 使用 LoRALinear
2. split heads
3. scaled dot-product attention
4. causal mask
5. merge heads
6. out_proj
```

---

## 5. 伪代码

```text
1. q = split_heads(q_proj(x))
2. k = split_heads(k_proj(x))
3. v = split_heads(v_proj(x))
4. scores = q @ k.T / sqrt(D)
5. apply causal mask
6. attn = softmax(scores)
7. context = attn @ v
8. return out_proj(merge_heads(context))
```

---

## 6. 代码骨架

```python
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    # TODO: reuse Day 23 implementation
    pass


class LoRACausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=2):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # TODO 1: define q_proj/k_proj/v_proj as LoRALinear
        # TODO 2: define out_proj

    def split_heads(self, x):
        # TODO 3: [B,T,C] -> [B,H,T,D]
        return None

    def merge_heads(self, x):
        # TODO 4: [B,H,T,D] -> [B,T,C]
        return None

    def apply_causal_mask(self, scores):
        # TODO 5: mask future positions
        return None

    def forward(self, x):
        # TODO 6: implement attention forward
        return None


def test():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    attn = LoRACausalSelfAttention(embed_dim=16, num_heads=4, rank=2)
    y = attn(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()

    loss = y.pow(2).mean()
    loss.backward()
    lora_params = [
        p for name, p in attn.named_parameters()
        if "lora_" in name
    ]
    assert len(lora_params) > 0
    assert all(p.grad is not None for p in lora_params if p.requires_grad)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 输出 shape 等于输入 shape
2. 输出没有 NaN
3. LoRA 参数存在
4. backward 后 LoRA 参数有梯度
```

---

## 8. 运行方式

```bash
python code/week4/day26/attention_qkv_lora.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([2, 5, 16])
Output shape: torch.Size([2, 5, 16])
Test passed.
```

---

## 10. 常见错误

```text
1. split_heads 维度顺序写错
2. mask 没放到 scores.device
3. LoRA 参数没有 requires_grad
4. merge_heads 忘记合并 H*D
```

---

## 11. 扩展任务

```text
1. 只给 q_proj/v_proj 加 LoRA
2. 统计 attention 中 LoRA 参数比例
3. 加 dropout
```

---

## 12. 今日理解问题

```text
1. q/k/v projection 的输入输出 shape 是什么？
2. 为什么 LoRA 可以接在线性层外面？
3. 哪些 projection 最常被 LoRA target？
4. attention 里 causal mask 的 shape 怎么广播？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出 shape：
- 最容易错的地方：
- 明天要复习的问题：
```

