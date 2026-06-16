# Day 32：DiT Block

## 1. 今日目标

今天实现一个最小 DiT block：

```python
DiTBlock(tokens, cond)
```

它和普通 Transformer block 的区别是：LayerNorm 不只归一化 token，还会被 timestep / label condition 调制。

---

## 2. Block 结构

```text
x
-> norm1
-> adaLN shift/scale modulation
-> self-attention
-> gate + residual
-> norm2
-> adaLN shift/scale modulation
-> MLP
-> gate + residual
```

condition vector 会生成 6 组参数：

```text
shift_msa, scale_msa, gate_msa
shift_mlp, scale_mlp, gate_mlp
```

---

## 3. 为什么初始化 gate 为 0

代码里把 adaLN 的最后一层初始化为 0，所以 block 初始时接近 identity：

```text
out ~= x
```

这样深层 DiT 一开始更稳定。今天的测试会检查这一点。

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day32_dit_block.py
```

---

## 5. 今日理解问题

```text
1. DiT 为什么不需要 causal mask？
2. gate_msa 和 gate_mlp 分别控制什么？
3. adaLN 和普通 LayerNorm 的区别是什么？
```
