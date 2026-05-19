# Week 2：Transformer 核心模块（2026-05-25 ~ 2026-05-31）

## 1. 本周目标

本周主要学习 Transformer 的核心组件：attention、mask、multi-head、FFN、residual、LayerNorm 和最小 GPT demo。

本周结束后，我应该能够：

- 手写 single-head attention；
- 理解 causal mask 如何阻止未来 token 泄漏；
- 写出 multi-head attention 的 reshape / transpose；
- 实现 Transformer FFN；
- 理解 residual connection 和 LayerNorm 的组合；
- 组合一个最小 Transformer block；
- 跑通一个 tiny character-level GPT demo。

---

## 2. 本周核心能力

```text
1. 理解 [B, T, C]、[B, H, T, D]、[B, T, T] 的含义
2. 会写 Q / K / V projection
3. 会计算 scaled dot-product attention
4. 会构造 causal mask
5. 会把多个 head 拆开再合并
6. 会组合 residual + norm + attention + FFN
```

---

## 3. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 8 | [day08_single_head_attention.md](day08_single_head_attention.md) | single-head attention | x `[B,T,C]` | y `[B,T,C]`, attn `[B,T,T]` | attention 每行和为 1 |
| Day 9 | [day09_causal_mask.md](day09_causal_mask.md) | causal mask | scores `[B,T,T]` | masked scores `[B,T,T]` | 未来 token 被 mask |
| Day 10 | [day10_multi_head_attention.md](day10_multi_head_attention.md) | multi-head attention | x `[B,T,C]` | y `[B,T,C]`, attn `[B,H,T,T]` | head 维度正确 |
| Day 11 | [day11_ffn.md](day11_ffn.md) | Transformer FFN | x `[B,T,C]` | y `[B,T,C]` | shape 不变，可 backward |
| Day 12 | [day12_residual_norm.md](day12_residual_norm.md) | residual + norm | x `[B,T,C]`, sublayer_out `[B,T,C]` | y `[B,T,C]` | residual 路径可 backward |
| Day 13 | [day13_transformer_block.md](day13_transformer_block.md) | Transformer block | x `[B,T,C]` | y `[B,T,C]` | forward/backward 跑通 |
| Day 14 | [day14_tiny_gpt.md](day14_tiny_gpt.md) | tiny GPT demo | token ids `[B,T]` | logits `[B,T,V]`, generated ids | loss 下降，可生成字符 |

---

## 4. 本周 mini demo

实现一个 tiny character-level GPT：

输入：

```text
一小段文本，例如 "hello world hello world"
```

输出：

```text
训练 loss
logits: [B, T, vocab_size]
生成出来的字符
```

验收：

```text
1. loss 能下降
2. logits shape 正确
3. 能从 prefix 生成若干字符
4. 脚本可以直接运行
```

---

## 5. 本周验收标准

```text
1. 所有每日脚本都能单独运行。
2. 每个脚本都有至少 2 个 assert。
3. 每个脚本都打印输入输出 shape。
4. 我能解释 Q / K / V 的 shape。
5. 我能解释 causal mask 为什么是下三角。
6. 我能不看答案重写 single-head attention 或 causal mask。
7. tiny GPT demo 能跑通 forward、backward 和 generate。
```

---

## 6. 本周复盘问题

```text
1. Q、K、V 分别是什么 shape？
2. attention score 为什么要除以 sqrt(head_dim)？
3. causal mask 为什么是下三角？
4. multi-head attention 为什么要 reshape 成 [B,H,T,D]？
5. residual connection 的作用是什么？
6. pre-norm 和 post-norm 有什么区别？
7. tiny GPT 的 logits 为什么是 [B,T,V]？
```

