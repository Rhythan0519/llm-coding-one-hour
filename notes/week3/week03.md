# Week 3：LLM 生成与推理（2026-06-01 ~ 2026-06-07）

## 1. 本周目标

本周主要学习 LLM 推理阶段的 decoding、sampling 和 KV cache。

本周结束后，我应该能够：

- 理解 logits 如何变成下一个 token；
- 手写 greedy decoding；
- 理解 temperature 对概率分布的影响；
- 手写 top-k sampling；
- 手写 top-p sampling；
- 实现 repetition penalty；
- 理解 KV cache 的 shape 和更新方式；
- 对比 full attention 和 cache attention 的计算差异。

---

## 2. 本周核心能力

```text
1. 理解 logits -> probs -> token id 的过程
2. 熟悉 torch.argmax / torch.softmax / torch.multinomial
3. 熟悉 topk、sort、scatter、masked_fill
4. 理解生成时为什么只取最后一个位置的 logits
5. 理解 KV cache 为什么能减少重复计算
6. 能把 Week2 的 tiny GPT generate 拆成多个可测试函数
```

---

## 3. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 15 | [day15_greedy_decoding.md](day15_greedy_decoding.md) | greedy decoding | logits `[B,V]` 或 `[B,T,V]` | token ids `[B]` | 等于 argmax |
| Day 16 | [day16_temperature_sampling.md](day16_temperature_sampling.md) | temperature sampling | logits `[B,V]`, temperature | sampled ids `[B]`, probs `[B,V]` | temperature 改变分布尖锐程度 |
| Day 17 | [day17_top_k_sampling.md](day17_top_k_sampling.md) | top-k sampling | logits `[B,V]`, k | sampled ids `[B]` | 只从 top-k token 采样 |
| Day 18 | [day18_top_p_sampling.md](day18_top_p_sampling.md) | top-p sampling | logits `[B,V]`, p | sampled ids `[B]` | 只从累计概率候选集采样 |
| Day 19 | [day19_repetition_penalty.md](day19_repetition_penalty.md) | repetition penalty | logits `[B,V]`, history `[B,T]` | adjusted logits `[B,V]` | 历史 token 被降权 |
| Day 20 | [day20_kv_cache.md](day20_kv_cache.md) | simple KV cache | new k/v `[B,H,1,D]` | cache k/v `[B,H,T,D]` | cache 长度正确增长 |
| Day 21 | [day21_cache_vs_full_attention.md](day21_cache_vs_full_attention.md) | full attention vs cache demo | token sequence | full/cache outputs | shape 一致，最后一步近似一致 |

---

## 4. 本周 mini demo

实现一个 sampling playground：

输入：

```text
一组 logits: [B, V]
可选历史 token: [B, T]
```

输出：

```text
greedy token
temperature sampled token
top-k sampled token
top-p sampled token
repetition penalty 后的 sampled token
```

验收：

```text
1. 每种 decoding 函数都能单独测试
2. top-k 不会采到 top-k 之外的 token
3. top-p 不会采到 nucleus set 之外的 token
4. repetition penalty 会降低历史 token 的 logits
5. mini demo 能直接运行
```

---

## 5. 本周验收标准

```text
1. 所有每日脚本都能单独运行。
2. 每个脚本都有至少 2 个 assert。
3. 每个脚本都打印输入输出 shape。
4. 我能解释 greedy、temperature、top-k、top-p 的区别。
5. 我能解释 generate 时为什么只用最后一个位置的 logits。
6. 我能画出 KV cache 的 shape 变化。
7. 我能把 Day15-19 的函数接进 Week2 tiny GPT 的 generate。
```

---

## 6. 本周复盘问题

```text
1. logits 和 probs 的区别是什么？
2. temperature 越大，分布为什么越平？
3. top-k 和 top-p 的候选集有什么区别？
4. repetition penalty 为什么要作用在 softmax 之前？
5. KV cache 缓存的是 K/V，为什么不是 Q？
6. full attention 和 cache attention 哪些计算是重复的？
7. 生成时为什么只需要最后一个 token 的输出？
```

