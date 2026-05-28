# Week 4：LoRA / PEFT / 微调基础（2026-06-08 ~ 2026-06-14）

## 1. 本周目标

本周主要学习参数高效微调的基础：manual linear、LoRA Linear、冻结 base 参数、统计可训练参数、给 attention 的 q/k/v 挂 LoRA、保存加载 LoRA 权重，以及一个最小 PEFT demo。

本周结束后，我应该能够：

- 手写一个和 `nn.Linear` 对齐的 Linear；
- 理解 LoRA 的低秩更新公式；
- 写出 `LoRALinear`；
- 冻结 base weight，只训练 LoRA 参数；
- 统计总参数量和可训练参数量；
- 把 LoRA 接到 attention 的 q/k/v projection 上；
- 只保存和加载 LoRA 参数；
- 跑通一个 toy fine-tuning demo。

---

## 2. 本周核心能力

```text
1. 理解 y = xW^T + b 的 shape
2. 理解 LoRA: W' = W + scale * B @ A
3. 熟悉 requires_grad 和参数冻结
4. 熟悉 named_parameters / state_dict
5. 能判断哪些参数应该参与训练
6. 能验证 save/load 前后输出一致
```

---

## 3. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 22 | [day22_manual_linear.md](day22_manual_linear.md) | manual linear | x `[B,in]`, weight `[out,in]`, bias `[out]` | y `[B,out]` | 对齐 `F.linear` |
| Day 23 | [day23_lora_linear.md](day23_lora_linear.md) | LoRA Linear | x `[B,in]` | y `[B,out]` | shape 不变，r=0 时等于 base |
| Day 24 | [day24_freeze_base_train_lora.md](day24_freeze_base_train_lora.md) | freeze base train LoRA | toy data | loss | 只有 LoRA 参数有梯度 |
| Day 25 | [day25_count_trainable_params.md](day25_count_trainable_params.md) | count params | model | total/trainable numbers | trainable ratio 正确 |
| Day 26 | [day26_attention_qkv_lora.md](day26_attention_qkv_lora.md) | attention qkv LoRA | x `[B,T,C]` | y `[B,T,C]` | q/k/v 可挂 LoRA |
| Day 27 | [day27_save_load_lora.md](day27_save_load_lora.md) | save/load LoRA | checkpoint | same output | reload 后输出一致 |
| Day 28 | [day28_mini_peft_demo.md](day28_mini_peft_demo.md) | mini PEFT demo | toy task | before/after loss | 微调后 loss 下降 |

---

## 4. 本周 mini demo

实现一个 tiny PEFT fine-tuning demo：

输入：

```text
toy features: [B, D]
toy labels: [B]
一个冻结 base linear / tiny classifier
```

输出：

```text
before loss
after loss
trainable parameter ratio
save/load 后输出一致性检查
```

验收：

```text
1. base 参数被冻结
2. LoRA 参数有梯度
3. 训练后 loss 下降
4. trainable 参数比例明显小于 100%
5. LoRA 权重保存加载后输出一致
```

---

## 5. 本周验收标准

```text
1. 所有每日脚本都能单独运行。
2. 每个脚本都有至少 2 个 assert。
3. 每个脚本都打印输入输出 shape。
4. 我能解释 LoRA 的 A/B 矩阵 shape。
5. 我能解释为什么 base weight 要冻结。
6. 我能说清楚 state_dict 里哪些是 LoRA 参数。
7. mini PEFT demo 能跑通 forward、backward、save/load。
```

---

## 6. 本周复盘问题

```text
1. Linear 的 weight 为什么通常是 [out_features, in_features]？
2. LoRA 的 A 和 B 分别是什么 shape？
3. 为什么 LoRA B 经常初始化为 0？
4. requires_grad=False 和 torch.no_grad() 有什么区别？
5. 怎么判断一个参数有没有梯度？
6. 为什么只保存 LoRA 权重就可以迁移 adapter？
7. attention 里 q_proj、k_proj、v_proj 哪些地方适合挂 LoRA？
```

