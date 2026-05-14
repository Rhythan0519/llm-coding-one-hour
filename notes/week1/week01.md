# Week 1：PyTorch Tensor 与基础函数（2026-05-18 ~ 2026-05-24）

## 1. 本周目标

本周主要学习 PyTorch tensor、shape、基础函数和 loss。

本周结束后，我应该能够：

- 手写 softmax；
- 手写 cross entropy；
- 手写 layer norm；
- 理解 embedding lookup；
- 写一个简单 MLP classifier；
- 用 assert 检查输出；
- 和 PyTorch 官方实现对齐。

---

## 2. 本周核心能力

```text
1. 理解 [B, C] 和 [B, T, C]
2. 熟悉 torch.max / torch.sum / torch.exp / torch.log
3. 熟悉 tensor indexing
4. 熟悉 nn.Module 的基本写法
5. 熟悉 loss.backward()
```

---

## 3. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 1 | [day01_softmax_ce.md](day01_softmax_ce.md) | softmax + cross entropy | logits `[B,C]`, labels `[B]` | probs `[B,C]`, loss scalar | 对齐 `F.cross_entropy` |
| Day 2 | [day02_layernorm.md](day02_layernorm.md) | layer norm | x `[B,T,C]` | y `[B,T,C]` | mean≈0, std≈1 |
| Day 3 | [day03_embedding.md](day03_embedding.md) | embedding lookup | token ids `[B,T]` | embeddings `[B,T,D]` | 对齐 `nn.Embedding` |
| Day 4 | [day04_positional_embedding.md](day04_positional_embedding.md) | positional embedding | positions `[T]` | pos emb `[T,D]` | 不同位置不同 |
| Day 5 | [day05_linear_classifier.md](day05_linear_classifier.md) | linear classifier | toy features `[B,D]` | logits `[B,num_classes]` | loss 能下降 |
| Day 6 | [day06_mlp_classifier.md](day06_mlp_classifier.md) | MLP classifier | toy features `[B,D]` | logits `[B,num_classes]` | 可 forward/backward |
| Day 7 | [day07_week01_demo.md](day07_week01_demo.md) | tiny classifier demo | toy dataset | prediction + loss curve | 训练闭环跑通 |

---

## 4. 本周 mini demo

实现一个 tiny MLP classifier：

输入：

```text
二维 toy points: [B, 2]
```

输出：

```text
类别预测: [B]
loss 变化
accuracy
```

验收：

```text
1. loss 下降
2. accuracy 高于随机猜测
3. 脚本可以直接运行
```

---

## 5. 本周验收标准

```text
1. 所有每日脚本都能单独运行。
2. 每个脚本都有 assert。
3. 每个脚本都打印输入输出 shape。
4. 我能解释本周每个模块的作用。
5. 我能不看答案重新写出至少 2 个核心模块。
6. 周末 mini demo 能跑通。
```

---

## 6. 本周复盘问题

```text
1. softmax 为什么要减最大值？
2. cross entropy 为什么是 -log 正确类别概率？
3. layer norm 是在哪个维度归一化？
4. embedding lookup 本质上是什么？
5. loss.backward() 后哪些参数有梯度？
```

