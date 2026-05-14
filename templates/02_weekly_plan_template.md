# Weekly Plan Template：每周计划怎么写

## 0. 使用方式

每周开始前，先让 Codex 根据本模板生成本周计划。

本模板的目标是：

> 把一周拆成 5 到 7 个小任务。  
> 每天一个 `.py` 文件。  
> 每天一个可验证模块。  
> 周末做一个 mini demo 或复盘。

不要让 Codex 直接写完整代码。  
只让它写任务说明、输入输出、TODO 骨架和测试标准。

---

# 1. 给 Codex 的固定提示词

每周开始时，把下面这段喂给 Codex：

```text
你是我的 LLM coding 教练，不是代写者。

请根据下面的 stage，为我生成一周 coding 计划。

要求：
1. 每天只安排一个小模块。
2. 每天的任务必须能在 1 小时内完成。
3. 每天必须有输入、输出、shape、测试标准。
4. 不要直接给完整实现代码。
5. 只给任务目标、函数接口、伪代码、TODO 骨架、测试要求。
6. 每天都要能生成一个可直接运行的 Python 文件。
7. 优先使用 PyTorch。
8. 如果能和 PyTorch 官方实现对齐，必须设置对齐测试。
9. 周末安排一个 mini demo，把本周模块串起来。
10. 最后给出本周完成标准和复盘问题。

我本周要学习的 stage 是：
[在这里填写 stage 名称]

我的当前水平：
- 能看懂基本 Python；
- PyTorch 不够熟；
- 想练大模型相关基础模块；
- 每天只有 1 小时；
- 不希望 Codex 直接代写完整答案。
```

---

# 2. 每周计划基本结构

每周计划应该包含以下部分：

```markdown
# Week X：主题名称

## 本周目标

## 本周核心能力

## 每日任务安排

## 每天文件命名

## 本周 mini demo

## 本周验收标准

## 本周复盘问题

## 给 Codex 的每日使用规则
```

---

# 3. 每周计划模板

下面是标准模板。

```markdown
# Week X：本周主题

## 1. 本周目标

本周主要学习：

- 模块 A
- 模块 B
- 模块 C

本周结束后，我应该能够：

- 独立写出 xxx；
- 看懂 xxx 代码；
- 用 assert 检查 xxx；
- 解释输入输出 shape。

---

## 2. 本周核心能力

本周重点训练：

```text
1. shape 推理能力
2. PyTorch tensor 操作
3. 模块化实现能力
4. 测试和 debug 能力
5. 用自己的话解释代码的能力
```

---

## 3. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 1 | dayXX_xxx.py | xxx | xxx | xxx | xxx |
| Day 2 | dayXX_xxx.py | xxx | xxx | xxx | xxx |
| Day 3 | dayXX_xxx.py | xxx | xxx | xxx | xxx |
| Day 4 | dayXX_xxx.py | xxx | xxx | xxx | xxx |
| Day 5 | dayXX_xxx.py | xxx | xxx | xxx | xxx |
| Day 6 | dayXX_xxx.py | xxx | xxx | xxx | xxx |
| Day 7 | weekXX_demo.py | mini demo | xxx | xxx | xxx |

---

## 4. 每天的固定流程

每天按照这个节奏：

```text
10 分钟：读任务卡，确认输入输出
35 分钟：自己填 TODO
10 分钟：运行测试，修 bug
5 分钟：写总结
```

---

## 5. 每天必须输出什么

每天结束时，必须有：

```text
1. 一个可运行的 .py 文件
2. 至少两个 assert
3. 输入 shape 打印
4. 输出 shape 打印
5. 测试通过信息
6. 今日总结
```

---

## 6. 本周 mini demo

本周最后一天，将前几天模块串成一个小 demo。

mini demo 要求：

```text
1. 有明确输入
2. 有明确输出
3. 能直接运行
4. 打印关键中间 shape
5. 至少一个可解释结果
```

---

## 7. 本周验收标准

本周完成的标准：

```text
1. 所有每日脚本都能单独运行。
2. 每个脚本都有 assert。
3. 每个脚本都打印输入输出 shape。
4. 我能解释本周每个模块的作用。
5. 我能不看答案重新写出至少 2 个核心模块。
6. 周末 mini demo 能跑通。
```

---

## 8. 本周复盘问题

周末问自己：

```text
1. 本周哪个 shape 最容易错？
2. 哪个 PyTorch API 我还不熟？
3. 哪个模块我能不看代码重写？
4. 哪个模块只是跑通但还没理解？
5. 下周需要复习哪个点？
```

---

## 9. 给 Codex 的规则

本周使用 Codex 时遵守：

```text
1. 不让 Codex 直接生成完整实现。
2. 先让 Codex 给 TODO 骨架。
3. 我自己写完后，再让 Codex 检查。
4. Codex 只能给最小修改建议。
5. Codex 要问我 3 到 5 个理解问题。
```
```

---

# 4. 示例：Week 1 计划

下面是第一周可以直接使用的版本。

```markdown
# Week 1：PyTorch Tensor 与基础大模型函数

## 1. 本周目标

本周目标是熟悉 PyTorch tensor、shape、基础函数和 loss。

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
| Day 1 | day01_softmax_ce.py | softmax + cross entropy | logits `[B,C]`, labels `[B]` | probs `[B,C]`, loss scalar | 对齐 `F.cross_entropy` |
| Day 2 | day02_layernorm.py | layer norm | x `[B,T,C]` | y `[B,T,C]` | mean≈0, std≈1 |
| Day 3 | day03_embedding.py | embedding lookup | token ids `[B,T]` | embeddings `[B,T,D]` | 对齐 `nn.Embedding` |
| Day 4 | day04_positional_embedding.py | positional embedding | positions `[T]` | pos emb `[T,D]` | 不同位置不同 |
| Day 5 | day05_linear_classifier.py | linear classifier | toy features `[B,D]` | logits `[B,num_classes]` | loss 能下降 |
| Day 6 | day06_mlp_classifier.py | MLP classifier | toy features `[B,D]` | logits `[B,num_classes]` | 可 forward/backward |
| Day 7 | week01_demo.py | tiny classifier demo | toy dataset | prediction + loss curve | 训练闭环跑通 |

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

## 5. 本周复盘问题

```text
1. softmax 为什么要减最大值？
2. cross entropy 为什么是 -log 正确类别概率？
3. layer norm 是在哪个维度归一化？
4. embedding lookup 本质上是什么？
5. loss.backward() 后哪些参数有梯度？
```
```

---

# 5. 示例：Week 2 计划

```markdown
# Week 2：Transformer 核心模块

## 1. 本周目标

本周目标是从零实现最小 Transformer 相关模块。

本周结束后，我应该能够：

- 写 single-head attention；
- 写 causal mask；
- 写 multi-head attention；
- 写 FFN；
- 组合 transformer block；
- 理解 `[B, H, T, D]` 的含义。

---

## 2. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 8 | day08_single_head_attention.py | 单头 attention | x `[B,T,C]` | y `[B,T,C]`, attn `[B,T,T]` | attention 每行和为 1 |
| Day 9 | day09_causal_mask.py | causal mask | scores `[B,T,T]` | masked scores | 未来 token 被 mask |
| Day 10 | day10_multi_head_attention.py | 多头 attention | x `[B,T,C]` | y `[B,T,C]`, attn `[B,H,T,T]` | shape 对齐 |
| Day 11 | day11_ffn.py | feed-forward network | x `[B,T,C]` | y `[B,T,C]` | shape 不变 |
| Day 12 | day12_residual_norm.py | residual + layer norm | x `[B,T,C]` | y `[B,T,C]` | 可 backward |
| Day 13 | day13_transformer_block.py | transformer block | x `[B,T,C]` | y `[B,T,C]` | forward/backward |
| Day 14 | week02_tiny_gpt.py | tiny GPT demo | token ids | logits / generated text | loss 下降 |

---

## 3. 本周 mini demo

实现一个 tiny character-level GPT。

输入：

```text
一小段文本，例如 "hello world hello world"
```

输出：

```text
训练 loss
生成字符
```

验收：

```text
1. loss 能下降
2. 能生成字符
3. forward/backward 不报错
```

---

## 4. 本周复盘问题

```text
1. Q/K/V 分别是什么 shape？
2. attention score 为什么要除以 sqrt(d_k)？
3. causal mask 为什么是下三角？
4. multi-head attention 为什么要 reshape 成 [B,H,T,D]？
5. residual connection 有什么作用？
```
```

---

# 6. 每周结束后让 Codex 做什么

周末把本周代码给 Codex，然后使用：

```text
请你作为 coding 教练 review 我本周的代码。

要求：
1. 不要重写完整代码。
2. 只指出最重要的 5 个问题。
3. 每个问题说明原因。
4. 每个问题给最小修改建议。
5. 最后问我 5 个理解问题，检查我是否真正理解。
6. 请根据我的表现安排下周计划。
```

---

# 7. 每周计划不要太满

每周宁愿少一点，也不要把任务排爆。

推荐难度：

```text
5 天基础任务
1 天综合 demo
1 天复盘 / 重写 / 修 bug
```

如果某天没完成，不要补两倍任务。  
第二天继续修同一个模块，直到真的理解。
