# LLM Coding One-Hour Plan：整体思路与使用方法

## 0. 这个计划解决什么问题

我现在的主要问题不是“完全不会编程”，而是：

- 能理解论文和方法；
- 能用 Codex 推动工程；
- 但是自己手写核心模块时，容易不知道从哪里开始；
- 对 PyTorch 张量 shape、训练循环、模块实现、测试验证的手感不够稳定。

所以这个计划的目标不是刷题，也不是从零造一个大模型工程，而是：

> 每天 1 小时，手写一个大模型相关的小模块。  
> 每个模块必须有输入、有输出、有测试、有可运行结果。  
> Codex 只负责给脚手架、检查 bug、设计测试、提问验收，不直接代写完整答案。

---

## 1. 总体原则

### 1.1 每天只做一个小模块

每天的任务必须小到可以在 1 小时内完成，例如：

- softmax
- cross entropy
- layer norm
- single-head attention
- causal mask
- LoRA Linear
- top-k sampling
- patchify / unpatchify
- timestep embedding

不要一上来做完整 Transformer、大型 RAG 系统、完整 diffusion pipeline。

---

### 1.2 每个任务必须有输入输出

每个 `.py` 文件都必须可以直接运行：

```bash
python day01_softmax_ce.py
```

运行后必须打印类似信息：

```text
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 5])
Test passed.
```

也就是说，每个任务必须包含：

```text
1. 明确输入
2. 明确输出
3. shape 检查
4. assert 测试
5. 和 PyTorch 官方实现或数学性质对齐
```

---

### 1.3 Codex 的角色是“教练”，不是“代写者”

Codex 允许做：

```text
1. 给任务卡
2. 给函数接口
3. 给 TODO 骨架
4. 给伪代码
5. 给测试用例
6. 检查我写的代码
7. 解释报错
8. 提问检查我是否理解
```

Codex 不允许做：

```text
1. 直接写完整实现
2. 一次性生成整个文件答案
3. 替我填所有 TODO
4. 在我没写之前直接优化代码
5. 用复杂工程掩盖基础实现
```

我每天的目标是：

> 先自己写，再让 Codex 检查。  
> 先理解 shape，再追求代码优雅。  
> 先跑通最小 demo，再扩展功能。

---

## 2. 每天 1 小时怎么用

每天固定按照这个节奏：

```text
第 1 步：10 分钟，看任务卡
第 2 步：35 分钟，自己填 TODO / 写实现
第 3 步：10 分钟，运行测试、修 bug
第 4 步：5 分钟，写当天总结
```

当天总结只需要写 5 行：

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出 shape：
- 最容易错的地方：
- 明天要复习的问题：
```

---

## 3. 每天的文件结构

建议仓库结构：

```text
llm-coding-one-hour/
├── README.md
├── stages/
│   ├── stage01_tensor_pytorch/
│   ├── stage02_transformer/
│   ├── stage03_generation_decoding/
│   ├── stage04_lora_peft/
│   ├── stage05_rag_agent/
│   ├── stage06_diffusion_dit/
│   └── stage07_mini_projects/
├── notes/
│   ├── week01.md
│   ├── week02.md
│   └── daily_logs.md
└── tests/
```

每日文件命名：

```text
day01_softmax_ce.py
day02_layernorm.py
day03_embedding.py
day04_single_head_attention.py
```

每个文件建议包含：

```python
"""
Day X: Module Name

Goal:
Input:
Output:
Check:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    pass


def test():
    pass


if __name__ == "__main__":
    test()
```

---

# 4. Stage 路线图

下面是整个计划的阶段安排。每个 stage 可以用 1 到 2 周完成，不需要赶进度。

---

## Stage 1：PyTorch Tensor 与基础函数

目标：

> 让自己熟悉 tensor、shape、广播、索引、loss、基础训练。

核心能力：

- 知道 `[B, C]`、`[B, T, C]`、`[B, C, H, W]` 分别是什么意思；
- 能手写常见函数；
- 能和 PyTorch 官方实现对齐；
- 能写简单 assert 检查。

建议任务：

| Day | 模块 | 输入 | 输出 | 验收 |
|---|---|---|---|---|
| 1 | softmax + cross entropy | logits, labels | probs, loss | 对齐 `F.cross_entropy` |
| 2 | layer norm | `[B, T, C]` | `[B, T, C]` | mean≈0, std≈1 |
| 3 | embedding lookup | token ids | embeddings | shape 正确 |
| 4 | positional embedding | positions | position vectors | 不同位置不同 |
| 5 | linear classifier | toy data | logits | loss 能下降 |
| 6 | simple MLP | toy data | logits | 可 backward |
| 7 | mini review project | toy classification | prediction | 训练闭环跑通 |

---

## Stage 2：Transformer 核心模块

目标：

> 从零手写最小 Transformer 相关模块，真正理解 attention、mask、FFN、residual、LayerNorm。

核心能力：

- 会写 Q/K/V；
- 会写 attention score；
- 会写 causal mask；
- 会处理 multi-head 的 reshape / transpose；
- 会组合 Transformer block。

建议任务：

| Day | 模块 | 输入 | 输出 | 验收 |
|---|---|---|---|---|
| 8 | single-head attention | `[B,T,C]` | `[B,T,C]`, attn | attention 每行和为 1 |
| 9 | causal mask | scores | masked scores | 未来 token 不可见 |
| 10 | multi-head attention | `[B,T,C]` | `[B,T,C]` | head 维度正确 |
| 11 | FFN | `[B,T,C]` | `[B,T,C]` | shape 不变 |
| 12 | residual + norm | hidden | hidden | 可 backward |
| 13 | transformer block | `[B,T,C]` | `[B,T,C]` | forward/backward |
| 14 | tiny GPT | token ids | logits | loss 下降，可生成字符 |

---

## Stage 3：LLM 生成与推理

目标：

> 理解大模型推理时的 decoding、采样、KV cache。

核心能力：

- 理解 logits 到 token 的过程；
- 能实现 greedy / temperature / top-k / top-p；
- 理解 KV cache 为什么能加速；
- 能做输出一致性检查。

建议任务：

| Day | 模块 | 输入 | 输出 | 验收 |
|---|---|---|---|---|
| 15 | greedy decoding | logits | token id | 取最大概率 |
| 16 | temperature sampling | logits, temperature | sampled token | 温度影响分布 |
| 17 | top-k sampling | logits, k | sampled token | 只从 top-k 采样 |
| 18 | top-p sampling | logits, p | sampled token | 累积概率截断 |
| 19 | repetition penalty | history, logits | adjusted logits | 重复 token 降权 |
| 20 | simple KV cache | K/V cache | updated cache | cache shape 正确 |
| 21 | full attention vs cache | token sequence | logits | 输出近似一致 |

---

## Stage 4：LoRA / PEFT / 微调基础

目标：

> 理解参数高效微调，不再只是会调用 PEFT，而是知道 LoRA 改了什么。

核心能力：

- 会写普通 Linear；
- 会写 LoRA Linear；
- 知道冻结原参数、只训练低秩矩阵；
- 会统计可训练参数量；
- 会保存和加载 LoRA 权重。

建议任务：

| Day | 模块 | 输入 | 输出 | 验收 |
|---|---|---|---|---|
| 22 | manual linear | x | y | 对齐 `nn.Linear` |
| 23 | LoRA Linear | x | y | 输出 shape 不变 |
| 24 | freeze base train LoRA | toy data | loss | 只有 LoRA 有梯度 |
| 25 | count params | model | numbers | trainable 比例正确 |
| 26 | attention qkv LoRA | hidden | output | q/k/v 可挂 LoRA |
| 27 | save/load LoRA | checkpoint | same output | reload 后一致 |
| 28 | mini PEFT demo | toy task | before/after | 微调有效 |

---

## Stage 5：RAG / Agent / Tool Calling 小工程

目标：

> 补充大模型应用工程能力，特别是检索、prompt 拼接、工具路由。

核心能力：

- 会文本切 chunk；
- 会写简单检索；
- 会拼 prompt；
- 会做最小 tool router；
- 会输出 source 和 answer。

建议任务：

| Day | 模块 | 输入 | 输出 | 验收 |
|---|---|---|---|---|
| 29 | text chunking | long text | chunks | chunk 数正确 |
| 30 | BM25 retrieval | query, docs | top-k docs | 能检索关键词 |
| 31 | embedding retrieval | vectors | top-k docs | cosine 相似度正确 |
| 32 | rerank | query, candidates | reranked docs | 顺序合理 |
| 33 | prompt builder | question, context | prompt | 格式清晰 |
| 34 | tool router | user query | tool name | 分类正确 |
| 35 | mini RAG | md folder, question | answer + sources | 有检索依据 |

---

## Stage 6：Diffusion / DiT / Flow Matching 基础

目标：

> 服务自己的生成模型、DiT、Flow Matching、驾驶世界模型方向。

核心能力：

- 理解加噪；
- 理解 timestep embedding；
- 理解 patchify / unpatchify；
- 理解 AdaLN；
- 理解 DiT block；
- 能写 2D toy diffusion / flow matching demo。

建议任务：

| Day | 模块 | 输入 | 输出 | 验收 |
|---|---|---|---|---|
| 36 | add Gaussian noise | image/tensor | noisy tensor | 噪声强度可控 |
| 37 | DDPM forward process | x0, t | xt | shape 正确 |
| 38 | timestep embedding | t | embedding | 不同 t 不同 |
| 39 | patchify/unpatchify | latent image | tokens/image | 可还原 |
| 40 | AdaLN | x, condition | modulated x | shape 不变 |
| 41 | mini DiT block | tokens, t emb | tokens | forward/backward |
| 42 | 2D toy diffusion/FM | 2D points | generated points | 可视化轨迹 |

---

## Stage 7：每周 Mini Project

每周最后一天或者每两周做一个小展示。

建议项目：

| Project | 输入 | 输出 |
|---|---|---|
| tiny text classifier | sentence | label |
| tiny GPT | text prefix | generated text |
| sampling playground | logits | sampled tokens |
| LoRA toy fine-tuning | toy data | before/after loss |
| mini RAG | markdown folder | answer + sources |
| DiT patchify demo | image/latent | patch tokens + reconstructed image |
| 2D diffusion/FM demo | random noise | generated 2D distribution |

每个 mini project 都要包含：

```text
1. README
2. 可运行脚本
3. 输入样例
4. 输出样例
5. 测试结果
6. 一段总结
```

---

# 5. Codex 使用规则

以后每次让 Codex 帮忙，都可以先复制下面这段：

```text
你是我的 coding 教练，不是代写者。

请遵守：
1. 不要直接给完整实现。
2. 只给任务目标、输入输出、伪代码、函数骨架、TODO、测试用例。
3. 我会自己填 TODO。
4. 我写完后，你再帮我检查 bug。
5. 检查时不要重写整个文件，只指出问题、原因和最小修改建议。
6. 每次任务都必须有 assert 测试。
7. 每个任务都必须能通过 `python xxx.py` 直接运行。
8. 优先训练 PyTorch tensor、Transformer、LLM 推理、LoRA、RAG、Diffusion/DiT 相关基础能力。
```

---

# 6. 每个任务的完成标准

一个任务只有满足下面条件，才算完成：

```text
1. 我自己填完核心实现。
2. 脚本可以直接运行。
3. 至少有 2 个 assert。
4. 打印输入输出 shape。
5. 没有 NaN。
6. 如果有官方实现，和官方实现对齐。
7. 我能用自己的话说出每一步在干什么。
```

---

# 7. 长期目标

连续 30 天后，希望达到：

```text
1. 能独立写 softmax、CE、LayerNorm、attention、FFN、Transformer block。
2. 能看懂 HuggingFace / diffusers / PEFT 里的核心模块。
3. 能定位常见 shape bug。
4. 能写最小训练循环和测试脚本。
5. 能把论文里的公式翻译成 PyTorch 模块。
6. 能更好地指挥 Codex，而不是完全依赖 Codex。
```

最终目标不是“完全不用 AI 写代码”，而是：

> 我知道代码应该怎么写，AI 只是帮我加速。  
> 我能判断 AI 写得对不对，也能自己修核心 bug。
