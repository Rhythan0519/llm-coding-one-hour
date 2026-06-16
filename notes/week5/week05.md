# Week 5：从零写 DiT / Diffusion Transformer（2026-06-15 ~ 2026-06-21）

## 1. 本周目标

本周从最小可运行代码开始学习 DiT：把图像切成 patch tokens，加入 timestep / class 条件，用 adaptive LayerNorm 改造 Transformer block，最后跑通一个 tiny diffusion denoising demo。

本周结束后，我应该能够：

- 解释图像 `[B,C,H,W]` 怎么变成 token 序列 `[B,N,D]`；
- 写出 `patchify` / `unpatchify`；
- 写出 diffusion timestep 的 sinusoidal embedding；
- 理解 classifier-free guidance 的 conditional / unconditional 两路预测；
- 写出带 adaLN 的 DiT block；
- 组装一个 mini DiT，输入 noisy image 和 timestep，输出预测噪声；
- 跑通最小 diffusion training step，让 loss 下降。

---

## 2. 本周核心能力

```text
1. 理解 patch_size、num_patches、patch_dim 的 shape 关系
2. 理解 timestep embedding 为什么是条件信息
3. 理解 class label dropout 和 CFG
4. 理解 DiT block: self-attention + MLP + adaLN modulation
5. 理解 diffusion 训练目标: predict noise
6. 能用 assert 检查 shape、梯度和 loss 下降
```

---

## 3. 每日任务安排

说明：`code/week5` 是练习区，代码文件由我自己创建和填写。下面的 Python 文件名只是建议命名，笔记只负责说明目标、shape 和验收点。

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 29 | [day29_dit_patchify.md](day29_dit_patchify.md) | patchify / unpatchify | image `[B,C,H,W]` | patches `[B,N,D]` | 还原后等于原图 |
| Day 30 | [day30_timestep_embedding.md](day30_timestep_embedding.md) | timestep embedding | timesteps `[B]` | embedding `[B,D]` | 相同 timestep embedding 相同 |
| Day 31 | [day31_label_embedding_cfg.md](day31_label_embedding_cfg.md) | label embedding / CFG | labels, model outputs | guided output | CFG 公式正确 |
| Day 32 | [day32_dit_block.md](day32_dit_block.md) | DiT block | tokens + condition | tokens | shape 不变，attention 合法 |
| Day 33 | [day33_mini_dit.md](day33_mini_dit.md) | mini DiT | noisy image + timestep | predicted noise image | 输入输出 shape 一致 |
| Day 34 | [day34_diffusion_training_step.md](day34_diffusion_training_step.md) | diffusion training step | clean image | MSE noise loss | 可 backward |
| Day 35 | [day35_mini_dit_demo.md](day35_mini_dit_demo.md) | mini DiT demo | toy images | before/after loss | loss 明显下降 |

---

## 4. 本周 mini demo

实现一个 tiny DiT denoising demo：

```text
clean image x0
+ random timestep t
+ Gaussian noise eps
=> noisy image xt
=> MiniDiT(xt, t) predicts eps
=> MSE(pred_eps, eps)
```

验收：

```text
1. image -> patches -> image 的 shape 链路正确。
2. timestep condition 能进入 DiT block。
3. MiniDiT 输出和输入 image shape 一致。
4. diffusion loss 可以 backward。
5. 在固定 toy batch 上训练后 loss 下降。
```

---

## 5. 本周复盘问题

```text
1. patch_size 变大时，num_patches 和 patch_dim 分别怎么变？
2. 为什么 DiT 不用 causal mask？
3. timestep embedding 和 position embedding 分别表达什么？
4. adaLN 里的 shift、scale、gate 各自控制什么？
5. diffusion 训练为什么常预测 noise，而不是直接预测 clean image？
6. classifier-free guidance 为什么需要 null label？
7. mini demo 和真实 DiT 还差哪些东西？
```
