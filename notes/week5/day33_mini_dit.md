# Day 33：Mini DiT

## 1. 今日目标

今天把前几天的模块拼起来：

```python
MiniDiT(noisy_images, timesteps) -> predicted_noise
```

这是一个很小的 DiT，不追求生成质量，只追求结构完整、shape 清楚、可以 backward。

---

## 2. 数据流

```text
image [B,C,H,W]
-> PatchEmbed Conv2d
-> tokens [B,N,hidden]
-> + position embedding
-> timestep embedding + MLP
-> DiT blocks
-> final linear
-> unpatchify
-> predicted noise [B,C,H,W]
```

---

## 3. 关键检查

```text
1. 输入输出 image shape 一致。
2. num_patches 计算正确。
3. position embedding shape 正确。
4. backward 后 patch_embed 和 final_linear 有梯度。
```

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day33_mini_dit.py
```

---

## 5. 今日理解问题

```text
1. PatchEmbed 为什么可以用 Conv2d 实现？
2. final_linear 输出的为什么是 patch_dim，而不是 hidden_size？
3. 为什么输出要 unpatchify 回 image shape？
```
