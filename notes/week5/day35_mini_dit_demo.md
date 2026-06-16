# Day 35：Mini DiT Demo

## 1. 今日目标

今天跑通本周 mini demo：

```python
run_mini_dit_demo()
```

它会创建 toy images，固定一批 timestep 和 noise，训练 MiniDiT 去预测这批 noise，并检查 loss 是否下降。

---

## 2. Demo 流程

```text
1. 创建 8 张 8x8 toy images。
2. 随机采样 timestep。
3. 采样 target_noise。
4. 用 q_sample 得到 noisy_images。
5. MiniDiT(noisy_images, timesteps) 预测 noise。
6. 用 MSE loss 训练若干步。
7. 对比 initial_loss 和 final_loss。
```

---

## 3. 验收标准

```text
1. clean_images shape 是 [8,1,8,8]。
2. noisy_images shape 是 [8,1,8,8]。
3. timesteps shape 是 [8]。
4. final_loss < initial_loss。
```

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day35_mini_dit_demo.py
```

写完后的参考输出应类似：

```text
Initial loss: 1.2767460346221924
Final loss: 0.0462687611579895
Test passed.
```

---

## 5. 今日理解问题

```text
1. 这个 demo 为什么只证明模型能 overfit 一批 toy data？
2. 真实 DiT 还需要哪些训练工程？
3. 如果 final_loss 不下降，应该先检查哪些 shape 和梯度？
```
