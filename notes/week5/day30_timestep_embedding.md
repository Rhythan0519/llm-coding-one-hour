# Day 30：Timestep Embedding

## 1. 今日目标

今天实现：

```python
sinusoidal_timestep_embedding(timesteps, dim)
TimestepMLP
```

Diffusion model 每一步看到的 noisy image 噪声强度不同，所以模型必须知道当前 timestep。

---

## 2. 这个模块在 DiT 里有什么用

```text
1. timestep t 先变成 sinusoidal embedding。
2. embedding 再经过 MLP 投影到 hidden_size。
3. DiT block 用这个 condition 调制 LayerNorm。
```

也就是：

```text
t -> time embedding -> condition vector -> adaLN
```

---

## 3. 关键检查

```text
1. 输入 timesteps 是 [B]。
2. 输出 embedding 是 [B, dim]。
3. 相同 timestep 的 embedding 必须相同。
4. 不同 timestep 的 embedding 应该不同。
5. MLP 输出可以 backward。
```

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day30_timestep_embedding.py
```

---

## 5. 今日理解问题

```text
1. timestep embedding 和 position embedding 有什么区别？
2. 为什么不能只把 timestep 当成一个普通整数直接喂给模型？
3. sinusoidal embedding 里 cos / sin 的作用是什么？
```
