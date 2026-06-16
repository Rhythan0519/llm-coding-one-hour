# Day 34：Diffusion Training Step

## 1. 今日目标

今天实现 diffusion 训练里最核心的一步：

```python
xt = q_sample(x0, t, noise, alphas_cumprod)
loss = mse(model(xt, t), noise)
```

也就是：给 clean image 加噪，然后训练模型预测加进去的 noise。

---

## 2. 前向加噪公式

```text
xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
```

其中：

```text
x0: clean image
xt: noisy image
noise: Gaussian noise
alpha_bar_t: 到 timestep t 为止的累计保留比例
```

---

## 3. 关键检查

```text
1. beta schedule shape 正确。
2. q_sample 输出 shape 和 x0 一致。
3. loss 是 scalar。
4. loss.backward() 后模型参数有梯度。
```

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day34_diffusion_training_step.py
```

---

## 5. 今日理解问题

```text
1. timestep 越大，xt 通常越接近什么？
2. 为什么训练目标可以是预测 noise？
3. beta schedule 控制了什么？
```
