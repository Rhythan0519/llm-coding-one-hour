# Day 31：Label Embedding / Classifier-Free Guidance

## 1. 今日目标

今天实现：

```python
LabelEmbedder
classifier_free_guidance(...)
```

这一天先不追求大模型效果，只学习 class condition 和 CFG 的数据流。

---

## 2. CFG 核心公式

```text
guided = uncond + scale * (cond - uncond)
```

含义：

```text
cond: 带 class label 的预测
uncond: 使用 null label 的预测
scale: 条件增强强度
```

训练时会随机把一部分 label 替换成 null label，让同一个模型同时学会 conditional 和 unconditional 两种预测。

---

## 3. 关键检查

```text
1. label embedding 输出 shape 是 [B, D]。
2. token_drop 能把指定 label 替换成 null label。
3. CFG 输出 shape 和模型输出 shape 一致。
4. CFG 数值符合公式。
```

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day31_label_embedding_cfg.py
```

---

## 5. 今日理解问题

```text
1. null label 为什么要占用 num_classes 这个额外 id？
2. CFG 的 scale 太大可能会发生什么？
3. 为什么训练时要随机 drop label？
```
