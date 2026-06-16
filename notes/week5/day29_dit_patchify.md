# Day 29：DiT Patchify

## 1. 今日目标

今天实现：

```python
patchify(images, patch_size)
unpatchify(patches, patch_size, channels, height, width)
```

DiT 的第一步不是直接对 `[B,C,H,W]` 做 Transformer，而是把图像切成 patch token 序列。

---

## 2. 核心 shape

```text
images:  [B, C, H, W]
patches: [B, N, patch_dim]

N = (H / patch_size) * (W / patch_size)
patch_dim = patch_size * patch_size * C
```

例子：

```text
[2, 1, 4, 4], patch_size=2
=> [2, 4, 4]
```

---

## 3. 关键点

```text
1. H 和 W 必须能被 patch_size 整除。
2. patchify 只是重排数据，不应该改变像素值。
3. unpatchify 后应该能完全还原原图。
4. Transformer 后续看到的是 token 序列，不是二维图片。
```

---

## 4. 运行方式

写完代码后运行：

```bash
.venv/bin/python code/week5/day29_dit_patchify.py
```

预期能看到：

```text
Image shape: torch.Size([2, 1, 4, 4])
Patch shape: torch.Size([2, 4, 4])
Test passed.
```

---

## 5. 今日理解问题

```text
1. patch_size 越大，token 数 N 会怎么变？
2. patch_size 越大，每个 token 的维度 patch_dim 会怎么变？
3. 为什么 patchify / unpatchify 最适合先用 torch.arange 测试？
```
