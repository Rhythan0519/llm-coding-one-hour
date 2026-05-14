# Day 1：Softmax 与 Cross Entropy

## 1. 今日目标

今天实现两个函数：

```python
my_softmax(logits)
my_cross_entropy(logits, labels)
```

目标是理解 logits 如何变成概率，以及分类 loss 是怎么计算的。

---

## 2. 这个模块在大模型里有什么用

在大模型中，最后一层通常输出：

```text
logits: [B, vocab_size]
```

softmax 把 logits 转成概率分布。cross entropy 用来衡量模型预测和真实 token 之间的差距。

---

## 3. 输入输出

### 输入

```text
logits: [B, C]
labels: [B]
```

示例：

```python
logits = torch.randn(4, 6)
labels = torch.tensor([0, 2, 3, 5])
```

### 输出

```text
probs: [B, C]
loss: scalar
```

---

## 4. 核心公式

Softmax：

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Cross entropy：

$$
\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B} \log p_{b, y_b}
$$

实现时先减去每行最大值，防止数值溢出。

---

## 5. 伪代码

```text
1. 对 logits 按最后一维减去最大值
2. 取 exp
3. 按最后一维求和并归一化
4. 用 labels 取出正确类别概率
5. 取负对数并求平均
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def my_softmax(logits, dim=-1):
    """
    Args:
        logits: [B, C]
    Returns:
        probs: [B, C]
    """
    # TODO 1: stabilize logits
    stable = None

    # TODO 2: exponentiate
    exp_x = None

    # TODO 3: normalize
    probs = None
    return probs


def my_cross_entropy(logits, labels):
    """
    Args:
        logits: [B, C]
        labels: [B]
    Returns:
        loss: scalar
    """
    # TODO 1: compute probabilities
    probs = my_softmax(logits)

    # TODO 2: pick the correct class probability
    correct_prob = None

    # TODO 3: negative log likelihood
    loss = None
    return loss


def test():
    torch.manual_seed(0)
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 2, 3, 5])

    my_probs = my_softmax(logits)
    ref_probs = F.softmax(logits, dim=-1)
    my_loss = my_cross_entropy(logits, labels)
    ref_loss = F.cross_entropy(logits, labels)

    assert my_probs.shape == logits.shape
    assert torch.allclose(my_probs.sum(dim=-1), torch.ones(logits.size(0)), atol=1e-6)
    assert torch.allclose(my_probs, ref_probs, atol=1e-6)
    assert torch.allclose(my_loss, ref_loss, atol=1e-6)

    print("Input shape:", logits.shape)
    print("Output shape:", my_probs.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. shape 测试
2. 数值正确性测试
3. NaN 检查
4. 和官方实现对齐
```

---

## 8. 运行方式

保存为：

```text
day01_softmax_ce.py
```

运行：

```bash
python day01_softmax_ce.py
```

---

## 9. 预期输出

```text
Input shape: torch.Size([4, 6])
Output shape: torch.Size([4, 6])
Test passed.
```

---

## 10. 常见错误

```text
1. 忘记减最大值
2. dim 写错
3. labels 没有用 long 类型
4. 取错正确类别概率
5. 没检查 NaN
```

---

## 11. 扩展任务

```text
1. 支持任意 dim
2. 支持 batchless 输入
3. 加入 ignore_index
4. 手写 log-softmax 版本
5. 和 F.log_softmax 对齐
```

---

## 12. 今日理解问题

```text
1. 为什么 softmax 要减最大值？
2. 为什么 cross entropy 等于 -log 正确类别概率？
3. 为什么概率和应当接近 1？
4. logits 和 probability 有什么区别？
5. 这个模块在 LLM 中放在哪一层？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出 shape：
- 我最容易错的地方：
- 我现在能不能不看代码重写：
- 明天要复习：
```

