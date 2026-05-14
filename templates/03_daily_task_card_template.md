# Daily Task Card Template：每天具体任务卡怎么写

## 0. 使用方式

每天开始前，把这个模板喂给 Codex，让它根据当天主题生成任务卡。

关键要求：

> Codex 只能给任务卡、伪代码、函数骨架、TODO、测试。  
> Codex 不能直接给完整实现。  
> 我自己填 TODO。  
> 写完后再让 Codex 检查。

---

# 1. 给 Codex 的固定提示词

每天开始时，把下面这段复制给 Codex：

```text
你是我的 LLM coding 教练，不是代写者。

今天我要练习的模块是：
[填写模块名称]

请你为我生成一张 daily task card。

要求：
1. 不要给完整实现。
2. 只给目标、背景、输入、输出、shape、公式、伪代码、函数骨架、TODO、测试用例。
3. 代码骨架中核心实现必须用 TODO 或 None 留空。
4. 测试代码可以完整给出。
5. 每个任务必须能保存成一个 `.py` 文件并直接运行。
6. 必须包含至少 2 个 assert。
7. 如果有 PyTorch 官方实现，必须安排和官方实现对齐。
8. 必须在最后给我 3 到 5 个理解问题。
9. 不要替我填 TODO。
```

---

# 2. Daily Task Card 标准结构

每天的任务卡应该长这样：

```markdown
# Day X：模块名称

## 1. 今日目标

## 2. 这个模块在大模型里有什么用

## 3. 输入输出

## 4. 核心公式或核心逻辑

## 5. 伪代码

## 6. 代码骨架

## 7. 测试要求

## 8. 运行方式

## 9. 预期输出

## 10. 常见错误

## 11. 扩展任务

## 12. 今日理解问题

## 13. 今日总结模板
```

---

# 3. Daily Task Card 模板

下面是可复用模板。

```markdown
# Day X：模块名称

## 1. 今日目标

今天要实现：

```python
function_or_class_name(...)
```

目标是：

```text
用自己的代码实现 xxx，并通过测试验证它是正确的。
```

---

## 2. 这个模块在大模型里有什么用

这个模块通常用于：

```text
1. xxx
2. xxx
3. xxx
```

例如在 LLM / Transformer / Diffusion / DiT 中，它的作用是：

```text
说明这个模块在真实模型中的位置。
```

---

## 3. 输入输出

### 输入

```text
输入变量名：
shape：
含义：
```

示例：

```python
x = torch.randn(...)
```

### 输出

```text
输出变量名：
shape：
含义：
```

---

## 4. 核心公式或核心逻辑

如果有公式，写在这里：

$$
公式
$$

用人话解释：

```text
第一步：
第二步：
第三步：
```

---

## 5. 伪代码

```text
1. xxx
2. xxx
3. xxx
4. return xxx
```

---

## 6. 代码骨架

要求：

```text
1. 核心实现留 TODO
2. 不要直接填完整答案
3. 测试函数可以完整
```

代码骨架：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_function(...):
    """
    Args:
        ...

    Returns:
        ...
    """

    # TODO 1:
    a = None

    # TODO 2:
    b = None

    # TODO 3:
    output = None

    return output


def test():
    torch.manual_seed(0)

    # prepare input
    ...

    # run my implementation
    ...

    # check shape
    assert ...

    # check numerical correctness or property
    assert ...

    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

必须通过：

```text
1. shape 测试
2. 数值正确性测试
3. NaN 检查
4. 如果有官方实现，对齐官方实现
```

示例：

```python
assert output.shape == expected_shape
assert not torch.isnan(output).any()
assert torch.allclose(my_output, torch_output, atol=1e-6)
```

---

## 8. 运行方式

保存为：

```text
dayXX_module_name.py
```

运行：

```bash
python dayXX_module_name.py
```

---

## 9. 预期输出

终端应该看到类似：

```text
Input shape: ...
Output shape: ...
Test passed.
```

---

## 10. 常见错误

今天容易错的地方：

```text
1. shape 维度写反
2. keepdim 忘记设置
3. dim 参数写错
4. tensor indexing 错误
5. 忘记检查 NaN
```

---

## 11. 扩展任务

基础版完成后，可以扩展：

```text
1. 支持更多输入 shape
2. 和官方实现对齐
3. 加入可视化
4. 加入更多测试 case
5. 重写一遍不看答案
```

---

## 12. 今日理解问题

请回答：

```text
1. 这个模块的输入 shape 是什么？
2. 输出 shape 是什么？
3. 哪一步最容易写错？
4. 为什么要这样实现？
5. 它在大模型中出现在哪里？
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
```

---

# 4. 示例任务卡：Day 1 Softmax + Cross Entropy

下面是一个完整示例。  
注意：核心实现必须留空，不直接给答案。

```markdown
# Day 1：Softmax 与 Cross Entropy

## 1. 今日目标

今天实现两个函数：

```python
my_softmax(logits)
my_cross_entropy(logits, labels)
```

目标：

```text
理解 logits 如何变成概率，以及分类 loss 是怎么计算的。
```

---

## 2. 这个模块在大模型里有什么用

在大模型中，最后一层通常输出：

```text
logits: [B, vocab_size]
```

每个位置表示模型对某个 token 的未归一化分数。

softmax 把 logits 转成概率分布。  
cross entropy 用来衡量模型预测的概率和真实 token 之间的差距。

---

## 3. 输入输出

### 输入

```text
logits: [B, C]
labels: [B]
```

其中：

```text
B = batch size
C = class number 或 vocab size
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
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

数值稳定版本：

$$
p_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

Cross Entropy：

$$
\mathrm{loss} = -\log p_{\mathrm{label}}
$$

---

## 5. 伪代码

softmax：

```text
1. 对每一行 logits 找最大值
2. logits 减去每一行最大值
3. 对结果取 exp
4. 每一行求和
5. exp / sum_exp 得到概率
```

cross entropy：

```text
1. 调用 my_softmax 得到 probs
2. 根据 labels 取出正确类别概率
3. 对正确类别概率取 log
4. 加负号
5. 对 batch 求平均
```

---

## 6. 代码骨架

```python
import torch
import torch.nn.functional as F


def my_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: [B, C]

    Returns:
        probs: [B, C]
    """

    # TODO 1: find max value for each row, keep shape [B, 1]
    max_values = None

    # TODO 2: subtract max_values from logits
    stable_logits = None

    # TODO 3: apply exp
    exp_logits = None

    # TODO 4: sum exp values for each row, keep shape [B, 1]
    sum_exp = None

    # TODO 5: normalize
    probs = None

    return probs


def my_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Args:
        logits: [B, C]
        labels: [B]

    Returns:
        loss: scalar
    """

    # TODO 1: compute probabilities
    probs = None

    # TODO 2: get batch size
    B = None

    # TODO 3: get probability of the correct class for each sample
    correct_probs = None

    # TODO 4: compute negative log likelihood
    losses = None

    # TODO 5: average over batch
    loss = None

    return loss


def test_softmax():
    torch.manual_seed(0)

    logits = torch.randn(3, 5)
    probs = my_softmax(logits)

    print("Input logits shape:", logits.shape)
    print("Output probs shape:", probs.shape)
    print("Row sums:", probs.sum(dim=1))

    assert probs.shape == logits.shape
    assert not torch.isnan(probs).any()
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)

    torch_probs = F.softmax(logits, dim=1)
    assert torch.allclose(probs, torch_probs, atol=1e-6)

    print("Softmax test passed.")


def test_cross_entropy():
    torch.manual_seed(0)

    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 2, 3, 5])

    my_loss = my_cross_entropy(logits, labels)
    torch_loss = F.cross_entropy(logits, labels)

    print("My loss:", my_loss.item())
    print("Torch loss:", torch_loss.item())

    assert not torch.isnan(my_loss)
    assert torch.allclose(my_loss, torch_loss, atol=1e-6)

    print("Cross entropy test passed.")


if __name__ == "__main__":
    test_softmax()
    test_cross_entropy()
```

---

## 7. 测试要求

必须通过：

```text
1. probs.shape == logits.shape
2. softmax 每一行求和等于 1
3. probs 没有 NaN
4. my_softmax 和 F.softmax 对齐
5. my_cross_entropy 和 F.cross_entropy 对齐
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
Input logits shape: torch.Size([3, 5])
Output probs shape: torch.Size([3, 5])
Row sums: tensor([1.0000, 1.0000, 1.0000])
Softmax test passed.
My loss: ...
Torch loss: ...
Cross entropy test passed.
```

---

## 10. 常见错误

```text
1. torch.max 返回的是 values 和 indices，不要直接当 tensor 用。
2. 忘记 keepdim=True，会导致广播 shape 不直观。
3. softmax 的 dim 写错。
4. correct_probs 的索引方式写错。
5. 没有加数值稳定处理，exp 可能溢出。
```

---

## 11. 扩展任务

完成基础版后，可以继续：

```text
1. 支持 dim 参数。
2. 测试更大的 logits，比如乘以 100。
3. 写一个不稳定版本，对比为什么会溢出。
4. 不看代码重新写一遍。
```

---

## 12. 今日理解问题

```text
1. softmax 为什么要减去最大值？
2. logits 和 probability 的区别是什么？
3. cross entropy 为什么只取正确类别的概率？
4. labels 的 shape 为什么是 [B] 而不是 [B, C]？
5. `probs[torch.arange(B), labels]` 这行是什么意思？
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
```

---

# 5. 写完代码后，让 Codex 怎么检查

当我写完 TODO 后，把代码贴给 Codex，并使用：

```text
请你检查我刚写的代码。

要求：
1. 不要重写完整代码。
2. 只指出 bug、shape 问题、数值稳定问题。
3. 每个问题给出原因。
4. 每个问题只给最小修改建议。
5. 如果代码是对的，请不要强行优化。
6. 最后问我 5 个理解问题。
```

---

# 6. 如果卡住了，怎么问 Codex

不要问：

```text
帮我写完整答案。
```

应该问：

```text
我卡在 TODO 3，不知道这个 tensor 的 shape 应该是什么。
请你只解释 shape，不要给完整代码。
```

或者：

```text
我这里报错了，请你解释错误原因，并提示我应该检查哪个维度。
不要直接给完整实现。
```

或者：

```text
请你给我一个更小的输入例子，让我手算一遍。
```

---

# 7. 每日任务完成标准

每天任务必须满足：

```text
1. 文件能直接运行。
2. 核心实现是我自己写的。
3. 至少两个 assert 通过。
4. 打印输入输出 shape。
5. 没有 NaN。
6. 如果有官方实现，和官方实现对齐。
7. 我能回答今日理解问题。
8. 我能在第二天不看答案重写核心部分。
```
