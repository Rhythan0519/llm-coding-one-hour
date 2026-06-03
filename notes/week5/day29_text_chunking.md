# Day 29：Text Chunking

## 1. 今日目标

今天实现：

```python
chunk_text(text, chunk_size, overlap)
```

目标是把长文本按词切成固定大小的 chunks，并支持相邻 chunk 之间有 overlap。

---

## 2. 这个模块在大模型里有什么用

RAG 不能直接把超长文档全部塞进 prompt，通常要先切成 chunks：

```text
1. 每个 chunk 控制在模型能处理的长度内
2. overlap 保留跨边界信息
3. 后续检索以 chunk 为最小单位
```

---

## 3. 输入输出

输入：

```text
text: str
chunk_size: int，每个 chunk 最多多少个 word
overlap: int，相邻 chunk 重叠多少个 word
```

输出：

```text
chunks: list[str]
```

---

## 4. 核心逻辑

```text
step = chunk_size - overlap
start = 0
while start < len(words):
    end = start + chunk_size
    chunk = words[start:end]
    start += step
```

要求：

```text
1. chunk_size 必须 > 0
2. overlap 必须 >= 0
3. overlap 必须 < chunk_size
4. 空文本返回 []
```

---

## 5. 伪代码

```text
1. 检查 chunk_size 和 overlap
2. 用 text.split() 得到 words
3. 如果 words 为空，返回 []
4. step = chunk_size - overlap
5. 从 start=0 开始滑动窗口
6. 把每个窗口用空格拼回字符串
7. 返回 chunks
```

---

## 6. 代码骨架

```python
"""
Day 29: Text Chunking

Goal:
Split a long text into word chunks with overlap.

Input:
text: str
chunk_size: int
overlap: int

Output:
chunks: list[str]

Check:
Chunk count, max chunk length, overlap, empty input, invalid overlap.
"""


def chunk_text(text, chunk_size, overlap):
    """
    Args:
        text: Input text.
        chunk_size: Max number of words per chunk.
        overlap: Number of words shared by neighboring chunks.

    Returns:
        A list of chunk strings.
    """

    # TODO 1: validate arguments
    # Hint:
    # - chunk_size must be > 0
    # - overlap must be >= 0
    # - overlap must be < chunk_size

    # TODO 2: split text into words

    # TODO 3: slide window over words
    # Hint:
    # - step = chunk_size - overlap
    # - keep the final shorter chunk

    # TODO 4: return list of chunk strings
    return None


def test():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert isinstance(chunks, list)
    assert len(chunks) == 5
    assert all(len(chunk.split()) <= 6 for chunk in chunks)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert chunk_text("", chunk_size=6, overlap=2) == []

    raised = False
    try:
        chunk_text(text, chunk_size=4, overlap=4)
    except ValueError:
        raised = True
    assert raised is True

    print("Input words:", len(words))
    print("Chunk count:", len(chunks))
    print("First chunk:", chunks[0])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. chunk 数正确
2. 每个 chunk 长度不超过 chunk_size
3. 相邻 chunk 的 overlap 正确
4. 空文本返回 []
5. overlap >= chunk_size 时抛出 ValueError
```

---

## 8. 运行方式

```bash
python code/week5/day29/text_chunking.py
```

---

## 9. 预期输出

```text
Input words: 20
Chunk count: 5
First chunk: w0 w1 w2 w3 w4 w5
Test passed.
```

---

## 10. 常见错误

```text
1. step 写成 overlap，导致窗口移动太慢
2. overlap 等于 chunk_size 时死循环
3. 最后一个不足 chunk_size 的片段被漏掉
4. 空字符串返回 [""] 而不是 []
```

---

## 11. 扩展任务

```text
1. 改成按字符切分
2. 支持保留 source 和 chunk_id
3. 支持 separator 参数
```

---

## 12. 今日理解问题

```text
1. 为什么 RAG 需要先切 chunk？
2. overlap 太大有什么代价？
3. 最后一个 chunk 不足 chunk_size 时应该保留吗？
4. 按词切分和按字符切分有什么区别？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape / 长度：
- 输出 chunks：
- 最容易错的地方：
- 明天要复习的问题：
```
