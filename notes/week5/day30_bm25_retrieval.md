# Day 30：BM25 Retrieval

## 1. 今日目标

今天实现：

```python
bm25_retrieve(query, docs, top_k)
```

目标是写一个最小 BM25 检索器，根据关键词相关性返回 top-k 文档。

---

## 2. 这个模块在大模型里有什么用

BM25 是经典稀疏检索方法，适合关键词匹配明显的问题：

```text
1. query 里有明确术语
2. 文档里有相同关键词
3. 不需要训练 embedding 模型
```

在 RAG 里，它常作为 baseline 或 hybrid retrieval 的一部分。

---

## 3. 输入输出

输入：

```text
query: str
docs: list[str]
top_k: int
```

输出：

```text
results: list[dict]
每个 dict 包含 doc_id、text、score
```

---

## 4. 核心公式

```text
score(q, d) = sum over term in query:
    idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(d) / avgdl))
```

其中：

```text
tf: term 在当前文档中的次数
df: term 出现在多少篇文档里
idf: term 的逆文档频率
avgdl: 平均文档长度
```

---

## 5. 伪代码

```text
1. tokenize query 和所有 docs
2. 统计每个 term 的 df
3. 计算 avg document length
4. 对每篇 doc 计算 BM25 score
5. 按 score 降序排序
6. 返回 top_k
```

---

## 6. 代码骨架

```python
"""
Day 30: BM25 Retrieval

Goal:
Implement a minimal BM25 keyword retriever.

Input:
query: str
docs: list[str]
top_k: int

Output:
results: list[dict] with doc_id, text, score.

Check:
Top-k length, descending scores, keyword-relevant document first.
"""

import math
from collections import Counter


def tokenize(text):
    """
    Args:
        text: Raw text.

    Returns:
        Lowercased tokens.
    """

    # TODO 1: lowercase and split text
    # Hint: start simple with text.lower().split()
    return None


def bm25_score(query_tokens, doc_tokens, doc_freq, num_docs, avg_doc_len, k1=1.5, b=0.75):
    """
    Args:
        query_tokens: list[str]
        doc_tokens: list[str]
        doc_freq: dict[str, int]
        num_docs: total document count
        avg_doc_len: average document length
        k1: term frequency saturation factor
        b: document length normalization factor

    Returns:
        BM25 score for one document.
    """

    # TODO 2: compute BM25 score for one doc
    # Hint:
    # - use Counter(doc_tokens) for tf
    # - use doc_freq[token] for df
    # - skip terms with tf == 0
    # - one common IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    return None


def bm25_retrieve(query, docs, top_k=3):
    """
    Args:
        query: User query.
        docs: Candidate documents.
        top_k: Number of results to return.

    Returns:
        Ranked results with doc_id, text, score.
    """

    # TODO 3: prepare tokens and statistics
    # Hint:
    # - tokenize query and every doc
    # - doc_freq counts in how many docs a term appears
    # - avg_doc_len is average len(doc_tokens)

    # TODO 4: score every doc

    # TODO 5: sort and return top_k dicts
    return None


def test():
    docs = [
        "Python and PyTorch tensors are used for model training.",
        "LoRA reduces trainable parameters with low rank adapters.",
        "RAG retrieves documents before building a prompt.",
        "Diffusion models add noise and learn to denoise.",
    ]
    query = "python tensor"

    results = bm25_retrieve(query, docs, top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert {"doc_id", "text", "score"} <= set(results[0].keys())
    assert results[0]["doc_id"] == 0
    assert results[0]["score"] >= results[1]["score"]

    print("Query:", query)
    print("Top doc:", results[0]["text"])
    print("Top score:", results[0]["score"])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 返回长度等于 top_k
2. 分数按降序排列
3. 包含 query 关键词的文档排在前面
4. 每个结果包含 doc_id、text、score
```

---

## 8. 运行方式

```bash
python code/week5/day30/bm25_retrieval.py
```

---

## 9. 预期输出

```text
Query: python tensor
Top doc: Python and PyTorch tensors are used for model training.
Top score: ...
Test passed.
```

---

## 10. 常见错误

```text
1. df 统计成了 tf
2. 没有按 score 降序排序
3. avg_doc_len 除以 0
4. tokenize 大小写不统一
```

---

## 11. 扩展任务

```text
1. 去掉标点符号
2. 支持中文分词
3. 返回命中的 query terms
```

---

## 12. 今日理解问题

```text
1. BM25 为什么需要 IDF？
2. 文档长度归一化解决什么问题？
3. BM25 和简单关键词计数有什么区别？
4. top_k 过大有什么影响？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入：
- 输出：
- 最容易错的地方：
- 明天要复习的问题：
```
