# Day 32：Rerank

## 1. 今日目标

今天实现：

```python
rerank(query, candidates, top_k)
```

目标是对粗检索得到的候选文档再排序，把更符合 query 的候选排到前面。

---

## 2. 这个模块在大模型里有什么用

RAG 通常分两步：

```text
1. retrieval：快速召回较多候选
2. rerank：用更细的相关性判断重新排序
```

今天不用真实 cross-encoder，先用关键词 overlap 写一个可解释 reranker。

---

## 3. 输入输出

输入：

```text
query: str
candidates: list[dict]
每个 candidate 包含 text、source、retrieval_score
top_k: int
```

输出：

```text
reranked: list[dict]
每个 dict 新增 rerank_score
```

---

## 4. 核心逻辑

```text
query_terms = set(tokenize(query))
doc_terms = set(tokenize(doc))
overlap = query_terms & doc_terms
rerank_score = len(overlap) + 0.1 * retrieval_score
```

---

## 5. 伪代码

```text
1. tokenize query
2. 遍历 candidates
3. 计算每个 candidate 的 keyword overlap
4. 融合 retrieval_score
5. 按 rerank_score 降序排序
6. 返回 top_k
```

---

## 6. 代码骨架

```python
"""
Day 32: Rerank

Goal:
Rerank retrieved candidates using a simple keyword-overlap score.

Input:
query: str
candidates: list[dict]

Output:
reranked: list[dict] with rerank_score.

Check:
Top-k length, best source, score order, no in-place mutation.
"""


def tokenize(text):
    """
    Args:
        text: Raw text.

    Returns:
        Lowercased tokens.
    """

    # TODO 1: lowercase and split
    return None


def keyword_overlap_score(query, doc):
    """
    Args:
        query: User question.
        doc: Candidate document text.

    Returns:
        Number of overlapping unique tokens.
    """

    # TODO 2: compute overlap score
    # Hint:
    # - tokenize query and doc
    # - convert both to set
    # - return len(query_terms & doc_terms)
    return None


def rerank(query, candidates, top_k=3):
    """
    Args:
        query: User query.
        candidates: list of dicts with text, source, retrieval_score.
        top_k: number of candidates to return.

    Returns:
        Reranked candidates with rerank_score.
    """

    # TODO 3: add rerank_score to each candidate
    # Hint: create a new dict instead of mutating candidates in-place

    # TODO 4: sort by rerank_score descending
    return None


def test():
    query = "how does lora save trainable parameters"
    candidates = [
        {
            "text": "BM25 is a sparse retrieval method for keyword search.",
            "source": "bm25.md",
            "retrieval_score": 8.0,
        },
        {
            "text": "LoRA saves trainable parameters by learning low rank adapter matrices.",
            "source": "lora.md",
            "retrieval_score": 5.0,
        },
        {
            "text": "Prompt builders combine context and question.",
            "source": "prompt.md",
            "retrieval_score": 3.0,
        },
    ]
    original = [dict(candidate) for candidate in candidates]

    results = rerank(query, candidates, top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert "rerank_score" in results[0]
    assert results[0]["source"] == "lora.md"
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert candidates == original

    print("Query:", query)
    print("Top source:", results[0]["source"])
    print("Top rerank score:", results[0]["rerank_score"])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 返回长度等于 top_k
2. 结果包含 rerank_score
3. 更匹配 query 的候选排在第一
4. 原始 candidates 不被原地破坏
```

---

## 8. 运行方式

```bash
python code/week5/day32/rerank.py
```

---

## 9. 预期输出

```text
Query: how does lora save trainable parameters
Top source: lora.md
Top rerank score: ...
Test passed.
```

---

## 10. 常见错误

```text
1. 把 candidates 原地修改，测试之间互相影响
2. set overlap 忽略大小写处理
3. retrieval_score 缺失时报 KeyError
4. 排序方向写反
```

---

## 11. 扩展任务

```text
1. 返回 matched_terms
2. 加入 phrase match 分数
3. 对重复词使用 count overlap 而不是 set overlap
```

---

## 12. 今日理解问题

```text
1. 为什么 rerank 通常不直接处理全部文档？
2. retrieval_score 和 rerank_score 应该怎么融合？
3. 关键词 overlap 的局限是什么？
4. 为什么要避免原地修改 candidates？
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
