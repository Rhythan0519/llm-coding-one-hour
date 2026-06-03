# Day 31：Embedding Retrieval

## 1. 今日目标

今天实现：

```python
embedding_retrieve(query_vec, doc_vectors, docs, top_k)
```

目标是用 cosine similarity 做最小向量检索。

---

## 2. 这个模块在大模型里有什么用

Embedding retrieval 适合语义相近但关键词不完全相同的场景：

```text
1. query 和 document 都转成向量
2. 用 cosine similarity 计算相似度
3. 返回最相似的 top-k chunks
```

真实 RAG 系统通常会用 embedding model 生成向量，本练习先使用 toy vectors。

---

## 3. 输入输出

输入：

```text
query_vec: [D]
doc_vectors: [N,D]
docs: list[str], 长度 N
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
cosine(q, d) = dot(q, d) / (norm(q) * norm(d))
```

---

## 5. 伪代码

```text
1. 检查 query_vec shape 是 [D]
2. 检查 doc_vectors shape 是 [N,D]
3. 对 query 和 docs 做 normalize
4. scores = doc_vectors_norm @ query_vec_norm
5. 取 top_k
6. 返回 result dicts
```

---

## 6. 代码骨架

```python
"""
Day 31: Embedding Retrieval

Goal:
Retrieve documents by cosine similarity between query and document vectors.

Input:
query_vec: [D]
doc_vectors: [N, D]
docs: list[str]

Output:
results: list[dict] with doc_id, text, score.

Check:
Cosine similarity matches PyTorch, top-k order is correct.
"""

import torch
import torch.nn.functional as F


def cosine_similarity(query_vec, doc_vectors, eps=1e-8):
    """
    Args:
        query_vec: Tensor [D]
        doc_vectors: Tensor [N, D]
        eps: Small number for numerical stability.

    Returns:
        scores: Tensor [N]
    """

    # TODO 1: normalize query_vec
    # Hint: divide by query_vec.norm() + eps

    # TODO 2: normalize doc_vectors along dim=-1
    # Hint: keepdim=True will make broadcasting easier

    # TODO 3: compute scores [N]
    return None


def embedding_retrieve(query_vec, doc_vectors, docs, top_k=3):
    """
    Args:
        query_vec: Tensor [D]
        doc_vectors: Tensor [N, D]
        docs: list of document strings
        top_k: number of documents to return

    Returns:
        Ranked results with doc_id, text, score.
    """

    # TODO 4: compute scores

    # TODO 5: top-k indices
    # Hint: torch.topk

    # TODO 6: return result dicts
    return None


def test():
    query_vec = torch.tensor([1.0, 0.0, 0.0])
    doc_vectors = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.2, 0.9],
            [0.6, 0.5, 0.0],
        ]
    )
    docs = [
        "LoRA adapts model weights with low rank matrices.",
        "BM25 retrieves documents using keyword matching.",
        "Diffusion models learn to remove noise.",
        "Attention uses query key value projections.",
    ]

    scores = cosine_similarity(query_vec, doc_vectors)
    ref_scores = F.cosine_similarity(doc_vectors, query_vec.unsqueeze(0), dim=1)
    results = embedding_retrieve(query_vec, doc_vectors, docs, top_k=2)

    assert scores.shape == (4,)
    assert torch.allclose(scores, ref_scores, atol=1e-6)
    assert len(results) == 2
    assert results[0]["doc_id"] == 0
    assert results[0]["score"] >= results[1]["score"]

    print("Query vector shape:", query_vec.shape)
    print("Doc vectors shape:", doc_vectors.shape)
    print("Top doc:", results[0]["text"])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. scores shape 是 [N]
2. cosine similarity 对齐 PyTorch 官方实现
3. top-1 文档符合预期
4. 返回结果按 score 降序排列
```

---

## 8. 运行方式

```bash
python code/week5/day31/embedding_retrieval.py
```

---

## 9. 预期输出

```text
Query vector shape: torch.Size([3])
Doc vectors shape: torch.Size([4, 3])
Top doc: LoRA adapts model weights with low rank matrices.
Test passed.
```

---

## 10. 常见错误

```text
1. 忘记 normalize，变成 dot product
2. query_vec 维度没有 unsqueeze
3. top_k 超过文档数量
4. score 排序方向写反
```

---

## 11. 扩展任务

```text
1. 支持 batch query vectors [B,D]
2. 支持 zero vector 的稳定处理
3. 和 BM25 结果做 hybrid score
```

---

## 12. 今日理解问题

```text
1. cosine similarity 为什么要除以 norm？
2. embedding retrieval 和 BM25 最大区别是什么？
3. 为什么真实系统要先离线保存 doc embeddings？
4. 如果 query_vec 是全 0 会发生什么？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入 shape：
- 输出：
- 最容易错的地方：
- 明天要复习的问题：
```
