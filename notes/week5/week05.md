# Week 5：RAG / Agent / Tool Calling 小工程（2026-06-15 ~ 2026-06-21）

## 1. 本周目标

本周主要学习大模型应用工程里的最小 RAG 与工具调用基础：文本切 chunk、BM25 检索、embedding 检索、rerank、prompt builder、tool router，以及一个 mini RAG demo。

本周结束后，我应该能够：

- 把长文本切成带 overlap 的 chunks；
- 写出一个最小 BM25 检索器；
- 用 cosine similarity 做 embedding retrieval；
- 对候选文档做 rerank；
- 把 question 和 context 拼成清晰的 RAG prompt；
- 根据用户问题选择工具；
- 从 markdown 文件夹里检索答案并返回 sources。

---

## 2. 本周核心能力

```text
1. 理解 chunk_size / overlap / top_k
2. 熟悉 tokenization、term frequency、document frequency
3. 熟悉 cosine similarity 和向量排序
4. 能把检索结果转成 prompt context
5. 能输出 answer + sources，而不是只输出 answer
6. 能写最小 tool router
7. 能用 assert 检查检索顺序和来源
```

---

## 3. 每日任务安排

| Day | 文件名 | 任务 | 输入 | 输出 | 验收 |
|---|---|---|---|---|---|
| Day 29 | [day29_text_chunking.md](day29_text_chunking.md) | text chunking | long text, chunk_size, overlap | chunks list | chunk 数和 overlap 正确 |
| Day 30 | [day30_bm25_retrieval.md](day30_bm25_retrieval.md) | BM25 retrieval | query, docs | top-k docs | 关键词相关文档排前面 |
| Day 31 | [day31_embedding_retrieval.md](day31_embedding_retrieval.md) | embedding retrieval | query vector, doc vectors | top-k docs | cosine similarity 排序正确 |
| Day 32 | [day32_rerank.md](day32_rerank.md) | rerank | query, candidates | reranked docs | 更相关候选排前面 |
| Day 33 | [day33_prompt_builder.md](day33_prompt_builder.md) | prompt builder | question, contexts | prompt string | 包含问题、上下文、来源 |
| Day 34 | [day34_tool_router.md](day34_tool_router.md) | tool router | user query | tool name | 常见 query 分类正确 |
| Day 35 | [day35_mini_rag.md](day35_mini_rag.md) | mini RAG | md folder, question | answer + sources | 有检索依据和来源 |

---

## 4. 本周 mini demo

实现一个从 markdown 文件夹读取资料的 mini RAG：

输入：

```text
一个 markdown 文件夹
一个用户问题
```

输出：

```text
answer: 基于检索内容生成的简短答案
sources: 命中的文件名和 chunk_id
prompt: 拼接后的 RAG prompt
```

验收：

```text
1. 能读取多个 .md 文件。
2. 能切 chunk 并保留 source。
3. 能检索 top-k chunks。
4. prompt 里包含 context 和 question。
5. answer 不是凭空生成，能对应 sources。
```

---

## 5. 本周验收标准

```text
1. 所有每日脚本都能单独运行。
2. 每个脚本都有至少 2 个 assert。
3. 每个脚本都打印关键输入输出。
4. 我能解释 chunk overlap 的作用。
5. 我能解释 BM25 中 TF / IDF 的含义。
6. 我能解释 cosine similarity 为什么适合向量检索。
7. 我能说清楚 RAG prompt 里为什么要带 sources。
8. mini RAG 能返回 answer + sources。
```

---

## 6. 本周复盘问题

```text
1. chunk_size 太大或太小分别有什么问题？
2. overlap 为什么能缓解上下文被切断的问题？
3. BM25 和 embedding retrieval 各自适合什么场景？
4. rerank 为什么通常放在粗检索之后？
5. prompt builder 里 context 和 question 应该怎么分隔？
6. tool router 分类错了会导致什么后果？
7. mini RAG 的 answer 怎么证明来自 sources？
```

