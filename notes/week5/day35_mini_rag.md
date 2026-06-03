# Day 35：Mini RAG

## 1. 今日目标

今天实现：

```python
run_mini_rag(folder, question)
```

目标是把本周模块串起来：读取 markdown、切 chunk、检索、构造 prompt、返回 answer + sources。

---

## 2. 这个模块在大模型里有什么用

一个最小 RAG 系统通常包含：

```text
1. document loader
2. chunker
3. retriever
4. prompt builder
5. answer generator
6. source attribution
```

今天不调用真实大模型，先用 extractive answer 证明检索和来源链路正确。

---

## 3. 输入输出

输入：

```text
folder: str，包含 .md 文件的目录
question: str
```

输出：

```text
result: dict
包含 answer、sources、prompt、retrieved_chunks
```

---

## 4. 核心逻辑

```text
1. 读取 folder 下所有 .md 文件
2. 每篇文档切 chunk
3. 用 BM25 / keyword score 检索 top-k chunks
4. 构造 RAG prompt
5. 用 top chunk 生成一个 extractive answer
6. 返回 sources
```

---

## 5. 伪代码

```text
1. docs = load_markdown_docs(folder)
2. chunks = chunk_documents(docs)
3. retrieved = retrieve(question, chunks, top_k=2)
4. prompt = build_rag_prompt(question, retrieved)
5. answer = make_extractive_answer(retrieved)
6. return result dict
```

---

## 6. 代码骨架

```python
"""
Day 35: Mini RAG

Goal:
Wire together loading markdown files, chunking, retrieval, prompt building,
and extractive answering.

Input:
folder: path containing .md files
question: str

Output:
result: dict with answer, sources, prompt, retrieved_chunks.

Check:
Reads docs, retrieves relevant source, builds prompt, returns answer + sources.
"""

from pathlib import Path
from tempfile import TemporaryDirectory


def load_markdown_docs(folder):
    """
    Args:
        folder: Directory containing .md files.

    Returns:
        docs: list[dict] with source and text.
    """

    # TODO 1: read all .md files
    # Hint:
    # - use Path(folder).glob("*.md")
    # - read text with encoding="utf-8"
    # - keep source as file.name
    return None


def chunk_documents(docs, chunk_size=40, overlap=10):
    """
    Args:
        docs: list of dicts with source and text.
        chunk_size: max words per chunk.
        overlap: shared words between neighboring chunks.

    Returns:
        chunks: list[dict] with source, chunk_id, text.
    """

    # TODO 2: split docs into chunks with source and chunk_id
    # Hint:
    # - use the Day 29 sliding-window idea
    # - each chunk should keep source and chunk_id
    return None


def retrieve(question, chunks, top_k=2):
    """
    Args:
        question: User question.
        chunks: Chunk dicts.
        top_k: Number of chunks to return.

    Returns:
        Retrieved chunks with score.
    """

    # TODO 3: score chunks and return top_k
    # Hint:
    # - tokenize question and chunk text
    # - score by keyword overlap
    # - return new dicts with score
    return None


def build_rag_prompt(question, contexts):
    """
    Args:
        question: User question.
        contexts: retrieved chunks.

    Returns:
        Prompt string.
    """

    # TODO 4: reuse Day 33 idea
    # Prompt should include Context:, Question:, and Answer:
    return None


def make_extractive_answer(retrieved_chunks):
    """
    Args:
        retrieved_chunks: ranked chunks.

    Returns:
        Short answer derived from top chunk.
    """

    # TODO 5: create a short answer from top chunk
    # Hint: use retrieved_chunks[0]["text"] and keep it short
    return None


def run_mini_rag(folder, question):
    """
    Args:
        folder: Markdown folder.
        question: User question.

    Returns:
        result dict with answer, sources, prompt, retrieved_chunks.
    """

    # TODO 6: wire the whole pipeline
    # 1. load docs
    # 2. chunk docs
    # 3. retrieve relevant chunks
    # 4. build prompt
    # 5. build answer
    # 6. return result dict
    return None


def test():
    question = "How does LoRA reduce trainable parameters?"

    with TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        (folder / "lora.md").write_text(
            "LoRA reduces trainable parameters by adding low rank adapter matrices "
            "while keeping the base model frozen.",
            encoding="utf-8",
        )
        (folder / "rag.md").write_text(
            "RAG retrieves relevant document chunks and puts them into the prompt.",
            encoding="utf-8",
        )

        result = run_mini_rag(folder, question)

    assert isinstance(result, dict)
    assert {"answer", "sources", "prompt", "retrieved_chunks"} <= set(result.keys())
    assert len(result["sources"]) >= 1
    assert result["retrieved_chunks"][0]["source"] == "lora.md"
    assert "Context:" in result["prompt"] and "Question:" in result["prompt"]
    assert "lora" in result["answer"].lower() or "low rank" in result["answer"].lower()

    print("Question:", question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 能读取多个 markdown 文件
2. 能返回至少一个 source
3. 检索结果和问题相关
4. prompt 包含 Context 和 Question
5. answer 里能看到来自 top chunk 的信息
```

---

## 8. 运行方式

```bash
python code/week5/day35/mini_rag.py
```

---

## 9. 预期输出

```text
Question: How does LoRA reduce trainable parameters?
Answer: ...
Sources: ['lora.md#chunk0']
Test passed.
```

---

## 10. 常见错误

```text
1. 只返回 answer，不返回 source
2. chunk_id 没保存，source 追踪断掉
3. prompt 没包含检索到的 context
4. 检索分数为 0 的 chunk 也排在前面
```

---

## 11. 扩展任务

```text
1. 把 Day 30 BM25 复制进来替代关键词计数
2. 加入 embedding retrieval
3. 支持引用格式 [source#chunk]
4. 接入真实 LLM API
```

---

## 12. 今日理解问题

```text
1. mini RAG 的每一步输入输出是什么？
2. 为什么 answer 必须带 sources？
3. 如果检索错了，prompt builder 能补救吗？
4. extractive answer 和生成式 answer 有什么区别？
5. 这个 demo 和真实 RAG 系统还差哪些能力？
```

---

## 13. 今日总结模板

```markdown
## 今日总结

- 今天实现了：
- 输入：
- 输出：
- 最容易错的地方：
- 下周要复习的问题：
```
