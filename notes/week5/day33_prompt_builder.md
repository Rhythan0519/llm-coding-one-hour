# Day 33：Prompt Builder

## 1. 今日目标

今天实现：

```python
build_rag_prompt(question, contexts)
```

目标是把用户问题和检索到的上下文拼成结构清晰的 RAG prompt。

---

## 2. 这个模块在大模型里有什么用

RAG 的质量不只取决于检索，也取决于怎么把 context 放进 prompt：

```text
1. 明确要求模型只基于 context 回答
2. 给每段 context 标 source
3. 把 question 和 context 清楚分隔
4. 在信息不足时允许回答不知道
```

---

## 3. 输入输出

输入：

```text
question: str
contexts: list[dict]
每个 context 包含 text、source、chunk_id
max_context_chars: int 或 None
```

输出：

```text
prompt: str
```

---

## 4. 核心格式

```text
You are a helpful assistant...

Context:
[source=xxx chunk=0]
...

Question:
...

Answer:
```

---

## 5. 伪代码

```text
1. 创建 system instruction
2. 遍历 contexts
3. 每段 context 加 source 和 chunk_id
4. 如果设置 max_context_chars，就截断 context
5. 拼接 question
6. 返回 prompt
```

---

## 6. 代码骨架

```python
"""
Day 33: Prompt Builder

Goal:
Build a clear RAG prompt from a question and retrieved contexts.

Input:
question: str
contexts: list[dict]

Output:
prompt: str

Check:
Prompt contains context, question, answer section, and sources.
"""


def build_rag_prompt(question, contexts, max_context_chars=None):
    """
    Args:
        question: User question.
        contexts: list of dicts with text, source, chunk_id.
        max_context_chars: Optional total context character budget.

    Returns:
        Prompt string.
    """

    # TODO 1: validate question
    # Hint: empty or whitespace-only question should raise ValueError

    # TODO 2: format each context with source and chunk_id
    # Example:
    # [source=lora.md chunk=0]
    # LoRA adds low rank adapter matrices...

    # TODO 3: apply max_context_chars if provided

    # TODO 4: build final prompt string
    # It should include Context:, Question:, and Answer: sections.
    return None


def test():
    question = "What is LoRA?"
    contexts = [
        {
            "text": "LoRA adds low rank adapter matrices to frozen model weights.",
            "source": "lora.md",
            "chunk_id": 0,
        },
        {
            "text": "PEFT trains only a small number of adapter parameters.",
            "source": "peft.md",
            "chunk_id": 1,
        },
    ]

    prompt = build_rag_prompt(question, contexts)
    limited_prompt = build_rag_prompt(question, contexts, max_context_chars=40)

    assert isinstance(prompt, str)
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert question in prompt
    assert "lora.md" in prompt and "peft.md" in prompt
    assert len(limited_prompt) < len(prompt)

    raised = False
    try:
        build_rag_prompt("", contexts)
    except ValueError:
        raised = True
    assert raised is True

    print("Question:", question)
    print("Prompt length:", len(prompt))
    print("Sources: lora.md, peft.md")
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. prompt 包含 question
2. prompt 包含每个 context 的 source
3. prompt 包含明确的 Context / Question / Answer 区块
4. max_context_chars 生效
```

---

## 8. 运行方式

```bash
python code/week5/day33/prompt_builder.py
```

---

## 9. 预期输出

```text
Question: What is LoRA?
Prompt length: ...
Sources: lora.md, peft.md
Test passed.
```

---

## 10. 常见错误

```text
1. prompt 里没有 source，后续无法引用依据
2. context 和 question 混在一起不清楚
3. 截断时把所有 context 都截没了
4. 空 question 没处理
```

---

## 11. 扩展任务

```text
1. 支持中文 prompt 模板
2. 支持 numbered citations
3. 支持不同 answer style
```

---

## 12. 今日理解问题

```text
1. 为什么 RAG prompt 要要求模型基于 context 回答？
2. source 信息应该放在哪里？
3. context 太长会有什么问题？
4. 信息不足时 prompt 应该怎么约束回答？
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
