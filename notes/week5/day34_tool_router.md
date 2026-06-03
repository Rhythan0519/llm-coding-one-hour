# Day 34：Tool Router

## 1. 今日目标

今天实现：

```python
route_tool(user_query, tools)
```

目标是根据用户问题选择最合适的工具。

---

## 2. 这个模块在大模型里有什么用

Agent 不是每次都直接回答，它需要判断是否调用工具：

```text
1. 算术问题可以路由到 calculator
2. 文档问题可以路由到 search_docs
3. 代码概念可以路由到 python_help
4. 不需要工具时返回 none
```

今天先写规则版 router，理解 tool calling 的输入输出。

---

## 3. 输入输出

输入：

```text
user_query: str
tools: list[dict]
每个 tool 包含 name、description、keywords
```

输出：

```text
route: dict
包含 tool_name、score、reason
```

---

## 4. 核心逻辑

```text
1. tokenize user_query
2. 对每个 tool 统计 keyword 命中
3. 选择 score 最高的 tool
4. 如果最高分为 0，返回 none
```

---

## 5. 伪代码

```text
1. 准备 query_terms
2. 遍历 tools
3. 计算 keyword overlap score
4. 记录最佳工具
5. 返回 route dict
```

---

## 6. 代码骨架

```python
"""
Day 34: Tool Router

Goal:
Route a user query to the best tool using simple keyword matching.

Input:
user_query: str
tools: list[dict]

Output:
route: dict with tool_name, score, reason.

Check:
Calculator, search_docs, python_help, and none routes.
"""


DEFAULT_TOOLS = [
    {
        "name": "calculator",
        "description": "Use for arithmetic and numeric computation.",
        "keywords": ["calculate", "compute", "math", "plus", "sum", "multiply", "divide"],
    },
    {
        "name": "search_docs",
        "description": "Use for searching notes, documents, and RAG sources.",
        "keywords": ["search", "docs", "document", "notes", "retrieve", "source", "rag"],
    },
    {
        "name": "python_help",
        "description": "Use for Python syntax and debugging questions.",
        "keywords": ["python", "code", "debug", "function", "list", "dict", "torch"],
    },
]


def route_tool(user_query, tools):
    """
    Args:
        user_query: Raw user query.
        tools: Tool specs with name, description, keywords.

    Returns:
        Route dict with tool_name, score, reason.
    """

    # TODO 1: tokenize user_query
    # Hint: lowercase and split is enough for today

    # TODO 2: score every tool by keyword overlap
    # Hint: compare query tokens with each tool["keywords"]

    # TODO 3: return best tool if score > 0

    # TODO 4: otherwise return none
    return None


def test():
    math_route = route_tool("please calculate 12 plus 30", DEFAULT_TOOLS)
    docs_route = route_tool("search my notes for RAG sources", DEFAULT_TOOLS)
    python_route = route_tool("debug this python list function", DEFAULT_TOOLS)
    none_route = route_tool("good morning", DEFAULT_TOOLS)

    assert math_route["tool_name"] == "calculator"
    assert docs_route["tool_name"] == "search_docs"
    assert python_route["tool_name"] == "python_help"
    assert none_route["tool_name"] == "none"
    assert {"tool_name", "score", "reason"} <= set(math_route.keys())
    assert math_route["score"] > 0

    print("Query: search my notes for RAG sources")
    print("Selected tool:", docs_route["tool_name"])
    print("Reason:", docs_route["reason"])
    print("Test passed.")


if __name__ == "__main__":
    test()
```

---

## 7. 测试要求

```text
1. 数学 query 路由到 calculator
2. 文档 query 路由到 search_docs
3. Python query 路由到 python_help
4. 无匹配 query 返回 none
5. 输出包含 tool_name、score、reason
```

---

## 8. 运行方式

```bash
python code/week5/day34/tool_router.py
```

---

## 9. 预期输出

```text
Query: please search my notes about lora
Selected tool: search_docs
Reason: matched keywords ...
Test passed.
```

---

## 10. 常见错误

```text
1. 只返回字符串，缺少 reason
2. 无匹配时硬选第一个工具
3. keyword 大小写不统一
4. tools 为空时没有处理
```

---

## 11. 扩展任务

```text
1. 支持多个工具并列返回
2. 加入工具参数抽取
3. 把 router 输出转成 OpenAI tool call 风格 JSON
```

---

## 12. 今日理解问题

```text
1. tool router 和普通分类器有什么相似点？
2. 为什么 route 结果里要包含 reason？
3. 错误调用工具会带来什么问题？
4. 什么时候应该返回 none？
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
