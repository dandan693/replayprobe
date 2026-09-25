"""被测 Agent 与它的工具。

放在包里而不是 tests 里，是因为**它本身也是项目的一部分**：
README 里所有关于「分叉」的演示跑的都是这一个 Agent。
"""

from .core import DEFAULT_MAX_STEPS, SYSTEM_PROMPT, ReActAgent
from .tools_builtin import (
    PARAM_DOCS,
    SCHEMA_HINT,
    make_tools,
    openai_tool_schemas,
    tool_schema_hash,
)

__all__ = [
    "ReActAgent",
    "SYSTEM_PROMPT",
    "DEFAULT_MAX_STEPS",
    "make_tools",
    "tool_schema_hash",
    "openai_tool_schemas",
    "PARAM_DOCS",
    "SCHEMA_HINT",
]
