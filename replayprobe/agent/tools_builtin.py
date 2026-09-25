"""被测 Agent 用的三个工具。

**工具是确定性的**，这一点很重要：整条轨迹里唯一真正的非确定性来源是模型。
把工具做成确定性的，才能确保「两条轨迹不一样」这件事只可能由模型行为引起 ——
否则你永远分不清是模型变了还是数据库变了。

真值库直接复用 faultprobe / evalguard 的 `retail_truth.db`（UCI Online Retail，
清洗后 392,692 行）。**同一个领域数据被三个项目共用是刻意的**：
`evalguard` 量对不对、`faultprobe` 打坏了会怎样、`replayprobe` 重放会不会走同一条路。
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

SCHEMA_HINT = (
    "表 retail(InvoiceNo, StockCode, Description, Quantity, InvoiceDate, "
    "UnitPrice, CustomerID, Country, Amount, YearMonth)"
)


def make_tools(db_path: str | Path) -> dict[str, Callable[..., Any]]:
    """构造工具注册表。工具名 → 可调用对象，参数即 JSON 里的字段名。

    工具签名刻意保持**平坦**（都是关键字参数），因为参数会被写进
    决策签名参与比较 —— 嵌套结构会让签名对无关的格式差异过于敏感。
    """
    db = Path(db_path)

    def run_sql(sql: str = "") -> Any:
        """执行一条只读 SQL。返回行列表。

        只允许 SELECT —— 重放系统绝不能有任何副作用，
        这条限制不是「礼貌」，是「重放绝不允许触发真实写操作」这条铁律的落地。
        见 docs/项目方案.md 里关于 replay 安全边界的说明。
        """
        if not sql.strip():
            raise ValueError("SQL 为空")
        lowered = sql.strip().lower()
        if not lowered.startswith("select") and not lowered.startswith("with"):
            raise ValueError("run_sql 只接受只读查询（SELECT / WITH）")
        for banned in (" insert ", " update ", " delete ", " drop ", " alter ", " attach "):
            if banned in f" {lowered} ":
                raise ValueError(f"run_sql 拒绝执行含写操作的语句：{banned.strip()}")
        # 注意用的是 closing() 而不是裸的 `with sqlite3.connect(...)`。
        # 后者是常见误解：它只管事务（进入时 begin、退出时 commit），
        # **不会关闭连接** —— 句柄一直留到被 GC，表现为 ResourceWarning。
        # 一个"看起来用了 with 就安全了"的写法，正是本项目最警惕的那类问题。
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]

    def get_schema(table: str = "retail") -> Any:
        """取表的列信息。**调用它会改变轨迹形状** ——
        基线里没查过表结构的轨迹，一旦多查一次，对齐器会把它识别为
        「新探索」而不是「重试」。这正是 replayprobe 要区分的两种变化。
        """
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            if not cols:
                raise ValueError(f"表不存在：{table}")
            return [{"name": c[1], "type": c[2]} for c in cols]

    def calc(expression: str = "") -> Any:
        """精确算术。**不调模型、不用浮点字面量** —— 只做四则与括号。

        它的存在是为了回答一个具体问题：当 Agent 需要「先取数再算一步」时，
        那一步是走工具（可复现）还是让模型心算（不可复现）。
        两条轨迹在这里分叉，是很有信息量的一件事。
        """
        import ast
        import operator as op

        allowed = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                   ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
                   ast.Mod: op.mod}
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"表达式无法解析：{exc}") from exc

        def ev(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                return allowed[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](ev(node.operand))
            raise ValueError(f"不支持的表达式节点：{type(node).__name__}")

        return round(ev(tree), 6)

    return {"run_sql": run_sql, "get_schema": get_schema, "calc": calc}


PARAM_DOCS: dict[tuple[str, str], str] = {
    ("run_sql", "sql"): "一条只读 SQL（必须以 SELECT 或 WITH 开头）。"
                        f"可用表结构：{SCHEMA_HINT}",
    ("get_schema", "table"): "表名，默认 retail。",
    ("calc", "expression"): "只含四则运算与括号的算术表达式，例如 (1-0.9)*100。",
}
"""每个参数的一句话说明，直接写死在这里。

**为什么不去解析 docstring。** 解析看起来聪明，但一旦 docstring 排版变了，
解析失败是**静默**的 —— 参数描述变成空字符串，模型照样能跑，只是 SQL 质量悄悄变差，
而你不会收到任何报错。这种"出错了但没人知道"的失败模式，
恰恰是本项目存在的理由。宁可多写三行，也不要一个会沉默降级的解析器。
"""

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def openai_tool_schemas(tools: dict[str, Callable[..., Any]]) -> list[dict]:
    """把工具注册表转成 OpenAI function-calling 格式的 `tools` 参数。

    只做「如实翻译」：函数名 → name，docstring 首段 → description，
    签名 → parameters。**不做任何美化或补全** ——
    模型看到的就是这套工具真实的样子，这样轨迹才有解释力。
    """
    import inspect

    schemas: list[dict] = []
    for name in sorted(tools):
        fn = tools[name]
        doc = inspect.getdoc(fn) or ""
        # 首段即功能说明；后面那些「为什么这么设计」的长段落不给模型看 ——
        # 它们是给人看的，塞进 prompt 只会占 token 并让行为难以归因。
        description = (doc.split("\n\n") or [""])[0].replace("\n", " ").strip()

        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in inspect.signature(fn).parameters.items():
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
            props[pname] = {
                "type": _TYPE_MAP.get(ann, "string"),
                "description": PARAM_DOCS.get((name, pname), f"参数 {pname}"),
            }
            if p.default is inspect.Parameter.empty:
                required.append(pname)

        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description or f"工具 {name}",
                "parameters": {"type": "object", "properties": props,
                               "required": required},
            },
        })
    return schemas


def tool_schema_hash(tools: dict[str, Callable[..., Any]]) -> str:
    """工具定义指纹，写进录制带的 manifest。

    **工具实现换了，旧轨迹就不再可比** —— 但如果不记录指纹，
    这种不可比是静默的：你会拿一份对不上的标准答案去判分，还奇怪为什么全红。

    指纹覆盖 **工具名 + 参数名 + 默认值**。
    第一版只哈希了工具名 —— 那等于说「改了参数默认值也算同一套工具」，
    而参数默认值恰恰会改变模型的行为（`run_sql(sql="")` 与带默认表名是两回事）。
    只哈希名字的指纹会**在改动发生时保持沉默**，这比没有指纹更糟：
    它让你以为自己验过了。

    已知局限（如实写下来）：签名里的**类型注解**不参与指纹，
    因为注解不影响运行时行为；函数体实现也不参与 —— 那需要源码 hash，
    而源码 hash 会被注释和格式变化干扰，噪声大于信号。
    """
    import inspect

    from ..signature import content_hash

    sigs: list[list] = []
    for name in sorted(tools):
        try:
            params = inspect.signature(tools[name]).parameters
            sigs.append([name, {
                p.name: ("<required>" if p.default is inspect.Parameter.empty
                         else str(p.default))
                for p in params.values()
            }])
        except (TypeError, ValueError):
            # 内建 / C 扩展拿不到签名 —— 退回只记名字，并明确标出来
            sigs.append([name, "<introspection-unavailable>"])
    return content_hash(sigs)
