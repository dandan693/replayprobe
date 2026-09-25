"""决策签名：把模型的自由文本压成可比较的规范化指纹。

**这是本项目的技术支点。** 没有它，两条轨迹的逐步比对就只能靠人眼看，
或者交给另一个模型去判断「这两步是不是一个意思」——
而后者会让整个项目自相矛盾：**用不可复现的判据，去测可复现性。**

所以这里全程是纯字符串处理，零模型调用，行为确定。

## payload 约定（全项目共用，改这里要同步改 recorder / player）

    LLM_CALL   {"content": str, "tool_call": {"name": str, "arguments": dict} | None,
                "finish": "tool_use" | "stop", "model": str, "usage": {...}}
    TOOL_CALL  {"name": str, "arguments": dict, "ok": bool, "value": Any, "error": str | None}
    FINAL      {"text": str}
    ERROR      {"type": str, "message": str}

## 三级签名：同一份 payload，三种分辨率

这是本项目能做出**分级**（而不是「分叉了 / 没分叉」二元）的原因：

| 级别 | 比什么 | 相同意味着 |
|---|---|---|
| `full` | 整个 payload，含措辞 | 连字都一样 → D0 |
| `decision` | 行动 + **工具名** + 规范化参数 | 决策一样，只是说法不同 → D1 |
| `intent` | 只比动作形状（做题 / 作答 / 拒答 / 崩） | 换的是手段，不是方向 |

三级是**严格的粗细阶梯**：`intent` ⊂ `decision` ⊂ `full`（越往下越粗）。
`intent` 级刻意连工具名都不比 —— 它要回答的是「这次只是换了手段，
还是真的改了方向」，把工具名算进去就答不了这个问题（见 `_llm_decision`）。

**「措辞不同」和「决策不同」必须能分开**，否则温度不为零时满屏都是假分叉 ——
门禁天天报红，两周内团队就会开始加 `--skip`，然后它就没用了。

## 规范化强度是一条刻度，不是一个开关

- **太强** → 把真分叉抹平。`SELECT SUM(amount)` 和 `SELECT SUM(qty)` 如果都被
  抹成「一个 sum 查询」，那就漏报了。
- **太弱** → 稳定存在的空格差异被记成路径分叉，同样是噪音。

**本项目的取舍：只做「表面规范化」（空白 / 大小写 / 标点），不做语义规范化。**
理由是可复现性：表面规范化是确定性且**可解释**的（能说出改了哪几个字符），
而语义规范化必须引入 SQL 解析器（重）或模型判断（不可复现）。

宁可漏报，也不引入不可复现的判据 —— 门禁的可信度是一次性的，
**人们不会信任一个「有时候会误报」的门禁。**
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .types import Step, StepKind

# --------------------------------------------------------------------------- #
# 基础：规范化与哈希
# --------------------------------------------------------------------------- #


def canonical_json(obj: Any) -> str:
    """规范化 JSON 序列化。同一份语义 → 同一个字符串。

    三条规矩：键排序、分隔符紧凑、不转义非 ASCII。
    规范化规则本身必须**稳定且版本化** —— 否则两个语义等价的字典会因为
    无关的格式差异算出不同 hash，那 hash 就失去了证据价值。
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_hash(obj: Any) -> str:
    """内容指纹。取 sha256 前 16 位 —— 够用，且报告里不会长到看不清。"""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


_WS_RE = re.compile(r"\s+")
_TRAILING_SEMI_RE = re.compile(r";\s*$")


def normalize_text(s: str) -> str:
    """把一段文本压到「只剩内容」的形式。

    只做三件事：合并连续空白、去掉首尾空白、去掉末尾分号。
    **刻意不做**：大小写统一（`SELECT` 和 `select` 在措辞层面确实是差异，
    但在决策层面又确实不是 —— 这种归类交给 decision 级签名去处理，不在这里一刀切）。
    """
    if not s:
        return ""
    return _TRAILING_SEMI_RE.sub("", _WS_RE.sub(" ", s).strip())


_SQL_KEYWORD_RE = re.compile(
    r"\b(select|from|where|group\s+by|order\s+by|having|limit|join|inner|left|right|"
    r"on|as|and|or|not|in|is|null|count|sum|avg|min|max|round|distinct|case|when|"
    r"then|else|end|desc|asc|with|partition|over|between|like|union|all)\b",
    re.IGNORECASE,
)


def normalize_sql(sql: str) -> str:
    """SQL 的额外规范化：关键字统一小写 + 空白压缩。

    **只到这一步。** 不做括号补全、不做别名消除、不做等价的表达式重写 ——
    那些都需要一个真正的 SQL 解析器，而且一旦做了，判定就不再可解释。
    """
    s = normalize_text(sql)
    return _SQL_KEYWORD_RE.sub(lambda m: m.group(0).lower(), s)


def normalize_arg(value: Any, key: str = "") -> Any:
    """递归规范化一个参数值。

    字符串走 `normalize_text`；键名里含 `sql` 的额外走 `normalize_sql`；
    列表保序（顺序在工具参数里通常有语义，比如「取前 N 行」）；
    字典按键排序后递归。
    """
    if isinstance(value, str):
        if "sql" in key.lower():
            return normalize_sql(value)
        return normalize_text(value)
    if isinstance(value, dict):
        return {k: normalize_arg(v, k) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_arg(v, key) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        # 去掉浮点的尾随零差异：1.50 与 1.5 是同一个数
        return round(value, 6)
    return value


# --------------------------------------------------------------------------- #
# 数字与拒答：用于 FINAL 步骤的「结论指纹」
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_REFUSAL_MARKERS = (
    "无法获取", "无法查询", "查不到", "查不到该", "没有找到", "未找到",
    "数据中不存在", "不存在该", "无法确定", "不能确定", "无法给出",
    "没有相关数据", "缺少数据", "无法回答", "not available", "no such",
    "cannot determine", "unable to",
)

_YEAR_LO, _YEAR_HI = 1900, 2100


def _is_significant(raw: str, val: float, text: str, end: int, min_int_digits: int) -> bool:
    """判断一个数字是不是「结论性」的。

    **这条规则的第一版是错的，改过一次，值得记下来。**

    第一版只按绝对值大小过滤（>= 100 才算数）。跑出来立刻暴露：
    `74.66%` 和 `81.97%` —— 零售数据集中最关键的两个占比 —— **被静默丢掉了**。
    原因是百分比永远落在 0~100 之间，按数量级筛必然被误伤，
    而它恰恰是最典型的结论数字。

    所以判据从「大不大」换成了「是不是结果值」，三条规则：

    1. **紧跟 `%` → 显著。** 百分比就是结论，不管多小。
    2. **带小数点的非整数 → 显著。** 比值 / 占比 / 均值天然带小数；
       而「前 3 名」「5 个国家」这类结构性计数几乎总是整数。
       这一条把两类数字干净地分开了。
    3. **整数按数量级筛。** 默认 >= 100 才算，用来滤掉结构性计数。

    已知局限（不修，写在这里免得被追问时说不清）：**年份会被误收** ——
    `2026` 是整数且 >= 100，会进结论集合。这个误报在实践中很少触发
    （两条轨迹同时提到年份、且年份不同），但它是真实存在的。
    """
    if text[end:end + 2].startswith("%") or text[end:end + 3].startswith(" %"):
        return True
    if "." in raw:
        return True
    return abs(val) >= 10 ** (min_int_digits - 1)


def extract_numbers(text: str, min_int_digits: int = 3) -> list[float]:
    """从文本里抽出**结论性**数字，归一化后排序。

    `min_int_digits=3` 只作用于**整数**：默认整数要 >= 100 才算数。
    百分比和带小数的数不受这条限制 —— 见 `_is_significant`。

    **为什么必须做这个筛选**：答案里天然混着大量结构性小数字 ——
    「前 3 名」「共 5 个国家」「第 2 步」。它们不承载结论，
    却会随措辞变化（「前三名」vs「前 3 名」），成为 D3 的误报源。

    门槛可配置，因为它是取舍而不是真理：
    调高 → 只有大数参与结论判定，稳但可能漏掉「订单数 42 vs 43」；
    调低 → 更敏感，但误报变多。
    """
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0)
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if _is_significant(raw, val, text or "", m.end(), min_int_digits):
            out.append(round(val, 6))
    return sorted(set(out))


def detect_refusal(text: str) -> bool:
    """这段文本是不是在说「我拿不到数据」。

    这是 D4 安全分叉的判定依据之一：一条轨迹拒答、另一条给了具体数字，
    意味着**防护策略本身变了个样**，比单纯算错严重。
    """
    low = (text or "").lower()
    return any(m in low or m in (text or "") for m in _REFUSAL_MARKERS)


def _decimals(x: float) -> int:
    """这个数在最短十进制表示里有几位小数。

    `repr(float)` 给的是能往返还原的**最短**表示，所以 `repr(8887208.89)` 是
    `'8887208.89'`（2 位）而 `repr(18532.0)` 是 `'18532.0'`（视为 0 位）。

    科学计数法直接放弃（返回 -1）：把 `1e6` 展开成 `1000000` 只会让
    "精度"这个概念变得没有意义。
    """
    s = repr(float(x))
    if "e" in s or "E" in s:
        return -1
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1]
    return 0 if set(frac) <= {"0"} else len(frac)


def same_number(a: float, b: float) -> bool:
    """两个数字**是不是同一个结论** —— 精度不同不算不同。

    这一条是接真实模型之后才补的，因为真实模型立刻把它撞了出来：

        真实模型答：全量总销售额是 8887208.894 美元。
        脚本替身答：全量总销售额为 8,887,208.89 美元，共 18,532 笔订单。

    两者对同一次 `SUM(Amount)` 的表述只差最后一位小数（原始值 vs 保留两位），
    但第一版的集合相等判断把它判成了**"结论被替换"→ D3**。

    **D3 是"该拦下来"的那一档，它的误报代价最高。**
    一个模型只要把小数位数从 3 位改成 2 位，门禁就会红 ——
    这种红法持续两周，团队就会开始加 `--skip`，然后门禁就废了。

    判据刻意选成**确定性的、可解释的**：短的数是长的数四舍五入到它自己
    的小数位数。不引入容差、不引入相对误差阈值 ——
    「相对误差小于 1e-6」在 8887208 这个量级上等于允许 ±8.89 的偏差，
    那会把 8887200 和 8887208 也判成同一个数，太松了。
    """
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        dp = _decimals(y)
        if dp >= 0 and round(x, dp) == round(y, dp):
            return True
    return False


def _match_all(src: list[float], dst: list[float]) -> int:
    """贪心配对：`src` 里有几个能在 `dst` 里找到（同一精度的）对应。

    重复值也要正确计数 —— 所以用「用过就划掉」而不是集合运算。
    """
    taken = [False] * len(dst)
    matched = 0
    for x in src:
        for i, y in enumerate(dst):
            if not taken[i] and same_number(x, y):
                taken[i] = True
                matched += 1
                break
    return matched


def number_relation(a: list[float], b: list[float]) -> str:
    """两个数字集合的关系。返回值：

        same / b_superset / a_superset / differs / both_empty

    **为什么要专门做这个判断。**

    签名层回答的是「是不是同一串数字」—— 这是个确定性问题。
    但「数字串不同」**不等于**「结论被推翻了」，差别的两种情况严重程度差很远：

        {8887208.89}  vs  {8887208.89, 18532}
            原来的数还在，只是多答了一个 → 信息增减，不是推翻

        {8887208.89}  vs  {4443606.45}
            原来的数没了，换成了另一个   → 结论被改变

    用集合运算就能把这两者确定性地分开，**不需要理解语义、不需要模型**。
    分级时前者记 D2（要人看一眼），后者记 D3（坏了）。

    配对用 `same_number`（精度不同算同一个数），所以：

        {8887208.894}  vs  {8887208.89}              → same（只是位数不同）
        {8887208.894}  vs  {8887208.89, 18532}       → b_superset（多答了一个）
        {8887208.894}  vs  {4443606.45}              → differs（真换了）

    这条规则是这个项目里少数几个「用结构运算替代语义判断」的漂亮地方 ——
    因为它经受得住追问：为什么多答一个数不算推翻结论？因为原结论的
    每一个数字都还在，它没有被任何新证据否定。

    **一侧全空、另一侧有数 → `differs`。** 这一点刻意保持保守：
    基线的结论里一个显著数字都没有，重放却给出了数字，
    这大概率是"结论换了"而不是"信息量增加了"。
    （两侧都空才是 `both_empty`，那一档交给文本相似度。）
    """
    la, lb = list(a or ()), list(b or ())
    if not la and not lb:
        return "both_empty"
    if not la or not lb:
        return "differs"
    a_in_b = _match_all(la, lb)
    b_in_a = _match_all(lb, la)
    if a_in_b == len(la) and b_in_a == len(lb):
        return "same"
    if a_in_b == len(la) and len(lb) > len(la):
        return "b_superset"
    if b_in_a == len(lb) and len(la) > len(lb):
        return "a_superset"
    return "differs"


# --------------------------------------------------------------------------- #
# 三级签名
# --------------------------------------------------------------------------- #


def _llm_decision(payload: dict, include_args: bool) -> tuple:
    """模型这一步的决策核心。

    `include_args=True`（decision 级）→ `("act", 工具名, 规范化参数)`
    `include_args=False`（intent 级）→ `("act",)` —— **连工具名都不比**。

    为什么要分这两档：`decision` 回答"选的是不是同一个工具"，
    `intent` 回答"**是不是还停在同一个动作形状上**"。
    后者用于一种具体的追问：这次改动只是换了手段（`run_sql` → `get_schema`），
    还是真的换了方向（从"调工具"变成"直接作答"）？

    第一版这里写的是 `("act", 工具名)` —— 于是换工具会让 intent 级也变，
    它就不再回答上面那个问题了。**文档承诺的能力，实现里必须真的存在。**
    """
    tc = payload.get("tool_call")
    if tc:
        if include_args:
            return ("act", tc.get("name") or "",
                    normalize_arg(tc.get("arguments") or {}))
        return ("act",)
    if payload.get("finish") == "tool_use":
        # 声明要调工具但没给出来 —— 这本身是一种可识别的不良形态
        return ("act_broken",)
    return ("answer",)


def _tool_decision(payload: dict, include_args: bool) -> tuple:
    """工具这一步的决策核心。intent 级只保留"成功/失败"这个形状，丢掉工具名。"""
    if payload.get("ok") is False:
        # 失败的工具调用：比「哪个工具失败了」而不比参数 ——
        # 因为失败时参数往往已被中间件重写过，那属于基础设施噪音
        if include_args:
            return ("tool_failed", payload.get("name") or "")
        return ("tool_failed",)
    if include_args:
        return ("tool", payload.get("name") or "",
                normalize_arg(payload.get("arguments") or {}))
    return ("tool",)


def _final_decision(payload: dict, include_numbers: bool,
                    min_int_digits: int = 3) -> tuple:
    text = payload.get("text") or ""
    refusal = detect_refusal(text)
    if not include_numbers:
        return ("final_refusal" if refusal else "final_answer",)
    return (
        "final_refusal" if refusal else "final_answer",
        tuple(extract_numbers(text, min_int_digits)),
    )


def full_signature(step: Step) -> str:
    """完整签名：整个 payload 的 hash。**含措辞**，用于判定 D0。"""
    return content_hash({"kind": step.kind.value, "payload": step.payload})


def decision_signature(step: Step, min_int_digits: int = 3) -> str:
    """决策签名：行动 + 目标 + 规范化参数。**不含措辞**，用于定位分叉点。

    这是对齐算法实际使用的那个签名。
    """
    if step.kind is StepKind.LLM_CALL:
        core = _llm_decision(step.payload, include_args=True)
    elif step.kind is StepKind.TOOL_CALL:
        core = _tool_decision(step.payload, include_args=True)
    elif step.kind is StepKind.FINAL:
        core = _final_decision(step.payload, include_numbers=True,
                               min_int_digits=min_int_digits)
    else:
        core = ("error", normalize_text(str(step.payload.get("type", ""))))
    return content_hash(list(core))


def intent_signature(step: Step) -> str:
    """意图签名：只比「做还是说」。**最粗的分辨率**，用于判断是否只是换了手段。"""
    if step.kind is StepKind.LLM_CALL:
        core = _llm_decision(step.payload, include_args=False)
    elif step.kind is StepKind.TOOL_CALL:
        core = _tool_decision(step.payload, include_args=False)
    elif step.kind is StepKind.FINAL:
        core = _final_decision(step.payload, include_numbers=False)
    else:
        core = ("error",)
    return content_hash(list(core))


def summarize(step: Step, max_len: int = 46) -> str:
    """把一步压成人类可读的一行，用于报告里的分叉展示。

    报告是给人看的，所以这里**展示原始信息，不做规范化** ——
    规范化是给比较用的，把 `SELECT ROUND(SUM(Amount), 2)` 显示成小写
    会让人怀疑"我的录的东西是不是被改了"。签名负责判断，摘要负责沟通，
    两件事不能共用一个变换。
    """
    if step is None:
        return "(缺失)"
    p = step.payload or {}
    if step.kind is StepKind.LLM_CALL:
        tc = p.get("tool_call")
        if tc:
            args = canonical_json(tc.get("arguments") or {})
            return _clip(f"调用 {tc.get('name')} {args}", max_len)
        return _clip(f"回答：{_squash(p.get('content') or '')}", max_len)
    if step.kind is StepKind.TOOL_CALL:
        args = canonical_json(p.get("arguments") or {})
        flag = "" if p.get("ok", True) else " [失败]"
        return _clip(f"{p.get('name')} {args}{flag}", max_len)
    if step.kind is StepKind.FINAL:
        return _clip(f"答复：{_squash(p.get('text') or '')}", max_len)
    return _clip(f"错误：{p.get('type')} {p.get('message')}", max_len)


def _squash(s: str) -> str:
    """只压空白，不动大小写 —— 摘要要保留原貌。"""
    return _WS_RE.sub(" ", s).strip()


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def self_check() -> list[str]:
    """签名层的自检。这个模块错了，后面所有结论都是错的，所以单测之外还要自查一遍。"""
    problems: list[str] = []

    s1 = Step(seq=1, kind=StepKind.LLM_CALL,
              payload={"content": "我来查一下", "tool_call": {"name": "run_sql",
                       "arguments": {"sql": "SELECT SUM(Amount) FROM retail;"}},
                       "finish": "tool_use"})
    s2 = Step(seq=2, kind=StepKind.LLM_CALL,
              payload={"content": "好的，马上去查。",   # 措辞不同
                       "tool_call": {"name": "run_sql",
                       "arguments": {"sql": "  select sum(Amount) from retail  "}},  # 空白/大小写/分号 不同
                       "finish": "tool_use"})
    s3 = Step(seq=3, kind=StepKind.LLM_CALL,
              payload={"content": "我查订单数吧", "tool_call": {"name": "run_sql",
                       "arguments": {"sql": "SELECT COUNT(*) FROM retail"}},
                       "finish": "tool_use"})

    if full_signature(s1) == full_signature(s2):
        problems.append("s1/s2 的完整签名不该相同 —— 措辞确实变了，D0 就失去了分辨力")
    if decision_signature(s1) != decision_signature(s2):
        problems.append("s1/s2 的决策签名应当相同 —— 空白/大小写/分号差异不得算作路径分叉")
    if decision_signature(s1) == decision_signature(s3):
        problems.append("s1/s3 的决策签名不该相同 —— 换了查询目标就是真分叉，这是漏报")
    if intent_signature(s1) != intent_signature(s3):
        problems.append("s1/s3 的意图签名应当相同 —— 两者都是「调工具」，意图层级没变")

    # 「换了手段」与「换了方向」必须能被区分开 —— 这是 intent 级存在的唯一理由。
    # 第一版实现把工具名也算进 intent，于是换工具就判成不同，
    # 这一档就答不了「是不是只是换了手段」。文档承诺的能力必须在实现里存在。
    s_schema = Step(seq=4, kind=StepKind.LLM_CALL,
                    payload={"content": "先看下表结构", "finish": "tool_use",
                             "tool_call": {"name": "get_schema",
                                           "arguments": {"table": "retail"}}})
    if intent_signature(s1) != intent_signature(s_schema):
        problems.append("换了工具（手段）却让意图签名变了 —— intent 级答不了「是不是只换了手段」")
    if decision_signature(s1) == decision_signature(s_schema):
        problems.append("换工具必须让决策签名不同 —— 否则手段替换会被漏报")
    s_answer = Step(seq=5, kind=StepKind.LLM_CALL,
                    payload={"content": "总销售额是 8887208.89。", "tool_call": None,
                             "finish": "stop"})
    if intent_signature(s1) == intent_signature(s_answer):
        problems.append("「调工具」与「直接作答」的意图签名不该相同 —— 方向变了")

    f1 = Step(seq=9, kind=StepKind.FINAL, payload={"text": "总销售额为 8,887,208.89 美元，涉及 18532 笔订单。"})
    f2 = Step(seq=9, kind=StepKind.FINAL, payload={"text": "全部销售额是 8887208.89（共 18532 单）。"})
    f3 = Step(seq=9, kind=StepKind.FINAL, payload={"text": "总销售额为 4,443,604.45 美元，涉及 18532 笔订单。"})
    if decision_signature(f1) != decision_signature(f2):
        problems.append("f1/f2 结论应当等价 —— 千分位与句式差异不得算作结论分叉")
    if decision_signature(f1) == decision_signature(f3):
        problems.append("f1/f3 结论不该等价 —— 金额差了一倍，这是本项目最不能漏的那类")

    if not detect_refusal("数据中不存在该字段，无法给出结果。"):
        problems.append("拒答检测漏了明显的拒答")
    if detect_refusal("总销售额为 8887208.89 美元。"):
        problems.append("正常答案被误判为拒答")

    # 回归：百分比曾经因为「只按数量级筛」被静默丢掉。
    # 74.66% 和 81.97% 是零售数据集最关键的两个占比，漏掉它们
    # 等于对「占比类结论分叉」完全失明 —— 而且不报错，只是静默少了一些数字。
    pct = extract_numbers("前十名客户贡献 74.66%，其中英国占 81.97%。")
    if 74.66 not in pct or 81.97 not in pct:
        problems.append(f"百分比被丢掉了（实收 {pct}）—— 它们是最典型的结论数字")
    struct = extract_numbers("前三名门店、共 5 个国家的数据。")
    if struct:
        problems.append(f"结构性小整数不该进入结论集合，实收 {struct}")
    big = extract_numbers("总销售额 8,887,208.89 美元，共 18532 笔订单。")
    if 8887208.89 not in big or 18532 not in big:
        problems.append(f"大数被漏掉了（实收 {big}）")

    return problems
