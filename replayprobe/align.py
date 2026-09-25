"""序列对齐：把两条轨迹的步一一对上，而不是按序号硬对。

## 为什么不能用序号对齐

因为改一次提示词，最常见的后果**不是**「某一步做错了」，
而是「**某一步多做了一次**」—— 多查一次表结构、多试一次 SQL、多一轮重试。
一旦中间多出一步，后面所有步号就整体错位。

按序号对齐会把「同一个决策，只是位置挪了一位」全部算成分叉。

**这个错误方向特别恶劣：它看起来像真的发现了问题。**
报告里长长一串分叉，没人会怀疑是工具算错了 ——
他们只会得出「这次改动影响很大」，然后动手去改那些根本没坏的地方。

所以这里改用**基于决策签名的序列对齐**（`difflib.SequenceMatcher`，标准库）：
先算签名，再对齐，未匹配的步如实标成插入 / 删除。

## 保留朴素实现做对照

`align_by_index` 是刻意留着的 —— 它不是「没删掉的旧代码」，
而是**对照实验的另一组**。README 里那个「误报 N 处 vs 真实 M 处」的数字
就是由它和 `align` 跑出来对比的。删掉它，那个数字就没有出处了。
"""

from __future__ import annotations

import difflib

from .signature import decision_signature, full_signature, intent_signature
from .types import Alignment, AlignedPair, Step

LEVELS = ("full", "decision", "intent")
"""三级分辨率，与 signature.py 的三级签名一一对应。

- `full`：含措辞。用它对齐最严格，但温度不为零时会大量错位 ——
  它的用途是**证明 D0**，不是定位分叉。
- `decision`：默认。比行动 + **工具名** + 规范化参数。
- `intent`：最粗，**连工具名都不比**，只比动作形状。用来回答
  「是不是只是换了手段（换了工具），而方向没变」。
"""


def _sig_of(step: Step, level: str, min_int_digits: int) -> str:
    if level == "full":
        return full_signature(step)
    if level == "intent":
        return intent_signature(step)
    return decision_signature(step, min_int_digits=min_int_digits)


def _align_impl(
    a_steps: list[Step],
    b_steps: list[Step],
    sig_a: list[str],
    sig_b: list[str],
    method: str,
) -> Alignment:
    """按给定签名序列对齐。对齐算法本身与签名的选取解耦。"""
    matcher = difflib.SequenceMatcher(None, sig_a, sig_b, autojunk=False)
    pairs: list[AlignedPair] = []
    matched = insertions = deletions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append(AlignedPair(a=a_steps[i1 + k], b=b_steps[j1 + k]))
                matched += 1
        elif tag == "replace":
            # 替换区段内逐一对齐；多出来的一侧如实算插入 / 删除。
            # **不要把替换简单记成「两边都变了」** —— 那会丢掉
            # 「A 有 3 步、B 有 4 步」里的结构信息。
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                pairs.append(AlignedPair(a=a_steps[i1 + k], b=b_steps[j1 + k]))
            for k in range(n, i2 - i1):
                pairs.append(AlignedPair(a=a_steps[i1 + k], b=None))
                deletions += 1
            for k in range(n, j2 - j1):
                pairs.append(AlignedPair(a=None, b=b_steps[j1 + k]))
                insertions += 1
        elif tag == "delete":
            for k in range(i1, i2):
                pairs.append(AlignedPair(a=a_steps[k], b=None))
                deletions += 1
        elif tag == "insert":
            for k in range(j1, j2):
                pairs.append(AlignedPair(a=None, b=b_steps[k]))
                insertions += 1

    return Alignment(pairs=pairs, matched=matched, insertions=insertions,
                     deletions=deletions, method=method)


def align(
    a_steps: list[Step],
    b_steps: list[Step],
    level: str = "decision",
    min_int_digits: int = 3,
) -> Alignment:
    """基于签名的序列对齐。这是默认对齐方式。"""
    if level not in LEVELS:
        raise ValueError(f"未知的签名级别：{level}，可选 {LEVELS}")
    sig_a = [_sig_of(s, level, min_int_digits) for s in a_steps]
    sig_b = [_sig_of(s, level, min_int_digits) for s in b_steps]
    return _align_impl(a_steps, b_steps, sig_a, sig_b,
                       method=f"signature-lcs[{level}]")


def align_by_index(
    a_steps: list[Step],
    b_steps: list[Step],
    level: str = "decision",
    min_int_digits: int = 3,
) -> Alignment:
    """对照实现：按序号一一对应。**这不是给用户用的**，是用来量化
    「序号对齐错得有多离谱」的基准线。见模块开头。
    """
    if level not in LEVELS:
        raise ValueError(f"未知的签名级别：{level}，可选 {LEVELS}")
    pairs: list[AlignedPair] = []
    matched = 0
    n = min(len(a_steps), len(b_steps))
    for i in range(n):
        pairs.append(AlignedPair(a=a_steps[i], b=b_steps[i]))
        if _sig_of(a_steps[i], level, min_int_digits) == _sig_of(b_steps[i], level, min_int_digits):
            matched += 1
    for i in range(n, len(a_steps)):
        pairs.append(AlignedPair(a=a_steps[i], b=None))
    for i in range(n, len(b_steps)):
        pairs.append(AlignedPair(a=None, b=b_steps[i]))

    return Alignment(
        pairs=pairs, matched=matched,
        insertions=max(0, len(b_steps) - n), deletions=max(0, len(a_steps) - n),
        method=f"seq-index[{level}]",
    )


def mismatch_positions(
    alignment: Alignment, level: str = "decision", min_int_digits: int = 3
) -> list[int]:
    """对齐结果里，所有「配对成功但签名不同」的位置。

    **只统计双向都有步的配对。** 插入 / 删除本身不算「决策分叉」——
    它是结构变化，由 `Alignment.shape_changed()` 单独报告。
    把两者混进一个数字，会让「多绕了一步」和「选择变了」看起来一样严重。
    """
    out: list[int] = []
    for idx, pair in enumerate(alignment.pairs):
        if not pair.is_pair():
            continue
        if _sig_of(pair.a, level, min_int_digits) != _sig_of(pair.b, level, min_int_digits):
            out.append(idx)
    return out


def divergence_count(
    a_steps: list[Step], b_steps: list[Step], level: str = "decision",
    min_int_digits: int = 3, method: str = "signature",
) -> int:
    """便捷函数：直接数两条轨迹在给定对齐方式下的决策分叉处数。

    README 里对照实验的两次调用就是这个函数（method 换一下）。
    """
    fn = align_by_index if method == "index" else align
    al = fn(a_steps, b_steps, level=level, min_int_digits=min_int_digits)
    return len(mismatch_positions(al, level=level, min_int_digits=min_int_digits))


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def _llm_call(name: str, args: dict, text: str = "") -> Step:
    from .types import StepKind

    return Step(
        seq=0, kind=StepKind.LLM_CALL,
        payload={"content": text or f"调用 {name}", "finish": "tool_use",
                 "tool_call": {"name": name, "arguments": args}},
        source="tape",
    )


def _tool(name: str, args: dict) -> Step:
    from .types import StepKind

    return Step(seq=0, kind=StepKind.TOOL_CALL,
                payload={"name": name, "arguments": args, "ok": True, "value": []},
                source="tape")


def _final(text: str) -> Step:
    from .types import StepKind

    return Step(seq=0, kind=StepKind.FINAL, payload={"text": text}, source="tape")


def self_check() -> list[str]:
    """自检的核心任务只有一个：**证明签名对齐确实比序号对齐强。**

    构造一个改变提示词后最常见的场景 —— 中间多做了一步，其余决策完全没变。
    好的对齐器应当报告「0 处决策分叉 + 1 处结构插入」；
    序号对齐会误报一串分叉。这两组数字如果跑不出差距，说明对齐是坏的。
     """
    problems: list[str] = []

    a = [
        _llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _tool("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _llm_call("finish", {}),
        _final("总销售额为 8887208.89 美元。"),
    ]
    # B 只是中间多查了一次表结构，其余三步决策完全一致
    b = [
        _llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _tool("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _llm_call("get_schema", {"table": "retail"}),
        _tool("get_schema", {"table": "retail"}),
        _llm_call("finish", {}),
        _final("总销售额为 8887208.89 美元。"),
    ]

    al = align(a, b)
    if mismatch_positions(al) != []:
        problems.append(
            f"签名对齐误报了决策分叉：{mismatch_positions(al)} —— "
            "B 只是多做了一步，没有任何决策发生变化"
        )
    if al.insertions != 2:
        problems.append(f"签名对齐的插入步数应为 2，实得 {al.insertions}")
    if al.matched != 4:
        problems.append(f"签名对齐应匹配 4 步，实得 {al.matched}")

    naive = align_by_index(a, b)
    naive_mm = mismatch_positions(naive)
    if not naive_mm:
        problems.append(
            "序号对齐居然没误报 —— 那本项目就失去了动机。请检查对照实现是不是被改坏了"
        )

    # 真分叉必须仍能被抓到：把 B 的查询目标换掉
    c = [s for s in a]
    c[0] = _llm_call("run_sql", {"sql": "SELECT COUNT(*) FROM retail"})
    if not mismatch_positions(align(a, c)):
        problems.append("换了查询目标却没报出分叉 —— 那是漏报，比误报更危险")

    # 措辞变化不得算作决策分叉。
    # 注意这里是**纯粹的**措辞变化 —— 数字一个不多一个不少。
    # 第一版用例手滑多带了一个数字（18532），结果被判出分叉，一度以为是 bug。
    # 其实那是对的：签名层只负责回答「是不是同一串数字」，
    # 「多答了一个数字算不算改变结论」是语义问题，由 diverge.py 用集合关系去分层。
    # **职责不能混** —— 混了就没法解释为什么判成这样。
    d = [s for s in a]
    d[3] = _final("全部销售额是 8,887,208.89 美元。")
    if mismatch_positions(align(a, d)):
        problems.append("结论措辞变化被算成了分叉 —— D1 档被击穿，噪音会淹没信号")

    return problems
