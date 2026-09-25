"""分叉分级：把「两条轨迹不一样」拆成五档。

## 核心立场：分叉不是一个布尔值

如果你只回答「分叉了 / 没分叉」，会立刻遇到两难：

- 把阈值调松 → 温度不为零时措辞本来就会变，于是**什么都算分叉**，
  门禁天天红，两周内团队就学会加 `--skip`，然后它彻底没用。
- 把阈值调紧 → 真的换了个工具、真的算错了，也可能被抹平。

**这不是调参问题，是模型缺了维度。** 所以这里把「不一样」拆成五档，
让噪音（D1）和信号（D3/D4）落在不同的桶里：

| 档 | 含义 | 该不该拦 CI |
|---|---|---|
| D0 | 连措辞都一样 | 不该 |
| D1 | 措辞变了，决策没变 | **不该** —— 这是常态，不是问题 |
| D2 | 路径变了，结论没变 | 看预算。多绕一步可以接受，换了工具要人看 |
| D3 | 结论变了 | 该拦 |
| D4 | 结局性质变了（拒答↔给答案、正常↔崩溃） | **必须拦** |

## 一条纪律：判据不掺启发式

定级全程只用**确定性判据**：签名相等、集合运算、文本相似度。
没有任何一处「让模型判断这两句话是不是一个意思」——
因为本项目的存在意义就是「可复现」，用一个不可复现的判据去判它，自相矛盾。

唯一的可调参数是 `text_sim_threshold`（无显著数字时的文本相似度阈值）。
它被显式暴露成参数、写进配置、并在报告里回显 —— **可调的东西必须看得见**。
"""

from __future__ import annotations

import difflib

from .align import align, mismatch_positions
from .signature import (
    decision_signature,
    detect_refusal,
    extract_numbers,
    normalize_text,
    number_relation,
    summarize,
)
from .types import (
    Divergence,
    ReplayMode,
    Severity,
    Step,
    StepKind,
    Trace,
    Verdict,
    severity_rank,
)

DEFAULT_TEXT_SIM_THRESHOLD = 0.85
"""无显著数字时的文本相似度阈值。高于它算「同一结论的两种说法」。

0.85 是个取舍，不是真理：调高→更容易判成结论分叉（保守，误报多）；
调低→更容易放过（激进，漏报多）。保守还是激进取决于你在护什么 ——
护「不能出错」就调高，护「别烦我」就调低。**没有普适正确值。**
"""


def _final_payload(steps: list[Step]) -> dict | None:
    for s in reversed(steps):
        if s.kind is StepKind.FINAL:
            return s.payload or {}
    return None


def _transcript(steps: list[Step]) -> list[Step]:
    """去掉 ERROR 的步骤序列，用于对齐 —— 崩溃不适合参与逐步比对。"""
    return [s for s in steps if s.kind is not StepKind.ERROR]


def _crashed(steps: list[Step]) -> bool:
    """崩溃判定读**完整**步列表，不读 transcript。

    **这里踩过一个真实的坑，值得写下来。**

    早先 `compare_traces` 把 `transcript_steps()`（已剔除 ERROR）喂进 `grade_steps`，
    而 `_crashed` 就读这几个步 —— 于是它永远返回 False。
    后果是：**走正常链路（CLI / 报告）时崩溃检测完全失效，而且不报错。**

    它不报错，只是安静。D4 那一档只剩"拒答↔给答案"一条触发路径，
    而"正常结束 ↔ 中途崩溃"这条更常见的路径被静默吞掉。
    这比报错危险得多：`diverge.self_check` 里那条崩溃用例是**直接**调
    `grade_steps` 的，所以自检一直是绿的，谁也没发现主链路是瞎的。

    教训：**过滤和判定不能共用一份数据。** 对齐要干净的数据，
    崩溃判定要完整的数据，那就各取各的，别让调用方替被调用方决定。
    """
    return any(s.kind is StepKind.ERROR for s in steps)


def _insertion_kind(
    alignment, a_steps: list[Step], min_int_digits: int
) -> str:
    """多出来的那些步，是「重试」还是「新探索」？

    这个区分是必须的，否则 D2 会泛滥：

    - **重试**：多调了一次同一个工具、同样的参数。决策集合没变，
      只是多试了一轮。温度不为零时这个很高频，把它记成路径分叉，
      等于给门禁加了个天天响的闹钟。
    - **新探索**：调了一个基线里从没出现过的工具。模型**获取了不同的信息**，
      这才是真的路径变化，值得人看一眼。

    判据是确定性的：看插入步的决策签名是否已经在基线的签名集合里出现过。
    """
    a_sigs = {decision_signature(s, min_int_digits=min_int_digits) for s in a_steps}
    for pair in alignment.pairs:
        if pair.is_insertion():
            if decision_signature(pair.b, min_int_digits=min_int_digits) not in a_sigs:
                return "explore"
    return "retry"


def conclusion_relation(
    a_steps: list[Step],
    b_steps: list[Step],
    min_int_digits: int = 3,
    text_sim_threshold: float = DEFAULT_TEXT_SIM_THRESHOLD,
) -> tuple[str, str]:
    """比较两条轨迹的**结论**，返回 `(关系, 人话解释)`。

    关系取值：
      - `safety`      一侧拒答、另一侧给了具体答复 —— 防护策略变了
      - `changed`     结论被改变（数字被替换 / 文本差异大 / 一侧给不出结论）
      - `info_change` 数字是包含关系 —— 原结论未被否定，只是信息量变了
      - `same`        结论一致
    """
    fa, fb = _final_payload(a_steps), _final_payload(b_steps)
    if fa is None or fb is None:
        side = "基线" if fa is None else "重放"
        return "changed", f"{side}一侧没有给出最终答复（未收敛）"

    ta, tb = fa.get("text") or "", fb.get("text") or ""
    ra, rb = detect_refusal(ta), detect_refusal(tb)
    if ra != rb:
        return "safety", (
            "一侧拒答、另一侧给出了具体答复 —— 这不是算错，"
            "是**防护策略本身发生了变化**，性质比数值错更严重"
        )

    na = extract_numbers(ta, min_int_digits)
    nb = extract_numbers(tb, min_int_digits)
    rel = number_relation(na, nb)

    if rel == "same":
        return "same", f"结论数字集合一致：{na}"
    if rel in ("b_superset", "a_superset"):
        return "info_change", (
            f"数字是包含关系（{na} → {nb}）：原结论的每个数字都还在，未被否定，"
            "只是信息量变了"
        )
    if rel == "differs":
        return "changed", f"数字集合被替换：{na} → {nb}"

    # both_empty：两边都没有显著数字，退回文本相似度。
    # 这是全项目唯一的「模糊」判据，所以阈值必须显式暴露、必须在报告里回显。
    ratio = difflib.SequenceMatcher(
        None, normalize_text(ta), normalize_text(tb)
    ).ratio()
    if ratio >= text_sim_threshold:
        return "same", f"无显著数字，文本相似度 {ratio:.3f} ≥ {text_sim_threshold}"
    return "changed", (
        f"无显著数字，文本相似度 {ratio:.3f} < {text_sim_threshold} —— 判为结论不同"
    )


def grade_steps(
    a_steps: list[Step],
    b_steps: list[Step],
    *,
    task_id: str = "",
    baseline_variant: str = "baseline",
    replay_variant: str = "replay",
    min_int_digits: int = 3,
    text_sim_threshold: float = DEFAULT_TEXT_SIM_THRESHOLD,
) -> Verdict:
    """给一次「基线 vs 重放」的比较定级。这是分级的唯一入口。

    入参是**完整的**步列表（含 ERROR）。对齐会自动剔除 ERROR 步，
    而崩溃判定需要它们 —— 所以由本函数内部各取所需，
    不要让调用方提前过滤（那正是之前崩溃检测静默失效的原因）。
    """
    # 崩溃判定要完整数据，对齐要干净数据 —— 各取各的，别共用一份。
    crash_a, crash_b = _crashed(a_steps), _crashed(b_steps)
    a_steps, b_steps = _transcript(a_steps), _transcript(b_steps)

    v = Verdict(task_id=task_id, baseline_variant=baseline_variant,
                replay_variant=replay_variant)

    al = align(a_steps, b_steps, level="decision", min_int_digits=min_int_digits)
    v.alignment = al
    mm = mismatch_positions(al, level="decision", min_int_digits=min_int_digits)

    al_full = align(a_steps, b_steps, level="full", min_int_digits=min_int_digits)
    mm_full = mismatch_positions(al_full, level="full")

    # 逐条收集分叉点，供报告展开
    for idx in mm:
        p = al.pairs[idx]
        v.divergences.append(
            Divergence(
                index=idx,
                a_seq=p.a.seq if p.a else None,
                b_seq=p.b.seq if p.b else None,
                a_summary=summarize(p.a) if p.a else "(缺失)",
                b_summary=summarize(p.b) if p.b else "(缺失)",
                kind=(p.a or p.b).kind.value,
                reason="决策签名不同",
            )
        )

    rel, rel_reason = conclusion_relation(
        a_steps, b_steps, min_int_digits=min_int_digits,
        text_sim_threshold=text_sim_threshold,
    )

    # ── 定级：从严到宽，第一个命中的就是结果 ──────────────────────────
    if rel == "safety":
        v.severity = Severity.D4_SAFETY
        v.reasons.append(rel_reason)
    elif crash_a != crash_b:
        v.severity = Severity.D4_SAFETY
        who = "基线" if crash_a else "重放"
        v.reasons.append(
            f"终止状态分歧：{who}一侧崩溃 / 未收敛，另一侧正常结束 —— "
            "结局性质不同，不是「算法差一点」"
        )
    elif rel == "changed":
        v.severity = Severity.D3_CONCLUSION
        v.reasons.append(rel_reason)
    elif mm:
        v.severity = Severity.D2_PATH
        if rel == "info_change":
            # 主因是信息量变了，路径差异是次因 —— 报告要先把主因说清楚
            v.reasons.append(rel_reason)
            v.reasons.append(
                f"另有 {len(mm)} 处决策签名差异（第 {mm[0] + 1} 处对齐位置起）"
            )
        else:
            v.reasons.append(
                f"第 {mm[0] + 1} 处对齐位置起决策路径不同（共 {len(mm)} 处），"
                "但最终结论一致"
            )
    elif al.shape_changed():
        # 结构变了：多出 / 少了步。必须区分「重试」和「新探索」，
        # 否则改一次提示词后模型偶然多试一次 SQL，就会淹没在 D2 里。
        kind = _insertion_kind(al, a_steps, min_int_digits)
        if kind == "explore":
            v.severity = Severity.D2_PATH
            v.reasons.append(
                f"路径上多出 {al.insertions} 步，且其中包含基线里没有过的**新决策** —— "
                "模型获取了不同的信息，需要人看一眼"
            )
        else:
            v.severity = Severity.D1_WORDING
            v.reasons.append(
                f"多出 {al.insertions} 步但全是重复调用（重试），决策集合没有变化 —— 属常态"
            )
    elif mm_full or al_full.shape_changed():
        v.severity = Severity.D1_WORDING
        v.reasons.append(
            "决策路径完全一致，仅措辞 / 结构有差异 —— 属于常态，不算问题"
        )
    else:
        v.severity = Severity.D0_IDENTICAL
        v.reasons.append("逐步完全一致")

    # 第一条真实分叉（供报告高亮）
    if v.divergences:
        v.first_divergence = v.divergences[0]
    elif v.severity in (Severity.D1_WORDING,) and mm_full:
        idx = mm_full[0]
        p = al_full.pairs[idx]
        v.first_divergence = Divergence(
            index=idx,
            a_seq=p.a.seq if p.a else None,
            b_seq=p.b.seq if p.b else None,
            a_summary=summarize(p.a) if p.a else "(缺失)",
            b_summary=summarize(p.b) if p.b else "(缺失)",
            kind=(p.a or p.b).kind.value,
            reason="仅措辞不同（决策签名一致）",
        )

    v.evidence = {
        "matched": al.matched,
        "insertions": al.insertions,
        "deletions": al.deletions,
        "decision_mismatches": len(mm),
        "full_mismatches": len(mm_full),
        "conclusion_relation": rel,
        "text_sim_threshold": text_sim_threshold,
        "min_int_digits": min_int_digits,
        "crashed": {"baseline": crash_a, "replay": crash_b},
    }
    return v


PROVENANCE_KEYS = ("model", "prompt_hash", "tool_schema_hash", "dataset_snapshot", "mode")
"""用来回答「这次比较到底改了什么」的那几个可追溯字段。

`mode` 也算一个变量 —— 这一点是实测逼出来的。
`drill`（前 N 步用带、之后真实执行）与 `live` 是**两个不同的实验**，
但第一版没把它算进 `changed`，于是 drill 的比较被归进了
「什么都没改」那一组，把对照组的「0% 分叉」污染成了 14.3%。

**「怎么跑的」和「改了什么配置」是在同一个问题下的两个方面**：
我拿两个东西来比，它们之间的全部差异，就是这次实验的自变量。
漏掉一个自变量，对照组就不纯了。
"""


def trace_provenance(trace: Trace) -> dict:
    """从轨迹的 manifest 里抽出**可用来回答「这是什么条件下跑的」**的几个字段。"""
    m = trace.manifest or {}
    return {
        "model": m.get("model", "") or trace.variant,
        "prompt_hash": m.get("prompt_hash", ""),
        "tool_schema_hash": m.get("tool_schema_hash", ""),
        "dataset_snapshot": m.get("dataset_snapshot", ""),
        "limitations": len(m.get("limitations") or []),
        "mode": trace.mode.value,
    }


def changed_variables(a: dict, b: dict) -> list[str]:
    """两条轨迹之间，**确实被改动**的那些变量。

    空哈希按「未知」处理，不计入改动 —— 把"没记录"当成"变了"会制造假警报，
    而假警报多了，真警报就没人看了。
    """
    out = []
    for k in PROVENANCE_KEYS:
        x, y = a.get(k) or "", b.get(k) or ""
        if not x or not y:
            continue
        if x != y:
            out.append(k)
    return out


def compare_traces(
    baseline: Trace,
    replay: Trace,
    *,
    min_int_digits: int = 3,
    text_sim_threshold: float = DEFAULT_TEXT_SIM_THRESHOLD,
) -> Verdict:
    """比较两条完整轨迹。

    **传完整步列表，不要传 transcript。** 对齐层会自己剔除 ERROR 步，
    而崩溃判定需要读到它们。见 `_crashed` 里记的那个坑。

    除了分级，这里还会把**溯源信息**写进 `evidence` —— 于是"这次比较到底改了什么"
    和"分叉程度"被放在同一个文件里。少了前者，一份分了 8 组的报告
    没法回答最基本的问题：**你改的那个变量，到底是什么？**
    """
    v = grade_steps(
        baseline.steps,
        replay.steps,
        task_id=baseline.task_id or replay.task_id,
        baseline_variant=baseline.variant,
        replay_variant=replay.variant,
        min_int_digits=min_int_digits,
        text_sim_threshold=text_sim_threshold,
    )
    pa, pb = trace_provenance(baseline), trace_provenance(replay)
    changed = changed_variables(pa, pb)
    v.evidence["provenance"] = {"baseline": pa, "replay": pb, "changed": changed}

    if _could_be_unexplained(baseline, replay, changed, v.severity):
        # 最值得被看见的一种局面：**什么都没改，行为却变了。**
        # 这说明分叉不是我们引入的，而是上游（供应商路由、缓存、
        # 模型别名指向了别的权重）带来的 —— 这是"确定性"这个前提本身出了裂缝。
        v.evidence["unexplained_fork"] = True
        v.reasons.append(
            "**两条轨迹的可追溯身份完全一致（模型 / prompt / 工具 / 数据集都没有变化），"
            "却出现了决策分叉** —— 这不是你改出来的，是上游的非确定性。"
        )
    return v


def _could_be_unexplained(baseline: Trace, replay: Trace, changed: list[str],
                          severity: Severity) -> bool:
    """这次分叉**可能是**「上游非确定性」造成的吗？

    这个告警第一版只看了「变量没改」，结果在 `drill` 比较上**误报了** ——
    而 drill 的分叉是工具**自己故意切出来的**（前 N 步用带，之后走真实执行），
    拿它去报"上游不稳定"，等于把实验设计当成了故障。

    误报的代价在这个项目里被反复强调：**警报一多，真警报就没人看了。**
    所以判据收紧成三条硬条件：

    1. **两侧都是 `live`（真实执行）。** 只要有一侧是带驱动的重放
       （exact / strict / drill），分叉就有一个已知的、由我们决定的原因，
       轮不到"上游非确定性"来解释。
    2. 变量确实一个都没改（`changed` 为空）。
    3. 确实到了 D2 以上。

    换句话说：**只有「同一配置、独立跑两遍」这个对照实验，才有资格报这条。**
    这也正是它唯一有信息量的场景。
    """
    if changed:
        return False
    if baseline.mode is not ReplayMode.LIVE or replay.mode is not ReplayMode.LIVE:
        return False
    return severity_rank(severity) >= severity_rank(Severity.D2_PATH)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def self_check() -> list[str]:
    """分级层的自检。每条都对应一个真实会遇到的场景。"""
    from .align import _final, _llm_call, _tool

    problems: list[str] = []

    base = [
        _llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _tool("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _final("总销售额为 8887208.89 美元，共 18532 笔订单。"),
    ]

    # ① 完全一致 → D0
    v = grade_steps(base, list(base))
    if v.severity is not Severity.D0_IDENTICAL:
        problems.append(f"完全一致判成了 {v.severity.value}")

    # ② 仅措辞不同 → D1（不得更重）
    wording = [
        base[0],
        base[1],
        _final("全部销售额是 8,887,208.89 美元，共 18532 笔订单。"),
    ]
    v = grade_steps(base, wording)
    if v.severity is not Severity.D1_WORDING:
        problems.append(f"仅措辞变化应判 D1，实得 {v.severity.value}：{v.reasons}")

    # ③ 多答了一个数字（原数都在）→ D2，不是 D3
    more = [
        base[0],
        base[1],
        _final("总销售额为 8887208.89 美元，共 18532 笔订单，覆盖 4338 位客户。"),
    ]
    v = grade_steps(base, more)
    if v.severity is not Severity.D2_PATH:
        problems.append(
            f"信息增加应判 D2（原结论未被否定），实得 {v.severity.value}：{v.reasons}"
        )

    # ④ 数字被替换 → D3。这是最不能漏的一档。
    wrong = [base[0], base[1], _final("总销售额为 4443606.45 美元，共 18532 笔订单。")]
    v = grade_steps(base, wrong)
    if v.severity is not Severity.D3_CONCLUSION:
        problems.append(f"金额被换成另一个数却判成 {v.severity.value} —— 最严重的漏报")

    # ⑤ 换了查询目标但结论碰巧一样 → D2
    other = [
        _llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail WHERE Country='United Kingdom'"}),
        _tool("run_sql", {"sql": "SELECT SUM(Amount) FROM retail WHERE Country='United Kingdom'"}),
        base[2],
    ]
    v = grade_steps(base, other)
    if v.severity is not Severity.D2_PATH:
        problems.append(f"换了查询目标但结论一致应判 D2，实得 {v.severity.value}")

    # ⑥ 拒答 ↔ 给答案 → D4
    refuse = [base[0], base[1], _final("数据中不存在该字段，无法给出结果。")]
    v = grade_steps(base, refuse)
    if v.severity is not Severity.D4_SAFETY:
        problems.append(f"拒答状态变化应判 D4，实得 {v.severity.value}")

    # ⑦ 崩溃 ↔ 正常 → D4
    from .types import Step as _S

    crash = [base[0], _S(seq=9, kind=StepKind.ERROR,
                         payload={"type": "RuntimeError", "message": "boom"})]
    v = grade_steps(base, crash)
    if v.severity is not Severity.D4_SAFETY:
        problems.append(f"终止状态分歧应判 D4，实得 {v.severity.value}")

    # ⑧ 百分比变化必须能被抓到（曾经因数量级门槛被静默丢掉）
    pa = [base[0], base[1], _final("前十名客户贡献 74.66%。")]
    pb = [base[0], base[1], _final("前十名客户贡献 74.65%。")]
    v = grade_steps(pa, pb)
    if v.severity is not Severity.D3_CONCLUSION:
        problems.append(f"占比变了 0.01 个百分点却没抓到，实得 {v.severity.value}")

    # ⑨ 单纯重试（多调一次同样的工具）→ D1，不得升级为 D2。
    # 改提示词后模型偶然多试一次 SQL 是高频事件，把它记成路径分叉，
    # D2 就会被噪音淹没 —— 那是这个量表最容易死掉的方式。
    from .align import _llm_call

    retry = [
        base[0], base[1],
        _llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _tool("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        base[2],
    ]
    v = grade_steps(base, retry)
    if v.severity is not Severity.D1_WORDING:
        problems.append(
            f"单纯重试应判 D1，实得 {v.severity.value} —— D2 会被高频噪音淹没"
        )

    # ⑩ 引入了基线里从没出现过的决策 → D2（真的换了获取信息的路径）
    explore = [
        base[0], base[1],
        _llm_call("get_schema", {"table": "retail"}),
        _tool("get_schema", {"table": "retail"}),
        base[2],
    ]
    v = grade_steps(base, explore)
    if v.severity is not Severity.D2_PATH:
        problems.append(f"引入新决策应判 D2，实得 {v.severity.value}")

    # ⑪ 崩溃判定必须能从**主链路**走到。
    #
    # 这一条是补的，因为踩过一个"安静"的 bug：`compare_traces` 早先提前把
    # ERROR 步过滤掉了，而 `_crashed` 就读那些步 —— 于是走 CLI / 报告时
    # 崩溃检测**永远返回 False，且不报错**。
    # 上面第 ⑦ 条是**直接**调 `grade_steps` 的，所以自检一直是绿的，
    # 没有人发现主链路是瞎的。
    #
    # **教训：过滤和判定不能共用一份数据；而且自检必须覆盖真实调用路径，
    # 只测底层函数的自检会给你一种虚假的安全感。**
    from .types import ReplayMode, Step, Trace

    t_base = Trace(run_id="a", task_id="t", variant="baseline",
                   mode=ReplayMode.LIVE, steps=list(base))
    t_crash = Trace(run_id="b", task_id="t", variant="replay", mode=ReplayMode.EXACT,
                    steps=[base[0], Step(seq=9, kind=StepKind.ERROR,
                                         payload={"type": "RuntimeError", "message": "boom"})])
    v = compare_traces(t_base, t_crash)
    if v.severity is not Severity.D4_SAFETY:
        problems.append(
            f"经 compare_traces 的崩溃判定失效（实得 {v.severity.value}）—— "
            "主链路会静默漏报 D4"
        )

    return problems
