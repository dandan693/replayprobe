"""分叉预算：把「允许分叉到什么程度」变成 CI 里可断言的策略。

## 这一块是我认为整个项目最有价值的设计

现有的观测工具（LangSmith / Langfuse / Phoenix / Braintrust）都能让你
**看**两条轨迹的差异。但看完之后呢？

差异是「可接受」还是「不可接受」，**只能靠人看**。于是它永远停在调试阶段，
进不了流水线 —— 因为流水线需要的是一个**能返回退出码的判断**。

这个模块补的就是这一环：把「这次改动允许分叉到什么程度」写成一个 JSON，
让 CI 去判。于是 replay 从**调试辅助**变成了**可执行的回归断言**。

## 为什么叫「预算」

因为「允许分叉多少」和「允许花多少钱」是同一种东西：

- 它是**有额度的**：D1 无限，D2 少量，D3/D4 零。
- 它是**可耗尽**的：改一次提示词花掉一部分，改十次就没了。
- 它**必须显式声明**：没人声明过额度的门禁，等于没有门禁。

## 七个可配项，每一个都对应一种真实的团队争论

| 配置 | 它回答的问题 |
|---|---|
| `max_severity` | 多严重的分叉要拦下来？ |
| `divergence_floor` | **从哪一档开始，算作「这个改动碰到了它」？** |
| `allow_fork_after` | 前几步是不是必须稳？（改提示词不该动第 1 步的工具选择） |
| `max_divergence_ratio` | 整批任务里，允许多大比例被碰到？ |
| `min_matched_ratio` | 对齐后至少要有多少步是匹配上的？ |
| `forbid_crash` | 崩溃能不能进主干？ |
| `forbid_safety_fork` | 结局性质（拒答↔给答案）能不能变？ |

**注意额度是「按批」而不是「按条」**：单条任务 D2 很正常，
但一批 50 条里 30 条都是 D2，那就是改动本身有问题。所以比率类配置必须存在。

## `divergence_floor` 是怎么被发现的

它不在第一版设计里，是**实测逼出来的**。`model_swap` 预算的原话是
「路径允许大改，但结论不许变」，上限设成 D3；同时分叉率上限 0.5。
拿真实模型一跑：两组比较**全是 D2**（SQL 写法不同、结论一模一样），
分叉率 100% → 被判违规。

问题不在于阈值定错了，而在于**两个旋钮在互相打架**：
`max_severity` 说「D2 允许」，分叉率却把 D2 算作违规。
于是这份预算永远无法通过，而它恰恰是「换便宜模型值不值得」这个问题
唯一的可执行表达。

修法不是把 0.5 调成 1.0（那叫把门禁调绿灯），而是**把混在一起的两个概念拆开**：

- `max_severity` 管「最坏那一条能坏到什么程度」
- `divergence_floor` 管「从哪一档起算『它被碰到了』」，是**计数口径**，不是严重度口径

换模型时把 floor 设成 `d3_conclusion`，含义就清楚了：
**路径变了不算「被碰到」，结论变了才算。** 两份预算于是各自自洽。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .types import Severity, Verdict, severity_rank


@dataclass
class Budget:
    """一份分叉预算。"""

    name: str = "default"
    max_severity: Severity = Severity.D1_WORDING
    """允许出现的最重分叉。默认只允许到 D1（措辞差异）。

    默认值刻意定得紧：**门禁应该一开始就红，然后由人一条条放宽**。
    反过来（先松后紧）在实践中几乎不会发生 —— 松惯了没人愿意再收紧。
    """

    allow_fork_after: int | None = None
    """只允许在第 N 个对齐位置**之后**出现分叉 —— 之前必须完全稳定。

    这是最有用的一个旋钮。典型用法：改提示词时，`allow_fork_after=2`
    表示「前两步（选工具、取数）不许变，后面随便」。
    它把「这次改动的影响范围」从形容词变成了数字。
    """

    max_divergence_ratio: float = 0.0
    """整批任务里「被碰到」的最大比例。0 表示一条都不许。

    被碰到的定义由 `divergence_floor` 决定 —— 改了 floor 就改了这条规则的分母口径。
    """

    divergence_floor: Severity = Severity.D2_PATH
    """从哪一档起计入分叉率。默认 D2（路径分叉及以上）。

    默认值与第一版行为一致：D0/D1 是常态不算数，D2 起算。
    换模型的场景通常要把它抬到 `d3_conclusion`，理由见模块开头那段。
    """

    min_matched_ratio: float = 0.0
    """对齐后匹配步数 / 最长侧步数 的最低要求。

    它专门用来抓一种情况：**两条轨迹长度差很多**时，LCS 仍然可能给出
    「匹配率很高」的假象（短的被长的完全包含）。这个下限是个兜底。
    """

    forbid_crash: bool = True
    forbid_safety_fork: bool = True

    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "max_severity": self.max_severity.value,
            "divergence_floor": self.divergence_floor.value,
            "allow_fork_after": self.allow_fork_after,
            "max_divergence_ratio": self.max_divergence_ratio,
            "min_matched_ratio": self.min_matched_ratio,
            "forbid_crash": self.forbid_crash,
            "forbid_safety_fork": self.forbid_safety_fork,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(raw: dict) -> "Budget":
        data = dict(raw)
        for key in ("max_severity", "divergence_floor"):
            if key in data and isinstance(data[key], str):
                data[key] = Severity(data[key])
        return Budget(**{k: v for k, v in data.items() if k in Budget.__annotations__})

    def audit(self) -> list[str]:
        """检查这份预算是否自相矛盾。返回问题列表，空表示自洽。

        刻意只做**一个**检查：`divergence_floor` 低于 `max_severity`
        意味着「某一档被明确允许，却又被计入违规率」——
        这正是 `model_swap` 踩过的坑。它会让门禁**永远红着**，
        而永远红着的门禁和没有门禁是同一回事。
        """
        out = []
        if severity_rank(self.divergence_floor) < severity_rank(self.max_severity):
            out.append(
                f"divergence_floor({self.divergence_floor.value}) 低于 "
                f"max_severity({self.max_severity.value})："
                "被明确允许的等级又被算进了分叉率，这份预算不可能稳定通过。"
                "把 floor 抬到与 max_severity 同级或更高。"
            )
        return out


@dataclass
class Violation:
    task_id: str
    rule: str
    detail: str
    severity: str = ""

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "rule": self.rule,
                "detail": self.detail, "severity": self.severity}


@dataclass
class GateResult:
    """一批任务的预算结算结果。"""

    budget: Budget
    total: int = 0
    distribution: dict[str, int] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    divergence_ratio: float = 0.0
    matched_ratio: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        dist = "  ".join(f"{k}={v}" for k, v in sorted(self.distribution.items()))
        head = "通过" if self.passed else f"未通过（{len(self.violations)} 条违例）"
        return (
            f"预算 [{self.budget.name}] {head}\n"
            f"  任务 {self.total} 条 · 分叉率 {self.divergence_ratio:.1%}"
            f"（{self.budget.divergence_floor.value} 及以上计入）· "
            f"匹配率 {self.matched_ratio:.1%}\n"
            f"  分布 {dist}"
        )


def _diverged(v: Verdict, floor: Severity = Severity.D2_PATH) -> bool:
    """这一条算不算「被这次改动碰到」。

    `floor` 之前是写死的 D2 —— 写死的结果是它和 `max_severity` 暗中打架，
    见模块开头记录的那次实测。现在它是一个显式的计数口径。
    """
    return severity_rank(v.severity) >= severity_rank(floor)


def check(verdicts: list[Verdict], budget: Budget) -> GateResult:
    """按预算结算一批比较结果。"""
    res = GateResult(budget=budget, total=len(verdicts))

    # 先审预算本身。**一份自相矛盾的预算不该被默默执行** ——
    # 它会让门禁永远红着，而团队对永远红着的东西的反应不是修它，是无视它。
    for problem in budget.audit():
        res.violations.append(Violation(
            task_id="(预算)", rule="budget_incoherent", detail=problem))

    if not verdicts:
        res.violations.append(Violation(
            task_id="-", rule="empty", detail="没有任何比较结果 —— 门禁不能对空集判通过"
        ))
        return res

    dist: dict[str, int] = {}
    diverged = 0
    matched_sum = 0
    len_sum = 0

    lim = severity_rank(budget.max_severity)

    for v in verdicts:
        dist[v.severity.value] = dist.get(v.severity.value, 0) + 1

        if _diverged(v, budget.divergence_floor):
            diverged += 1

        al = v.alignment
        matched_sum += al.matched
        len_sum += max(1, al.matched + al.insertions + al.deletions)

        # ① 严重度上限
        if severity_rank(v.severity) > lim:
            res.violations.append(Violation(
                task_id=v.task_id, rule="max_severity", severity=v.severity.value,
                detail=f"分叉等级 {v.severity.value} 超过上限 {budget.max_severity.value}"
                       + (f"；{v.reasons[0]}" if v.reasons else ""),
            ))

        # ② 崩溃
        if budget.forbid_crash and v.evidence.get("crashed", {}).get("replay"):
            res.violations.append(Violation(
                task_id=v.task_id, rule="forbid_crash", severity=v.severity.value,
                detail="重放一侧崩溃 / 未收敛",
            ))

        # ③ 安全分叉
        if budget.forbid_safety_fork and v.severity is Severity.D4_SAFETY:
            res.violations.append(Violation(
                task_id=v.task_id, rule="forbid_safety_fork", severity=v.severity.value,
                detail="结局性质发生变化（拒答↔给答案 / 正常↔崩溃）",
            ))

        # ④ 前缀稳定性
        if budget.allow_fork_after is not None and v.first_divergence is not None:
            if v.first_divergence.index < budget.allow_fork_after:
                res.violations.append(Violation(
                    task_id=v.task_id, rule="allow_fork_after",
                    severity=v.severity.value,
                    detail=f"分叉出现在对齐位置 {v.first_divergence.index}，"
                           f"早于允许的 {budget.allow_fork_after} —— 前缀必须稳定",
                ))

    res.distribution = dist
    res.divergence_ratio = diverged / len(verdicts)
    res.matched_ratio = matched_sum / max(1, len_sum)

    # ⑤ 批次级比率
    if res.divergence_ratio > budget.max_divergence_ratio:
        res.violations.append(Violation(
            task_id="(整批)", rule="max_divergence_ratio",
            detail=f"分叉率 {res.divergence_ratio:.1%}（{budget.divergence_floor.value} "
                   f"及以上计入）超过上限 {budget.max_divergence_ratio:.1%} —— "
                   f"单条都可以接受，但被碰到的比例这么高，说明改动的影响面比预期大",
        ))

    # ⑥ 匹配率下限
    if res.matched_ratio < budget.min_matched_ratio:
        res.violations.append(Violation(
            task_id="(整批)", rule="min_matched_ratio",
            detail=f"匹配率 {res.matched_ratio:.1%} 低于下限 {budget.min_matched_ratio:.1%}",
        ))

    return res


def load_budgets(directory: str | Path) -> dict[str, Budget]:
    """从目录加载所有预算 JSON，键是文件名（去掉扩展名）。"""
    out: dict[str, Budget] = {}
    d = Path(directory)
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw.setdefault("name", p.stem)
        out[p.stem] = Budget.from_dict(raw)
    return out


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def self_check() -> list[str]:
    from .align import _final, _llm_call, _tool
    from .diverge import grade_steps

    problems: list[str] = []

    base = [
        _llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _tool("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        _final("总销售额为 8887208.89 美元。"),
    ]
    d0 = grade_steps(base, list(base), task_id="t-d0")
    d1 = grade_steps(base, [base[0], base[1], _final("全部销售额是 8,887,208.89 美元。")],
                     task_id="t-d1")
    d3 = grade_steps(base, [base[0], base[1], _final("总销售额为 4443606.45 美元。")],
                     task_id="t-d3")

    # 默认预算（只允许到 D1）应放过 d0/d1、拦下 d3
    strict = Budget(name="strict", max_severity=Severity.D1_WORDING)
    r = check([d0, d1], strict)
    if not r.passed:
        problems.append(f"默认预算不该拦下 D0/D1：{[v.rule for v in r.violations]}")
    r = check([d1, d3], strict)
    if r.passed:
        problems.append("默认预算没拦下 D3 —— 门禁是坏的")
    if not any(v.rule == "max_severity" for v in r.violations):
        problems.append("D3 违例没有归到 max_severity 规则上")

    # 空集必须判失败 —— 门禁不能对空集盖章
    r = check([], strict)
    if r.passed:
        problems.append("空集被判通过 —— 那 CI 里一个 glob 写错就全绿了")

    # 分叉率是按批算的：单条 D2 可接受，但 2/3 是 D2 就该拦
    d2 = grade_steps(
        base,
        [base[0], base[1], _final("总销售额为 8887208.89 美元，覆盖 4338 位客户。")],
        task_id="t-d2",
    )
    r = check([d0, d1, d2, d2], strict)
    if not any(v.rule == "max_divergence_ratio" for v in r.violations):
        problems.append("分叉率超限没有被抓到 —— 批次级统计是缺的")

    # allow_fork_after：分叉出现在第 2 位之前要拦
    early = grade_steps(
        base,
        [_llm_call("get_schema", {"table": "retail"}), base[1], base[2]],
        task_id="t-early",
    )
    b = Budget(name="prefix", max_severity=Severity.D2_PATH, allow_fork_after=1)
    r = check([early], b)
    if not any(v.rule == "allow_fork_after" for v in r.violations):
        problems.append("前缀稳定性没有被检查 —— 这是最常用的那个旋钮")

    # divergence_floor 必须真的改变计数口径。
    # 这条断言来自一次实测：model_swap 声明「路径允许大改」（上限 D3），
    # 却用「D2 及以上计入」的分叉率把 100% 的 D2 判成违规 ——
    # 两个旋钮互相打架，这份预算永远无法通过。
    swap = Budget(name="swap", max_severity=Severity.D3_CONCLUSION,
                  divergence_floor=Severity.D3_CONCLUSION,
                  max_divergence_ratio=0.5)
    r_swap = check([d2, d2], swap)
    if r_swap.divergence_ratio != 0.0:
        problems.append(
            f"floor=D3 时 D2 不该计入分叉率，实得 {r_swap.divergence_ratio:.1%}"
        )
    if any(v.rule == "max_divergence_ratio" for v in r_swap.violations):
        problems.append("floor=D3 时 D2 触发了分叉率违例 —— 计数口径没生效")

    # 而默认 floor=D2 时，同一批必须被判出来
    r_def = check([d2, d2], Budget(name="d", max_severity=Severity.D1_WORDING,
                                   max_divergence_ratio=0.5))
    if r_def.divergence_ratio != 1.0:
        problems.append(f"floor=D2 时两条 D2 的分叉率应为 100%，实得 {r_def.divergence_ratio:.1%}")

    # 自相矛盾的预算必须被 audit 拦下，而不是被默默执行
    bad = Budget(name="bad", max_severity=Severity.D3_CONCLUSION,
                 divergence_floor=Severity.D2_PATH, max_divergence_ratio=0.5)
    if not bad.audit():
        problems.append("floor 低于 max_severity 的预算没被 audit 抓出来")
    if not any(v.rule == "budget_incoherent"
               for v in check([d0], bad).violations):
        problems.append("自相矛盾的预算没有被记成违例 —— 它会被默默执行")

    return problems


def self_check_budgets_dir() -> str:
    """把当前默认预算渲染成可读文本，方便肉眼核对。"""
    b = Budget(name="default", notes="默认：只允许措辞差异")
    return json.dumps(b.to_dict(), ensure_ascii=False, indent=2)
