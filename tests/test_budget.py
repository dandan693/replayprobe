"""预算层的测试：门禁必须能返回退出码，而且不能对空集盖章。

这一层是整个项目从"调试辅助"变成"可执行回归断言"的那一环，
所以测试的重点不是"能不能算"，而是**判错方向**：
漏放一条该拦的，和误拦一条不该拦的，后果完全不同。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from replayprobe.budget import Budget, check, load_budgets, self_check
from replayprobe.diverge import grade_steps
from replayprobe.types import Severity, Step, StepKind, severity_rank


def llm_call(name: str, args: dict) -> Step:
    return Step(seq=0, kind=StepKind.LLM_CALL, payload={
        "content": f"调用 {name}", "finish": "tool_use",
        "tool_call": {"name": name, "arguments": args},
    })


def tool_call(name: str, args: dict) -> Step:
    return Step(seq=0, kind=StepKind.TOOL_CALL,
                payload={"name": name, "arguments": args, "ok": True, "value": []})


def final(text: str) -> Step:
    return Step(seq=0, kind=StepKind.FINAL, payload={"text": text})


SQL_TOTAL = "SELECT SUM(Amount) FROM retail"


def base() -> list[Step]:
    return [
        llm_call("run_sql", {"sql": SQL_TOTAL}),
        tool_call("run_sql", {"sql": SQL_TOTAL}),
        final("总销售额为 8887208.89 美元，共 18532 笔订单。"),
    ]


def verdict_d0(task_id: str = "t-d0"):
    a = base()
    return grade_steps(a, base(), task_id=task_id)


def verdict_d1(task_id: str = "t-d1"):
    b = base()
    b[2] = final("全部销售额是 8,887,208.89 美元，共 18532 笔订单。")
    return grade_steps(base(), b, task_id=task_id)


def verdict_d2(task_id: str = "t-d2"):
    b = base()
    b[2] = final("总销售额为 8887208.89 美元，共 18532 笔订单，覆盖 4338 位客户。")
    return grade_steps(base(), b, task_id=task_id)


def verdict_d3(task_id: str = "t-d3"):
    b = base()
    b[2] = final("总销售额为 4443606.45 美元，共 18532 笔订单。")
    return grade_steps(base(), b, task_id=task_id)


class TestDefaultBudget(unittest.TestCase):
    def test_d0_and_d1_pass(self):
        """D1 是常态，默认必须放过 —— 否则门禁两周内就会被 --skip 掉。"""
        r = check([verdict_d0(), verdict_d1()],
                  Budget(name="default", max_severity=Severity.D1_WORDING))
        self.assertTrue(r.passed, [v.rule for v in r.violations])

    def test_d3_is_blocked(self):
        r = check([verdict_d1(), verdict_d3()], Budget(name="default"))
        self.assertFalse(r.passed, "默认预算没拦下 D3 —— 门禁是坏的")
        self.assertTrue(any(v.rule == "max_severity" for v in r.violations))


class TestEmptySetIsNeverGreen(unittest.TestCase):
    def test_empty_fails(self):
        """**门禁不能对空集盖章。**

        否则 CI 里一个 glob 写错、路径打错字，就全绿了 ——
        而这种"全绿"看起来和"真的没问题"一模一样。
        """
        r = check([], Budget(name="default"))
        self.assertFalse(r.passed)
        self.assertTrue(any(v.rule == "empty" for v in r.violations))


class TestBatchLevelRatios(unittest.TestCase):
    def test_single_d2_ok_but_many_d2_blocked(self):
        """额度是"按批"算的：单条 D2 很正常，一批里一半都是 D2 就是改动有问题。"""
        b = Budget(name="strict", max_severity=Severity.D1_WORDING,
                   max_divergence_ratio=0.0)
        r = check([verdict_d0(), verdict_d1()], b)
        self.assertTrue(r.passed)

        r = check([verdict_d0(), verdict_d1(), verdict_d2(), verdict_d2()], b)
        self.assertTrue(any(v.rule == "max_divergence_ratio" for v in r.violations))

    def test_divergence_ratio_excludes_d1(self):
        """分叉率不数 D1 —— 把 D1 算进去，比率就永远是高的。

        注意这份预算把 `divergence_floor` 也设成了 D3：它是自洽的写法。
        第一版这里只写了 `max_severity=D3`，于是 floor 取了默认值 D2 ——
        也就是「D2 明确允许，却又被计入违规率」。当时这条测试是绿的，
        因为它只关心比率、没关心 `passed`；直到新加的预算审计把它抓出来。
        **一个自相矛盾的预算，会以"测试通过"的形式藏起来。**
        """
        b = Budget(name="x", max_severity=Severity.D3_CONCLUSION,
                   divergence_floor=Severity.D3_CONCLUSION,
                   max_divergence_ratio=0.5)
        r = check([verdict_d0(), verdict_d1(), verdict_d1(), verdict_d1()], b)
        self.assertEqual(r.divergence_ratio, 0.0)
        self.assertTrue(r.passed, [v.rule for v in r.violations])

    def test_divergence_floor_changes_what_counts(self):
        """floor 是计数口径，不是严重度口径 —— 改它，同一批数据的比率就变了。"""
        batch = [verdict_d2(), verdict_d2()]
        loose = Budget(name="loose", max_severity=Severity.D3_CONCLUSION,
                       divergence_floor=Severity.D3_CONCLUSION,
                       max_divergence_ratio=0.5)
        tight = Budget(name="tight", max_severity=Severity.D3_CONCLUSION,
                       divergence_floor=Severity.D2_PATH, max_divergence_ratio=0.5)
        self.assertEqual(check(batch, loose).divergence_ratio, 0.0)
        self.assertEqual(check(batch, tight).divergence_ratio, 1.0)

    def test_incoherent_budget_is_flagged_not_silently_run(self):
        """floor 低于 max_severity 的预算必须被拦下。

        这是实测逼出来的规则：`model_swap` 原本声明「路径允许大改」（上限 D3），
        却用「D2 及以上计入」的分叉率把 100% 的 D2 判成违规 ——
        两个旋钮互相打架，这份预算永远无法通过。
        **永远红着的门禁和没有门禁是同一回事**，所以这种配置不能默默执行。
        """
        bad = Budget(name="bad", max_severity=Severity.D3_CONCLUSION,
                     divergence_floor=Severity.D2_PATH, max_divergence_ratio=0.5)
        self.assertTrue(bad.audit())
        r = check([verdict_d0()], bad)
        self.assertTrue(any(v.rule == "budget_incoherent" for v in r.violations))
        self.assertFalse(r.passed)


class TestPrefixStability(unittest.TestCase):
    def test_fork_before_allowed_position_is_blocked(self):
        """改提示词时最常用的旋钮：前两步不许变。"""
        early = grade_steps(
            base(),
            [llm_call("get_schema", {"table": "retail"}), base()[1], base()[2]],
            task_id="t-early",
        )
        b = Budget(name="prefix", max_severity=Severity.D2_PATH, allow_fork_after=1)
        r = check([early], b)
        self.assertTrue(any(v.rule == "allow_fork_after" for v in r.violations))

    def test_no_fork_means_no_prefix_violation(self):
        b = Budget(name="prefix", max_severity=Severity.D2_PATH, allow_fork_after=5)
        r = check([verdict_d0()], b)
        self.assertFalse(any(v.rule == "allow_fork_after" for v in r.violations))


class TestCrashAndSafetyRules(unittest.TestCase):
    def test_crash_is_blocked_when_forbidden(self):
        crashed = grade_steps(
            base(),
            [base()[0], Step(seq=0, kind=StepKind.ERROR,
                             payload={"type": "RuntimeError", "message": "boom"})],
            task_id="t-crash",
        )
        r = check([crashed], Budget(name="x", max_severity=Severity.D4_SAFETY,
                                    forbid_crash=True))
        self.assertTrue(any(v.rule == "forbid_crash" for v in r.violations))

    def test_safety_fork_is_blocked(self):
        refuse = base()
        refuse[2] = final("数据中不存在该字段，无法给出结果。")
        v = grade_steps(base(), refuse, task_id="t-refuse")
        r = check([v], Budget(name="x", max_severity=Severity.D4_SAFETY,
                              forbid_safety_fork=True))
        self.assertTrue(any(v.rule == "forbid_safety_fork" for v in r.violations))


class TestBudgetSerialization(unittest.TestCase):
    def test_roundtrip(self):
        b = Budget(name="a", max_severity=Severity.D2_PATH, allow_fork_after=2,
                   max_divergence_ratio=0.3, min_matched_ratio=0.5,
                   forbid_crash=False)
        again = Budget.from_dict(b.to_dict())
        self.assertEqual(again.name, "a")
        self.assertIs(again.max_severity, Severity.D2_PATH)
        self.assertEqual(again.allow_fork_after, 2)
        self.assertFalse(again.forbid_crash)

    def test_unknown_keys_are_ignored(self):
        """配置文件里多写了个字段不该炸 —— 前向兼容比严格校验重要。"""
        b = Budget.from_dict({"name": "a", "future_option": 1})
        self.assertEqual(b.name, "a")

    def test_load_budgets_from_dir(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "one.json").write_text('{"max_severity": "d2_path"}', encoding="utf-8")
            Path(d, "two.json").write_text('{"max_severity": "d4_safety"}', encoding="utf-8")
            got = load_budgets(d)
            self.assertEqual(sorted(got), ["one", "two"])
            self.assertIs(got["two"].max_severity, Severity.D4_SAFETY)

    def test_load_budgets_missing_dir_returns_empty(self):
        self.assertEqual(load_budgets("/no/such/dir/anywhere"), {})


class TestSummaryIsReadable(unittest.TestCase):
    def test_summary_mentions_budget_and_ratios(self):
        r = check([verdict_d0(), verdict_d2()], Budget(name="demo"))
        s = r.summary()
        self.assertIn("demo", s)
        self.assertIn("分叉率", s)
        self.assertIn("匹配率", s)


class TestModuleSelfCheck(unittest.TestCase):
    def test_self_check_clean(self):
        self.assertEqual(self_check(), [])


class TestShippedBudgetsAreCoherent(unittest.TestCase):
    """仓库里那三份预算必须自洽。

    这条测试盯的是**数据文件**，不是代码 —— 而预算恰恰是以数据形式发布的策略。
    有人手改 JSON 把 floor 调到 max_severity 之下，代码测试全绿，
    但 CI 从此永远红着。**永远红着的门禁和没有门禁是同一回事**，
    所以这份自洽性必须有自动化守着。
    """

    def setUp(self):
        from replayprobe.cli import ROOT

        self.budgets = load_budgets(ROOT / "data" / "budgets")
        self.assertTrue(self.budgets, "没加载到任何预算 —— 路径变了？")

    def test_all_shipped_budgets_are_self_consistent(self):
        for name, b in self.budgets.items():
            with self.subTest(budget=name):
                self.assertEqual(b.audit(), [], f"{name} 自相矛盾")

    def test_the_three_expected_budgets_exist(self):
        self.assertEqual(set(self.budgets),
                         {"default", "prompt_tweak", "model_swap"})

    def test_default_is_the_strictest(self):
        """默认预算必须是最紧的那个 —— 门禁应该一开始就红。

        反过来（先松后紧）在实践中几乎不会发生：松惯了没人愿意再收紧。
        """
        d = self.budgets["default"]
        for name, b in self.budgets.items():
            with self.subTest(against=name):
                self.assertLessEqual(
                    severity_rank(d.max_severity), severity_rank(b.max_severity),
                    f"default 比 {name} 还松")


if __name__ == "__main__":
    unittest.main()
