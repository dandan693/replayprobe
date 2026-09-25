"""分级层的测试：五档必须各就各位。

这一层最怕的不是报错，而是**悄悄把 D3 记成 D1** ——
报表一片绿，实际问题被抹平。所以每档都单独钉一条。
"""

from __future__ import annotations

import unittest

from replayprobe.diverge import (
    DEFAULT_TEXT_SIM_THRESHOLD,
    conclusion_relation,
    compare_traces,
    grade_steps,
    self_check,
)
from replayprobe.types import ReplayMode, Severity, Step, StepKind, Trace, severity_rank


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


def error(msg: str = "boom") -> Step:
    return Step(seq=0, kind=StepKind.ERROR,
                payload={"type": "RuntimeError", "message": msg})


SQL_TOTAL = "SELECT SUM(Amount) FROM retail"
SQL_UK = "SELECT SUM(Amount) FROM retail WHERE Country='United Kingdom'"


def base() -> list[Step]:
    return [
        llm_call("run_sql", {"sql": SQL_TOTAL}),
        tool_call("run_sql", {"sql": SQL_TOTAL}),
        final("总销售额为 8887208.89 美元，共 18532 笔订单。"),
    ]


class TestSeverityLadder(unittest.TestCase):
    def test_d0_identical(self):
        v = grade_steps(base(), base())
        self.assertIs(v.severity, Severity.D0_IDENTICAL)

    def test_d1_wording_only(self):
        """措辞变了、决策没变 → D1。**不能更重**，否则噪音淹没信号。"""
        b = base()
        b[2] = final("全部销售额是 8,887,208.89 美元，共 18532 笔订单。")
        v = grade_steps(base(), b)
        self.assertIs(v.severity, Severity.D1_WORDING)

    def test_d2_when_extra_number_but_original_kept(self):
        """多答一个数字（原数都在）→ D2，不是 D3。

        这是"用集合运算替代语义判断"那一处设计在起作用：
        原结论的每个数字都还在，它没有被任何新证据否定。
        """
        b = base()
        b[2] = final("总销售额为 8887208.89 美元，共 18532 笔订单，覆盖 4338 位客户。")
        v = grade_steps(base(), b)
        self.assertIs(v.severity, Severity.D2_PATH)

    def test_d2_when_query_target_changed_but_conclusion_same(self):
        b = [
            llm_call("run_sql", {"sql": SQL_UK}),
            tool_call("run_sql", {"sql": SQL_UK}),
            base()[2],
        ]
        v = grade_steps(base(), b)
        self.assertIs(v.severity, Severity.D2_PATH)

    def test_d3_when_number_replaced(self):
        """最不能漏的一档。"""
        b = base()
        b[2] = final("总销售额为 4443606.45 美元，共 18532 笔订单。")
        v = grade_steps(base(), b)
        self.assertIs(v.severity, Severity.D3_CONCLUSION)

    def test_d3_when_percentage_shifted_by_hair(self):
        """占比变了 0.01 个百分点也必须抓到。"""
        a = base()
        a[2] = final("前十名客户贡献 74.66%。")
        b = list(a)
        b[2] = final("前十名客户贡献 74.65%。")
        v = grade_steps(a, b)
        self.assertIs(v.severity, Severity.D3_CONCLUSION)

    def test_d4_when_refusal_appears(self):
        b = base()
        b[2] = final("数据中不存在该字段，无法给出结果。")
        v = grade_steps(base(), b)
        self.assertIs(v.severity, Severity.D4_SAFETY)

    def test_d4_when_crash_diverges(self):
        v = grade_steps(base(), [base()[0], error()])
        self.assertIs(v.severity, Severity.D4_SAFETY)


class TestRetryVersusExplore(unittest.TestCase):
    """改一次提示词最常见的结果是"多做了一次"。

    把重试记成 D2，门禁就成了天天响的闹钟。这个区分必须成立。
    """

    def test_pure_retry_is_d1(self):
        a = base()
        b = [a[0], a[1],
             llm_call("run_sql", {"sql": SQL_TOTAL}),
             tool_call("run_sql", {"sql": SQL_TOTAL}),
             a[2]]
        v = grade_steps(a, b)
        self.assertIs(v.severity, Severity.D1_WORDING,
                      f"单纯重试被判成 {v.severity.value} —— D2 会被高频噪音淹没")

    def test_new_exploration_is_d2(self):
        a = base()
        b = [a[0], a[1],
             llm_call("get_schema", {"table": "retail"}),
             tool_call("get_schema", {"table": "retail"}),
             a[2]]
        v = grade_steps(a, b)
        self.assertIs(v.severity, Severity.D2_PATH)


SIM_A = "本次分析没有发现季节性差异，需要更多数据。"
SIM_B = "本次分析没有发现季节性差异，需要更多的数据。"


class TestConclusionRelation(unittest.TestCase):
    def test_safety_relation_has_priority_over_numbers(self):
        a = [final("总销售额为 8887208.89 美元。")]
        b = [final("无法查询该数据，数据中不存在相关信息 99999。")]
        rel, reason = conclusion_relation(a, b)
        self.assertEqual(rel, "safety")
        self.assertIn("防护策略", reason)

    def test_missing_final_on_one_side(self):
        rel, _ = conclusion_relation([final("答案 1000。")], [llm_call("t", {})])
        self.assertEqual(rel, "changed")

    def test_similar_texts_are_same(self):
        """两边都没有显著数字时，才退回文本相似度。"""
        rel, reason = conclusion_relation([final(SIM_A)], [final(SIM_B)])
        self.assertEqual(rel, "same")
        self.assertIn(str(DEFAULT_TEXT_SIM_THRESHOLD), reason)

    def test_threshold_is_actually_wired(self):
        """同一对文本，把阈值提到 0.99 就该判成不同 ——

        证明那个阈值不是写着好看的。**全项目唯一的模糊判据必须真的可控。**
        """
        rel, _ = conclusion_relation([final(SIM_A)], [final(SIM_B)],
                                     text_sim_threshold=0.99)
        self.assertEqual(rel, "changed")

    def test_numbers_take_priority_over_text_similarity(self):
        """有显著数字时，文本再像也不能走相似度 —— 数字才是结论。"""
        rel, _ = conclusion_relation(
            [final("总销售额为 8887208.89 美元。")],
            [final("总销售额为 4443606.45 美元。")],
        )
        self.assertEqual(rel, "changed")


class TestCompareTracesAndEvidence(unittest.TestCase):
    def test_evidence_is_populated(self):
        v = compare_traces(
            Trace(run_id="r1", task_id="t", variant="a", mode=ReplayMode.LIVE, steps=base()),
            Trace(run_id="r2", task_id="t", variant="b", mode=ReplayMode.EXACT, steps=base()),
        )
        for key in ("matched", "insertions", "deletions", "decision_mismatches",
                    "conclusion_relation", "text_sim_threshold", "crashed"):
            self.assertIn(key, v.evidence)

    def test_error_steps_are_excluded_from_alignment(self):
        """崩溃不参与逐步比对 —— 它由 _crashed 独立判定。"""
        a = base()
        b = [a[0], error(), a[1], a[2]]
        v = compare_traces(
            Trace(run_id="r1", task_id="t", variant="a", mode=ReplayMode.LIVE, steps=a),
            Trace(run_id="r2", task_id="t", variant="b", mode=ReplayMode.LIVE, steps=b),
        )
        self.assertTrue(v.evidence["crashed"]["replay"])
        self.assertFalse(v.evidence["crashed"]["baseline"])
        self.assertIs(v.severity, Severity.D4_SAFETY)

    def test_first_divergence_has_both_sides(self):
        """第一处分叉必须左右两侧都能说清"各自做了什么"。

        这里刻意换**工具**而不是换 SQL 参数：摘要会被截断到固定宽度，
        两条只在 WHERE 子句上不同的 SQL 截断后长得一模一样 ——
        那测的就不是对照展示，而是截断算法了。
        """
        a = base()
        b = [llm_call("get_schema", {"table": "retail"}), a[1], a[2]]
        v = grade_steps(a, b)
        fd = v.first_divergence
        self.assertIsNotNone(fd)
        self.assertTrue(fd.a_summary and fd.b_summary)
        self.assertNotEqual(fd.a_summary, fd.b_summary)


class TestProvenanceAndUnexplainedFork(unittest.TestCase):
    """溯源：一份分了 8 组的报告，必须能回答"你到底改了什么"。

    这一块是接真实模型之后补的。之前 `trace.manifest` 一直是空的，
    于是报告拿着一条轨迹，却不知道自己是谁、在什么条件下跑出来的 ——
    而「同名不同源」正是这套工具最想消除的困惑。
    """

    @staticmethod
    def _trace(variant: str, manifest: dict) -> Trace:
        return Trace(run_id="r", task_id="t", variant=variant,
                     mode=ReplayMode.LIVE, steps=base(), manifest=manifest)

    def test_changed_variables_detects_model_swap(self):
        a = self._trace("m1", {"model": "qwen-plus", "prompt_hash": "p1",
                               "tool_schema_hash": "t1", "dataset_snapshot": "d1"})
        b = self._trace("m2", {"model": "qwen-turbo", "prompt_hash": "p1",
                               "tool_schema_hash": "t1", "dataset_snapshot": "d1"})
        v = compare_traces(a, b)
        self.assertEqual(v.evidence["provenance"]["changed"], ["model"])

    def test_changed_variables_detects_prompt_tweak(self):
        a = self._trace("m1", {"model": "m", "prompt_hash": "p1"})
        b = self._trace("m1", {"model": "m", "prompt_hash": "p2"})
        self.assertEqual(compare_traces(a, b).evidence["provenance"]["changed"],
                         ["prompt_hash"])

    def test_unknown_hash_is_not_reported_as_change(self):
        """空哈希是「没记录」，不是「变了」。

        把「没记录」当成「变了」会制造假警报 ——
        而假警报多了，真警报就没人看了。
        """
        a = self._trace("m1", {"model": "m", "prompt_hash": ""})
        b = self._trace("m1", {"model": "m", "prompt_hash": "p2"})
        self.assertEqual(compare_traces(a, b).evidence["provenance"]["changed"], [])

    def test_identical_identity_but_forked_is_flagged(self):
        """什么都没改却分叉 —— 必须显式标出来。

        这说明分叉不是使用者的改动引入的，而是上游（供应商路由、缓存、
        别名指向别的权重）带来的。**这是"确定性"这个前提本身出了裂缝**，
        而不是一条普通的 D2。混在一起统计，就等于把工具自身的局限
        算到了使用者的账上。
        """
        m = {"model": "m", "prompt_hash": "p", "tool_schema_hash": "t",
             "dataset_snapshot": "d"}
        a = self._trace("m1", m)
        forked = base()
        forked[2] = final("总销售额为 8887208.89 美元，共 18532 笔订单，覆盖 4338 位客户。")
        b = Trace(run_id="r", task_id="t", variant="m1", mode=ReplayMode.LIVE,
                  steps=forked, manifest=m)
        v = compare_traces(a, b)
        self.assertGreaterEqual(severity_rank(v.severity), severity_rank(Severity.D2_PATH))
        self.assertTrue(v.evidence["unexplained_fork"])
        self.assertTrue(any("非确定性" in r for r in v.reasons))

    def test_d0_with_same_identity_is_not_flagged(self):
        m = {"model": "m", "prompt_hash": "p"}
        v = compare_traces(self._trace("m1", m), self._trace("m1", m))
        self.assertIs(v.severity, Severity.D0_IDENTICAL)
        self.assertNotIn("unexplained_fork", v.evidence)

    def test_mode_difference_counts_as_a_changed_variable(self):
        """**"怎么跑的"也是自变量。**

        drill 与 live 是两个不同的实验。第一版没把 mode 算进 `changed`，
        于是 drill 的比较被归进「什么都没改」那一组 ——
        对照组的「0% 分叉」被污染成了 14.3%，而对照组恰恰是
        唯一能测出上游非确定性的那组。
        """
        m = {"model": "m", "prompt_hash": "p"}
        a = self._trace("live", m)
        b = Trace(run_id="r", task_id="t", variant="drill", mode=ReplayMode.DRILL,
                  steps=base(), manifest=m)
        self.assertIn("mode", compare_traces(a, b).evidence["provenance"]["changed"])

    def test_same_mode_is_not_a_change(self):
        m = {"model": "m", "prompt_hash": "p"}
        v = compare_traces(self._trace("m1", m), self._trace("m1", m))
        self.assertEqual(v.evidence["provenance"]["changed"], [])

    def test_drill_fork_is_not_reported_as_unexplained(self):
        """**drill 的分叉是我们自己切出来的，不是上游抖动。**

        第一版只检查「变量没改」就报警，于是在 drill 比较上误报 ——
        而 drill 的整个意义就是"从第 N 步起让它走另一条路"。
        拿它去报"上游非确定性"，等于把实验设计当成了故障。

        误报的代价在这个项目里被反复强调：警报一多，真警报就没人看了。
        """
        m = {"model": "m", "prompt_hash": "p", "tool_schema_hash": "t",
             "dataset_snapshot": "d"}
        a = self._trace("live", m)  # mode=LIVE
        forked = base()
        forked[2] = final("总销售额为 8887208.89 美元，共 18532 笔订单，覆盖 4338 位客户。")
        b = Trace(run_id="r", task_id="t", variant="drill", mode=ReplayMode.DRILL,
                  steps=forked, manifest=m, fork_at=2)
        v = compare_traces(a, b)
        self.assertGreaterEqual(severity_rank(v.severity), severity_rank(Severity.D2_PATH))
        self.assertFalse(v.evidence.get("unexplained_fork"), "drill 的分叉被误报成上游抖动")
        self.assertFalse(any("非确定性" in r for r in v.reasons))

    def test_exact_replay_divergence_is_not_reported_as_unexplained(self):
        """比较 live 与 exact 重放时同理：差异只可能来自"一边打带一边没打"。"""
        m = {"model": "m", "prompt_hash": "p"}
        a = self._trace("live", m)
        forked = base()
        forked[2] = final("总销售额为 4443606.45 美元。")
        b = Trace(run_id="r", task_id="t", variant="exact", mode=ReplayMode.EXACT,
                  steps=forked, manifest=m)
        v = compare_traces(a, b)
        self.assertFalse(v.evidence.get("unexplained_fork"))


class TestModuleSelfCheck(unittest.TestCase):
    def test_self_check_clean(self):
        self.assertEqual(self_check(), [])


if __name__ == "__main__":
    unittest.main()
