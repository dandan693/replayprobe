"""对齐层的测试。

核心命题只有一个：**签名对齐比序号对齐强**，而且这个差距是能量化的。
如果这里跑不出差距，本项目就失去了存在理由。
"""

from __future__ import annotations

import unittest

from replayprobe.align import (
    LEVELS,
    align,
    align_by_index,
    divergence_count,
    mismatch_positions,
    self_check,
)
from replayprobe.types import Step, StepKind


def llm_call(name: str, args: dict, text: str = "") -> Step:
    return Step(seq=0, kind=StepKind.LLM_CALL, payload={
        "content": text or f"调用 {name}", "finish": "tool_use",
        "tool_call": {"name": name, "arguments": args},
    }, source="tape")


def tool_call(name: str, args: dict) -> Step:
    return Step(seq=0, kind=StepKind.TOOL_CALL,
                payload={"name": name, "arguments": args, "ok": True, "value": []},
                source="tape")


def final(text: str) -> Step:
    return Step(seq=0, kind=StepKind.FINAL, payload={"text": text}, source="tape")


def base_trace() -> list[Step]:
    return [
        llm_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        tool_call("run_sql", {"sql": "SELECT SUM(Amount) FROM retail"}),
        llm_call("finish", {}),
        final("总销售额为 8887208.89 美元。"),
    ]


class TestSignatureAlignmentBeatsIndexAlignment(unittest.TestCase):
    """**对照实验。** 这是 README 里那个数字的出处，删了它数字就没了依据。"""

    def setUp(self):
        self.a = base_trace()
        # B 在中间多查了一次表结构，其余三步决策完全一致
        self.b = [
            self.a[0], self.a[1],
            llm_call("get_schema", {"table": "retail"}),
            tool_call("get_schema", {"table": "retail"}),
            self.a[2], self.a[3],
        ]

    def test_signature_alignment_reports_no_decision_fork(self):
        """没有任何决策发生变化，就不该报决策分叉。"""
        al = align(self.a, self.b)
        self.assertEqual(mismatch_positions(al), [])
        self.assertEqual(al.matched, 4)
        self.assertEqual(al.insertions, 2)

    def test_index_alignment_false_positives(self):
        """序号对齐会误报 —— 它把"整体错位"当成了"决策改变"。

        **这个错误方向特别恶劣：它看起来像真的发现了问题。**
        """
        naive = align_by_index(self.a, self.b)
        self.assertTrue(mismatch_positions(naive),
                        "序号对齐居然没误报 —— 对照实现可能被改坏了")

    def test_the_gap_is_real_and_quantified(self):
        n_index = divergence_count(self.a, self.b, method="index")
        n_sig = divergence_count(self.a, self.b, method="signature")
        self.assertGreater(n_index, n_sig)

    def test_divergence_count_default_method_is_signature(self):
        self.assertEqual(divergence_count(self.a, self.b), 0)


class TestRealForksStillDetected(unittest.TestCase):
    def test_changed_query_target_is_detected(self):
        a = base_trace()
        c = list(a)
        c[0] = llm_call("run_sql", {"sql": "SELECT COUNT(*) FROM retail"})
        self.assertTrue(mismatch_positions(align(a, c)),
                        "换了查询目标却没报出分叉 —— 漏报比误报更危险")

    def test_wording_only_is_not_a_fork(self):
        """**纯粹的**措辞变化：数字一个不多一个不少。

        第一版用例手滑多带了一个数字（18532），被判出分叉，一度以为是 bug。
        其实那是对的 —— 签名层只回答"是不是同一串数字"，
        "多答一个数算不算改变结论"由 diverge.py 用集合关系分层。
        **职责不能混。**
        """
        a = base_trace()
        d = list(a)
        d[3] = final("全部销售额是 8,887,208.89 美元。")
        self.assertEqual(mismatch_positions(align(a, d)), [])


class TestAlignmentLevels(unittest.TestCase):
    def test_unknown_level_raises(self):
        with self.assertRaises(ValueError):
            align(base_trace(), base_trace(), level="nope")

    def test_levels_constant(self):
        self.assertEqual(LEVELS, ("full", "decision", "intent"))

    def test_full_level_flags_wording(self):
        """full 级含措辞，所以措辞变化在 full 级必须能被看见 —— 这是 D0 的判据来源。"""
        a = base_trace()
        d = list(a)
        d[3] = final("全部销售额是 8,887,208.89 美元。")
        self.assertTrue(mismatch_positions(align(a, d, level="full"), level="full"))


class TestShapeInfo(unittest.TestCase):
    def test_insert_and_delete_are_counted_separately(self):
        a = base_trace()
        b = a[:2] + [a[0], a[1]] + a[2:]  # B 多两步
        al = align(a, b)
        self.assertEqual(al.insertions, 2)
        self.assertEqual(al.deletions, 0)
        self.assertTrue(al.shape_changed())

    def test_mismatch_positions_ignores_insertions(self):
        """插入/删除本身不算"决策分叉" —— 它是结构变化，由 shape_changed 单独报告。

        混进同一个数字，会让"多绕了一步"和"选择变了"看起来一样严重。
        """
        a = base_trace()
        b = [a[0], a[1], llm_call("get_schema", {"t": 1}), a[2], a[3]]
        al = align(a, b)
        self.assertTrue(al.shape_changed())
        self.assertEqual(len(mismatch_positions(al)), 0)


class TestModuleSelfCheck(unittest.TestCase):
    def test_self_check_clean(self):
        self.assertEqual(self_check(), [])


if __name__ == "__main__":
    unittest.main()
