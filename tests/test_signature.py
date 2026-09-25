"""签名层的测试。

这一层是技术支点：它错了，后面所有判定都是错的。
所以这里每一条断言都写明"它防的是哪种错"。
"""

from __future__ import annotations

import unittest

from replayprobe.signature import (
    canonical_json,
    content_hash,
    decision_signature,
    detect_refusal,
    extract_numbers,
    intent_signature,
    normalize_sql,
    number_relation,
    self_check,
    summarize,
)
from replayprobe.types import Step, StepKind


def llm_step(sql: str, content: str = "查一下") -> Step:
    return Step(seq=1, kind=StepKind.LLM_CALL, payload={
        "content": content, "finish": "tool_use",
        "tool_call": {"name": "run_sql", "arguments": {"sql": sql}},
    })


def final_step(text: str) -> Step:
    return Step(seq=9, kind=StepKind.FINAL, payload={"text": text})


class TestCanonicalJson(unittest.TestCase):
    def test_key_order_does_not_change_hash(self):
        """规范化序列化必须与键顺序无关 —— 否则同一份内容会有两个指纹。"""
        self.assertEqual(content_hash({"a": 1, "b": 2}), content_hash({"b": 2, "a": 1}))

    def test_non_ascii_not_escaped(self):
        """不转义非 ASCII：中文以原样出现，报告里能直接肉眼核对。"""
        self.assertIn("销售", canonical_json({"k": "销售"}))

    def test_semantic_change_changes_hash(self):
        """真的改了内容，指纹必须变 —— 否则指纹没有证据价值。"""
        self.assertNotEqual(content_hash({"a": 1}), content_hash({"a": 2}))


class TestNormalizeSql(unittest.TestCase):
    def test_only_case_and_whitespace(self):
        """只做关键字小写 + 空白压缩，不做等价重写。"""
        a = normalize_sql("SELECT SUM(Amount) FROM retail;")
        b = normalize_sql("  select   sum(Amount)\n from retail  ")
        self.assertEqual(a, b)

    def test_does_not_flatten_different_columns(self):
        """**不能把不同的列名抹平。** 抹平了就是漏报真分叉。"""
        self.assertNotEqual(normalize_sql("SELECT SUM(Amount) FROM retail"),
                            normalize_sql("SELECT SUM(Quantity) FROM retail"))


class TestDecisionSignature(unittest.TestCase):
    def test_wording_and_sql_formatting_are_not_a_fork(self):
        """措辞 + SQL 空白/大小写/分号差异 → 决策签名必须一致。"""
        a = llm_step("SELECT SUM(Amount) FROM retail;", "我来查总销售额。")
        b = llm_step("  select sum(Amount) from retail  ", "好的，马上查。")
        self.assertEqual(decision_signature(a), decision_signature(b))

    def test_different_query_target_is_a_fork(self):
        """换了查询目标 → 决策签名必须不同。漏报比误报危险得多。"""
        a = llm_step("SELECT SUM(Amount) FROM retail")
        b = llm_step("SELECT COUNT(*) FROM retail")
        self.assertNotEqual(decision_signature(a), decision_signature(b))

    def test_intent_signature_survives_tool_swap(self):
        """意图层更粗：换了工具但都还在"调工具"，意图签名应当相同。"""
        a = llm_step("SELECT SUM(Amount) FROM retail")
        b = Step(seq=1, kind=StepKind.LLM_CALL, payload={
            "content": "看下表结构", "finish": "tool_use",
            "tool_call": {"name": "get_schema", "arguments": {"table": "retail"}},
        })
        self.assertNotEqual(decision_signature(a), decision_signature(b))
        self.assertEqual(intent_signature(a), intent_signature(b))

    def test_final_equivalent_wording(self):
        """千分位与句式不同，结论数字一致 → 决策签名相同。"""
        a = final_step("总销售额为 8,887,208.89 美元，涉及 18532 笔订单。")
        b = final_step("全部销售额是 8887208.89（共 18532 单）。")
        self.assertEqual(decision_signature(a), decision_signature(b))

    def test_final_replaced_number_differs(self):
        """金额被替换 → 签名必须不同。这是本项目最不能漏的一类。"""
        a = final_step("总销售额为 8887208.89 美元，涉及 18532 笔订单。")
        b = final_step("总销售额为 4443604.45 美元，涉及 18532 笔订单。")
        self.assertNotEqual(decision_signature(a), decision_signature(b))


class TestExtractNumbers(unittest.TestCase):
    def test_percentages_are_kept(self):
        """**回归测试。** 曾经因为"只按数量级筛"把百分比静默丢掉。

        74.66% 和 81.97% 是零售数据集最关键的两个占比，
        漏掉它们等于对"占比类结论分叉"完全失明 —— 而且不报错，只是少几个数字。
        """
        got = extract_numbers("前十名客户贡献 74.66%，其中英国占 81.97%。")
        self.assertIn(74.66, got)
        self.assertIn(81.97, got)

    def test_decimals_are_kept(self):
        self.assertIn(0.85, extract_numbers("阈值取 0.85。"))

    def test_small_structural_integers_dropped(self):
        """「前 3 名」「5 个国家」不承载结论，必须滤掉，否则是 D3 的误报源。"""
        self.assertEqual(extract_numbers("前三名门店、共 5 个国家的数据。"), [])

    def test_large_numbers_kept(self):
        got = extract_numbers("总销售额 8,887,208.89 美元，共 18532 笔订单。")
        self.assertIn(8887208.89, got)
        self.assertIn(18532, got)

    def test_known_limitation_years_are_collected(self):
        """**已知局限，不是 bug。** 年份是整数且 >= 100，会被收进来。

        把这个行为钉成测试，是为了防止有人"顺手修一下"却没意识到
        它会让 4 位数的结构性数字重新溜进来。
        """
        self.assertIn(2026, extract_numbers("数据止于 2026 年。"))


class TestNumberRelation(unittest.TestCase):
    def test_same(self):
        self.assertEqual(number_relation([1.0], [1.0]), "same")

    def test_superset_means_info_change_not_refutation(self):
        """多答一个数：原数都还在 → 不是推翻结论。这个区分撑起了 D2/D3 的分界。"""
        self.assertEqual(number_relation([8887208.89], [8887208.89, 18532]), "b_superset")
        self.assertEqual(number_relation([8887208.89, 18532], [8887208.89]), "a_superset")

    def test_differs_means_refutation(self):
        self.assertEqual(number_relation([8887208.89], [4443604.45]), "differs")

    def test_both_empty(self):
        self.assertEqual(number_relation([], []), "both_empty")

    def test_precision_difference_is_not_a_refutation(self):
        """**只差小数位数，不算换了结论。**

        这一条是接真实模型之后才补的，因为真实模型立刻把它撞了出来：

            真实模型答：全量总销售额是 8887208.894 美元。
            脚本替身答：全量总销售额为 8,887,208.89 美元，共 18,532 笔订单。

        同一次 SUM 的两种表述（原始值 vs 保留两位），第一版判成了
        "结论被替换" → D3。而 D3 是"该拦下来"的那一档，误报代价最高 ——
        模型只要改个小数位数，门禁就会红。
        """
        self.assertEqual(number_relation([8887208.894], [8887208.89]), "same")
        self.assertEqual(number_relation([74.66], [74.7]), "same")

    def test_different_magnitude_is_still_a_refutation(self):
        """但"位数差一位"和"数值差一点"必须分得开，否则这个放宽就太松了。"""
        self.assertEqual(number_relation([8887208.894], [8887208]), "differs")
        self.assertEqual(number_relation([8887200.0], [8887208.0]), "differs")

    def test_one_side_empty_is_differs_not_superset(self):
        """一侧一个显著数字都没有 → 保守判"结论换了"。

        基线什么都没给出、重放给了个数，这更可能是换了结论，
        而不是"信息量增加了"。把这种情况放宽成 D2 会静默削弱 D3 档。
        """
        self.assertEqual(number_relation([], [8887208.89]), "differs")
        self.assertEqual(number_relation([8887208.89], []), "differs")

    def test_partial_overlap_is_differs(self):
        """有交集但不是包含关系 → `differs`。

        这是比"完全不重叠"更隐蔽的一种坏：两个结论各说对了一半。
        用集合的 `<` 判断会漏掉它，必须是"每个元素都能配上"才算包含。

        （注：上游 `extract_numbers` 会去重，所以到达这里的都是互不相同的值；
        这里的多重集配对是给直接调用 `number_relation` 的用法兜底。）
        """
        self.assertEqual(number_relation([1.0, 2.0], [1.0, 3.0]), "differs")
        self.assertEqual(number_relation([1.0, 2.0], [1.0, 2.0, 3.0]), "b_superset")
        self.assertEqual(number_relation([1.0, 2.0, 3.0], [1.0, 2.0]), "a_superset")

    def test_precision_tolerated_inside_superset(self):
        """精度差异不该让"信息量增加"退化成"结论被替换"。"""
        self.assertEqual(number_relation([8887208.894], [18532, 8887208.89]),
                         "b_superset")


class TestRefusal(unittest.TestCase):
    def test_detects_refusal(self):
        self.assertTrue(detect_refusal("数据中不存在该字段，无法给出结果。"))

    def test_normal_answer_not_refusal(self):
        self.assertFalse(detect_refusal("总销售额为 8887208.89 美元。"))


class TestSummarize(unittest.TestCase):
    def test_shows_original_sql_not_normalized(self):
        """摘要给人看，必须保留原始大小写 ——

        把 `SELECT ROUND(SUM(Amount), 2)` 显示成小写，会让人怀疑
        "我录的东西是不是被改了"。签名负责判断，摘要负责沟通。

        `max_len` 给足，否则测的就不是"保留原貌"而是"截断算法"。
        """
        sql = "SELECT ROUND(SUM(Amount), 2) FROM retail"
        s = summarize(llm_step(sql), max_len=200)
        self.assertIn(sql, s)

    def test_clips_long_text(self):
        s = summarize(final_step("答" * 200), max_len=20)
        self.assertLessEqual(len(s), 20)

    def test_none_step(self):
        self.assertEqual(summarize(None), "(缺失)")


class TestModuleSelfCheck(unittest.TestCase):
    def test_self_check_clean(self):
        self.assertEqual(self_check(), [])


if __name__ == "__main__":
    unittest.main()
