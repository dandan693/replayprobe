"""端到端：录制 → 精确重放 → 钻取重放 → 断带保护。

这一组测试是整个项目的**自证**。它要证明的不是"代码能跑"，
而是三件更强的事：

1. **确定性**：只用录制带重放 N 次，逐字一致（跑不出这个，后面全是空话）。
2. **可切开**：钻取重放真的在第 N 步切到真实执行，而不是看起来切了。
3. **不撒谎**：带不完整时，回放器拒绝即兴编一个成功。

需要真值库（`data/truth/retail_truth.db`）。缺库时整组跳过而不是失败 ——
但在 CI 里应当把库建出来，否则你等于没测。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from replayprobe.agent import ReActAgent, make_tools, tool_schema_hash
from replayprobe.diverge import compare_traces
from replayprobe.llm import LLMResponse, ScriptedLLM, TASKS, scripted
from replayprobe.player import Player, TapeMiss, TapeMismatch, load_tape, save_tape
from replayprobe.recorder import Recorder, llm_key, tool_key
from replayprobe.types import (
    ReplayMode,
    Severity,
    StepKind,
    Tape,
    TapeEntry,
    TapeManifest,
)

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "truth" / "retail_truth.db"
QUESTION = TASKS["total"]["question"]


def record(variant: str = "baseline", task: str = "total",
           tape_id: str = "test-tape"):
    """真实跑一遍并产出 (录制带, 基线轨迹)。"""
    tools = make_tools(DB)
    manifest = TapeManifest(
        tape_id=tape_id, task_id=task, agent_variant="react-v1",
        model=f"scripted-{variant}", tool_schema_hash=tool_schema_hash(tools),
        dataset_snapshot=DB.name,
    )
    rec = Recorder(manifest)
    agent = ReActAgent(rec.wrap_llm(scripted(variant, task)), rec.wrap_tools(tools))
    trace = agent.run(TASKS[task]["question"], task_id=task, run_id=f"r-{variant}",
                      variant=variant, mode=ReplayMode.LIVE)
    return rec.finish(), trace


def replay(tape, mode, *, live_variant=None, fork_at=None, variant="replay",
           task="total", question=None):
    """用给定模式和变体回放，返回 (轨迹, Player)。

    `question` 必须和录制时一致 —— 它进 messages，进而进 `llm_key`。
    传错问题的后果不是"结果不同"，而是**第一条记录就找不到**，
    于是 exact 模式直接抛 TapeMismatch。这正好说明 key 是真的在起作用。
    """
    tools = make_tools(DB)
    live = scripted(live_variant, task) if live_variant else None
    player = Player(tape, mode, live_llm=live, fork_at=fork_at)
    agent = ReActAgent(player.wrap_llm(live), player.wrap_tools(tools))
    trace = agent.run(question or TASKS[task]["question"], task_id=task,
                      run_id=f"r-{mode.value}", variant=variant, mode=mode,
                      fork_at=fork_at)
    return trace, player


@unittest.skipUnless(DB.exists(),
                     f"需要真值库：{DB}（先跑 tools/build_truth_db.py --download）")
class TestRecordAndReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tape, cls.baseline = record()

    # ── ① 确定性 ────────────────────────────────────────────────────

    def test_tape_is_not_empty(self):
        self.assertGreater(len(self.tape.entries), 0)
        self.assertTrue(self.tape.manifest.tool_schema_hash)

    def test_limitations_are_declared(self):
        """凡不能保证不可变的东西必须写出来 —— 这是诚实与自我欺骗的分界线。"""
        self.assertTrue(self.tape.manifest.limitations)

    def test_exact_replay_is_byte_identical_across_runs(self):
        a, pa = replay(self.tape, ReplayMode.EXACT)
        b, pb = replay(self.tape, ReplayMode.EXACT)
        self.assertEqual(pa.used_live, 0, "exact 模式绝不允许真实执行")
        self.assertEqual(pb.used_live, 0)
        self.assertIs(compare_traces(a, b).severity, Severity.D0_IDENTICAL)

    def test_exact_replay_matches_baseline(self):
        """这是整个项目的前提：带完好，且重放确定性。"""
        a, _ = replay(self.tape, ReplayMode.EXACT)
        v = compare_traces(self.baseline, a)
        self.assertIs(v.severity, Severity.D0_IDENTICAL, v.reasons)

    def test_all_steps_come_from_tape(self):
        a, _ = replay(self.tape, ReplayMode.EXACT)
        self.assertTrue(all(s.source == "tape" for s in a.steps))
        # FINAL 步没有对应的带条目（它是模型回复的收口，不是一次调用），
        # 所以只对 LLM_CALL / TOOL_CALL 要求 tape_seq 可回溯。
        for s in a.steps:
            if s.kind in (StepKind.LLM_CALL, StepKind.TOOL_CALL):
                self.assertIsNotNone(s.tape_seq, f"{s.kind.value} 步没有回溯到带")

    # ── ② 可切开 ────────────────────────────────────────────────────

    def test_drill_really_switches_to_live(self):
        """钻取必须真的切到真实执行。

        这一条专门防一个隐蔽的假象：脚本替身如果只看"调用计数器"，
        切开后它会吐出第一轮的答案，于是 drill 看起来"什么都没变" ——
        而那是假的。`_turn_of` 从 messages 推轮次就是为了这个。
        """
        drilled, p = replay(self.tape, ReplayMode.DRILL,
                            live_variant="late_fork", fork_at=2, variant="late_fork")
        self.assertEqual(p.used_tape, 2)
        self.assertGreater(p.used_live, 0)
        self.assertTrue(any(s.source == "live" for s in drilled.steps))

    def test_drill_produces_a_real_fork(self):
        """late_fork 中途多查一次总件数 → 新探索 → D2（而不是 D0/D1）。"""
        drilled, _ = replay(self.tape, ReplayMode.DRILL,
                            live_variant="late_fork", fork_at=2, variant="late_fork")
        v = compare_traces(self.baseline, drilled)
        self.assertIs(v.severity, Severity.D2_PATH, v.reasons)
        self.assertGreater(v.evidence["insertions"], 0)

    def test_drill_prefix_is_stable(self):
        """切开点之前必须是逐字一致的 —— 否则"控制变量"无从谈起。"""
        drilled, _ = replay(self.tape, ReplayMode.DRILL,
                            live_variant="late_fork", fork_at=2, variant="late_fork")
        self.assertEqual(drilled.fork_at, 2)
        head_a = [s.payload for s in self.baseline.steps[:2]]
        head_b = [s.payload for s in drilled.steps[:2]]
        self.assertEqual(head_a, head_b)

    # ── ③ 不撒谎 ────────────────────────────────────────────────────

    def test_exact_refuses_to_improvise_on_incomplete_tape(self):
        """**铁律。** 带不完整宁可炸，也不能编一个成功。"""
        truncated = Tape(manifest=self.tape.manifest, entries=self.tape.entries[:2])
        with self.assertRaises(TapeMismatch):
            replay(truncated, ReplayMode.EXACT)

    def test_strict_records_deviation_and_falls_back_to_live(self):
        """strict 是我不知道会不会偏、让它自己偏 —— 但要如实记录。"""
        truncated = Tape(manifest=self.tape.manifest, entries=self.tape.entries[:2])
        trace, p = replay(truncated, ReplayMode.STRICT, live_variant="baseline")
        self.assertEqual(len(p.deviations), 1)
        self.assertGreater(len(trace.steps), 0)

    def test_drill_before_fork_point_must_hit_tape(self):
        """切开点之前带里没有，就是真错误，不能悄悄走 live。"""
        truncated = Tape(manifest=self.tape.manifest, entries=self.tape.entries[:1])
        with self.assertRaises(TapeMiss):
            replay(truncated, ReplayMode.DRILL, live_variant="baseline", fork_at=4)

    def test_missing_live_llm_raises_instead_of_faking(self):
        """没有可用模型时也要报错，不能返回一个空回复。"""
        truncated = Tape(manifest=self.tape.manifest, entries=self.tape.entries[:2])
        with self.assertRaises(TapeMiss):
            replay(truncated, ReplayMode.STRICT)  # 不给 live


@unittest.skipUnless(DB.exists(), f"需要真值库：{DB}")
class TestToolFailurePayloadIsPreserved(unittest.TestCase):
    """工具失败路径的 payload 必须逐字保真。

    这是修过的一个真 bug：录制层记 `TypeError: x`，agent 层重建成
    `RuntimeError: x`，于是重放时凭空多出一处**假分叉**。
    修法是让录制/回放层把权威 payload 挂在异常上，agent 层直接用它。
    """

    @staticmethod
    def _bad_sql_policy(n: int, messages: list[dict]) -> LLMResponse:
        turn = max(1, (len(messages) - 1) // 2 + 1) if messages else n
        if turn == 1:
            return LLMResponse(content="我来查一下。", finish="tool_use",
                               tool_call={"name": "run_sql",
                                          "arguments": {"sql": "SELECT * FROM no_such_table"}})
        return LLMResponse(content="数据中不存在该表，无法给出结果。", finish="stop")

    def _record_failure(self):
        tools = make_tools(DB)
        manifest = TapeManifest(tape_id="fail-tape", task_id="t-fail",
                                tool_schema_hash=tool_schema_hash(tools))
        rec = Recorder(manifest)
        agent = ReActAgent(rec.wrap_llm(ScriptedLLM(self._bad_sql_policy)),
                           rec.wrap_tools(tools))
        trace = agent.run("查一个不存在的表", task_id="t-fail", mode=ReplayMode.LIVE)
        return rec.finish(), trace

    def test_failure_is_recorded_honestly(self):
        tape, trace = self._record_failure()
        tool_steps = [s for s in trace.steps if s.kind is StepKind.TOOL_CALL]
        self.assertEqual(len(tool_steps), 1)
        self.assertFalse(tool_steps[0].payload["ok"])
        self.assertTrue(tool_steps[0].payload["error"])

    def test_exact_replay_reproduces_failure_payload_verbatim(self):
        tape, baseline = self._record_failure()
        replayed, p = replay(tape, ReplayMode.EXACT, variant="exact",
                             question="查一个不存在的表")
        self.assertEqual(p.used_live, 0)

        def tool_payloads(tr):
            return [s.payload for s in tr.steps if s.kind is StepKind.TOOL_CALL]

        self.assertEqual(tool_payloads(baseline), tool_payloads(replayed),
                         "工具失败的 payload 在重放时漂移了 —— 会凭空多出假分叉")
        self.assertIs(compare_traces(baseline, replayed).severity,
                      Severity.D0_IDENTICAL)


class TestTapeSerialization(unittest.TestCase):
    """不依赖真值库：录制带的读写必须无损。"""

    def test_roundtrip_is_lossless(self):
        tape = Tape(
            manifest=TapeManifest(tape_id="t", task_id="q", model="m",
                                  limitations=["模型别名不算不可变标识"]),
            entries=[
                TapeEntry(seq=1, kind=StepKind.LLM_CALL, key="llm:abc",
                          payload={"content": "hi", "tool_call": None, "finish": "stop"},
                          input_hash="i1", output_hash="o1", meta={"model": "m"}),
                TapeEntry(seq=2, kind=StepKind.TOOL_CALL, key="tool:run_sql:def",
                          payload={"name": "run_sql", "ok": False,
                                   "error": "OperationalError: no such table"},
                          parent_seq=1),
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            p = save_tape(tape, Path(d) / "t.json")
            again = load_tape(p)
        self.assertEqual(again.manifest.tape_id, "t")
        self.assertEqual(again.manifest.limitations, tape.manifest.limitations)
        self.assertEqual([e.to_dict() for e in again.entries],
                         [e.to_dict() for e in tape.entries])

    def test_keys_are_stable_under_formatting_noise(self):
        """key 的职责是"找回记录"，不是"判断分叉" ——

        所以它用规范化参数计算：顺手改个空格不该导致找不到记录。
        """
        k1 = tool_key("run_sql", {"sql": "SELECT  1  FROM t"})
        k2 = tool_key("run_sql", {"sql": "select 1 from t;"})
        self.assertEqual(k1, k2)

        messages = [{"role": "user", "content": "问题"}]
        self.assertEqual(llm_key(messages), llm_key(list(messages)))

    def test_tool_schema_hash_notices_a_changed_default(self):
        """**回归测试。** 第一版指纹只哈希了工具名。

        那等于说"改了参数默认值也算同一套工具" —— 而默认值会改变模型行为。
        只哈希名字的指纹会**在改动发生时保持沉默**，比没有指纹更糟：
        它让你以为自己验过了。
        """
        def f(a: int = 1):
            return a

        def g(a: int = 2):
            return a

        self.assertNotEqual(tool_schema_hash({"t": f}), tool_schema_hash({"t": g}))
        self.assertNotEqual(tool_schema_hash({"f": f}), tool_schema_hash({"g": f}))
        self.assertEqual(tool_schema_hash({"t": f}), tool_schema_hash({"t": f}))


if __name__ == "__main__":
    unittest.main()
