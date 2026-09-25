#!/usr/bin/env python3
"""端到端演示：录一次，放三次。

这个脚本是整个项目最短的上手入口 —— 它不做任何需要联网、需要 Key 的事，
纯离线跑完，把四件事一次性证明给你看：

    ① 录制        真实跑一遍，把每个决策接缝固化成「带」
    ② 精确重放    只用带跑，跑 N 次结果必须完全一致（可复现性自证）
    ③ 钻取重放    前 N 步照旧，之后让模型自由发挥 → 制造出真实分叉
    ④ 断带保护    带不完整时，回放器**拒绝即兴编一个成功**，而是如实报偏离

用法：
    python tools/demo_replay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replayprobe.agent import ReActAgent, make_tools, tool_schema_hash  # noqa: E402
from replayprobe.budget import Budget, check  # noqa: E402
from replayprobe.diverge import compare_traces  # noqa: E402
from replayprobe.llm import scripted  # noqa: E402
from replayprobe.player import (  # noqa: E402
    Player, TapeMismatch, load_tape, save_tape,
)
from replayprobe.recorder import Recorder  # noqa: E402
from replayprobe.types import ReplayMode, Severity, Tape, TapeManifest  # noqa: E402

DB = ROOT / "data" / "truth" / "retail_truth.db"
TAPE_PATH = ROOT / "data" / "tapes" / "baseline_q_total.json"
QUESTION = "全量总销售额是多少美元？"
LINE = "=" * 72


def rule(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def show(trace, label: str) -> None:
    print(f"  [{label}] {len(trace.steps)} 步  variant={trace.variant} mode={trace.mode.value}")
    for s in trace.steps:
        from replayprobe.signature import summarize

        flag = "带" if s.source == "tape" else "实"
        print(f"    seq{s.seq:<2d} {flag} {s.kind.value:9s} {summarize(s, 56)}")


def main() -> int:
    tools = make_tools(DB)
    ok = True

    # ── ① 录制 ────────────────────────────────────────────────────────
    rule("① 录制：真实跑一遍，把每条「输入 → 输出」固化下来")
    manifest = TapeManifest(
        tape_id="baseline_q_total", task_id="q-total",
        agent_variant="react-v1", model="scripted-baseline",
        tool_schema_hash=tool_schema_hash(tools),
        dataset_snapshot="retail_truth.db / 392,692 行",
        notes="基线：模型一轮取数后直接给结论",
    )
    rec = Recorder(manifest)
    agent = ReActAgent(rec.wrap_llm(scripted("baseline")), rec.wrap_tools(tools))
    baseline = agent.run(QUESTION, task_id="q-total", run_id="r-baseline", variant="baseline",
                         mode=ReplayMode.LIVE)
    tape = rec.finish()
    save_tape(tape, TAPE_PATH)
    show(baseline, "baseline / live")
    print(f"\n  录制带落盘：{TAPE_PATH.relative_to(ROOT)}")
    print(f"  条目 {len(tape.entries)} 条 | 工具指纹 {tape.manifest.tool_schema_hash}")
    for lim in tape.manifest.limitations:
        print(f"  [限制] {lim}")

    # ── ② 精确重放 ────────────────────────────────────────────────────
    rule("② 精确重放：只用带跑，跑 2 次结果必须逐字一致")
    reps = []
    for i in range(2):
        p = Player(tape, ReplayMode.EXACT)
        a = ReActAgent(p.wrap_llm(), p.wrap_tools(tools))
        reps.append(a.run(QUESTION, task_id="q-total", run_id=f"r-exact-{i}", variant="exact",
                          mode=ReplayMode.EXACT))
    show(reps[0], "exact #1")

    d_same = compare_traces(reps[0], reps[1])
    d_base = compare_traces(baseline, reps[0])
    print(f"\n  两次 exact 之间      : {d_same.severity.value}")
    print(f"  baseline vs exact    : {d_base.severity.value}")
    if d_same.severity.value != "d0_identical" or d_base.severity.value != "d0_identical":
        ok = False
        print("  [!!] 精确重放没能做到逐字一致 —— 可复现性自证失败")
    else:
        print("  [OK] 带是完好的，且重放是确定性的 —— 后面所有比较都建立在这个前提上")

    # ── ③ 钻取重放 ────────────────────────────────────────────────────
    rule("③ 钻取重放：前 2 步照旧，第 3 步起让模型自由发挥")
    print("  场景：模型版本升级后变得「更谨慎」，取数之后又多核对了一次总件数。")
    print("  fork_at=2 表示：第 1、2 步严格按原样，从第 3 步开始真实执行。\n")
    p = Player(tape, ReplayMode.DRILL, live_llm=scripted("late_fork"), fork_at=2)
    a = ReActAgent(p.wrap_llm(), p.wrap_tools(tools))
    drilled = a.run(QUESTION, task_id="q-total", run_id="r-drill", variant="late_fork",
                    mode=ReplayMode.DRILL, fork_at=2)
    show(drilled, "drill / late_fork")
    print(f"\n  用带 {p.used_tape} 步 / 真实执行 {p.used_live} 步")

    v = compare_traces(baseline, drilled)
    print(f"\n  分叉判定：{v.severity.value}")
    for r in v.reasons:
        print(f"    · {r}")
    print(f"    对齐：匹配 {v.evidence['matched']} 步，"
          f"插入 {v.evidence['insertions']} 步，"
          f"决策分叉 {v.evidence['decision_mismatches']} 处")
    if v.first_divergence:
        d = v.first_divergence
        print(f"    第一处分叉（对齐位置 {d.index}）：")
        print(f"      基线 seq{d.a_seq}: {d.a_summary}")
        print(f"      重放 seq{d.b_seq}: {d.b_summary}")

    # ── ③b 对照：序号对齐会怎么报 ─────────────────────────────────────
    rule("③b 对照实验：同一对轨迹，换「序号对齐」会怎么报")
    from replayprobe.align import divergence_count

    n_index = divergence_count(baseline.transcript_steps(), drilled.transcript_steps(),
                               method="index")
    n_sig = divergence_count(baseline.transcript_steps(), drilled.transcript_steps(),
                             method="signature")
    print(f"  序号对齐（朴素做法）报出 {n_index} 处决策分叉")
    print(f"  签名对齐（本项目）  报出 {n_sig} 处决策分叉")
    print()
    print("  真实情况：只有 1 处新决策（多查了一次总件数），")
    print("  其余「分叉」全是多了一步之后整体错位造成的假象。")
    print("  **这就是为什么不能按序号对齐** —— 它看起来像真的发现了问题。")

    # ── ④ 断带保护 ────────────────────────────────────────────────────
    rule("④ 断带保护：带不完整时，回放器会不会即兴编一个成功？")
    truncated = Tape(manifest=tape.manifest, entries=tape.entries[:2])
    print(f"  故意把带截断成前 {len(truncated.entries)} 条（模拟录制中途进程被杀）\n")

    p_x = Player(truncated, ReplayMode.EXACT)
    a_x = ReActAgent(p_x.wrap_llm(), p_x.wrap_tools(tools))
    try:
        a_x.run(QUESTION, task_id="q-total", run_id="r-trunc", variant="exact", mode=ReplayMode.EXACT)
    except TapeMismatch as exc:
        print(f"  exact  -> 如实抛错：{str(exc)[:64]}…")
    else:
        ok = False
        print("  [!!] exact 模式居然没报错 —— 那它就在假装成功")

    p_s = Player(truncated, ReplayMode.STRICT, live_llm=scripted("baseline"))
    a_s = ReActAgent(p_s.wrap_llm(), p_s.wrap_tools(tools))
    strict_trace = a_s.run(QUESTION, task_id="q-total", run_id="r-strict", variant="strict",
                           mode=ReplayMode.STRICT)
    print(f"  strict -> 记录 {len(p_s.deviations)} 处偏离并用真实执行兜底，跑完 "
          f"{len(strict_trace.steps)} 步")
    for d in p_s.deviations:
        print(f"            第 {d['step']} 步：{d['detail']}")
    print()
    print("  两种反应都对，但都不是「编一个成功」。")
    print("  一个会自己编答案的裁判，比没有裁判更危险。")

    # ── ⑤ 预算结算 ────────────────────────────────────────────────────
    rule("⑤ 分叉预算：把「允许分叉多少」变成 CI 能判的退出码")
    verdicts = [d_base, v]
    for name, b in (
        ("default（只允许 D1：措辞差异）", Budget(name="default")),
        ("relaxed（允许 D2，且容忍一半任务分叉）",
         Budget(name="relaxed", max_severity=Severity.D2_PATH,
                max_divergence_ratio=0.6)),
    ):
        r = check(verdicts, b)
        print(f"\n  {name}")
        for line in r.summary().splitlines():
            print(f"    {line}")
        for vio in r.violations:
            print(f"    [!] {vio.task_id:12s} {vio.rule:24s} {vio.detail[:52]}")

    rule("结论")
    print("  录制把一次运行变成了资产：它现在可以被重放、被切分、被断言。")
    print("  而「分叉」不是一个布尔值 —— 措辞变了（D1）、多绕一步（D1/D2）、")
    print("  结论变了（D3）、结局性质变了（D4），四种事的处置完全不同。")
    print()
    print("  最终状态:", "全部通过" if ok else "有断言失败（见上）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
