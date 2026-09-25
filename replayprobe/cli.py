"""命令行入口。

一条立场，和 faultprobe 保持一致：**框架必须能在没有网络、没有 Key 的情况下
完整跑一遍**。否则没人会在 CI 里跑它，也没人会在 clone 之后验证它真的能跑 ——
而一个没人跑的门禁，等于不存在。

    检查自己      replayprobe check
    录一条带      replayprobe run --task total --variant baseline --out reports/base.json
    精确重放      replayprobe run --tape reports/base.tape.json --mode exact --out reports/exact.json
    钻取重放      replayprobe run --tape reports/base.tape.json --mode drill --fork-at 2 \
                                   --live late_fork --task total --out reports/drill.json
    比较两条轨迹  replayprobe compare --a reports/base.json --b reports/drill.json
    跑门禁        replayprobe gate --dir reports --budget default
    出报告        replayprobe report --dir reports

真实模型（需要 Key，见 tools/probe_real_llm.py 先验协议）：

    replayprobe run --task total --llm real --model qwen-plus-latest --temperature 0 \
                    --api-key-file "/path/to/key.txt" --out reports/real_a.json

## 一条命名约定（曾经在这里踩过坑）

`--out` **永远指轨迹**（trace），不管是录制还是回放。
录制时额外产出的录制带写到 `--tape-out`，默认是 `<out 同目录>/<out 文件名>.tape.json`。

这么定是因为之前录制分支把带写到 `--out`、回放分支把轨迹写到 `--out` ——
同一个参数在两个分支里意思不一样，于是 `compare` 拿着回放产物的路径去读轨迹，
报 `FileNotFoundError`。**同名参数必须同义，否则调用方只能靠猜。**
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.json"


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _db_path(cfg: dict, override: str | None = None) -> Path:
    if override:
        return Path(override)
    rel = (cfg.get("data") or {}).get("truth_db", "data/truth/retail_truth.db")
    return ROOT / rel


def _write_json(obj, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _sidecar(path: str | Path, suffix: str) -> Path:
    """在同一目录里派生一个旁挂文件名。

    `reports/base.json` + `.tape.json` → `reports/base.tape.json`
    （先把原扩展名摘掉再后缀，否则会得到 `base.json.tape.json` 这种
    「两个扩展名叠在一起」的名字，肉眼一眼看不出它属于谁）。
    """
    p = Path(path)
    stem = p.with_suffix("") if p.suffix else p
    return stem.with_name(stem.name + suffix)


# --------------------------------------------------------------------------- #
# check：先证明工具自己是好的
# --------------------------------------------------------------------------- #


def cmd_check(args: argparse.Namespace) -> int:
    """跑全部模块自检。

    **这一步比跑实验重要。** 框架的 bug 会报错；而**判据**的 bug
    会让你得出错误结论还浑然不觉 —— 尤其是「漏报分叉」这一类，
    它表现为一片安静，而不是一片红。
    """
    from . import align, budget, diverge, llm, signature

    problems: list[str] = []
    modules = [
        ("signature", signature),
        ("align", align),
        ("diverge", diverge),
        ("budget", budget),
        ("llm", llm),
    ]
    print(f"replayprobe {__version__} · 模块自检")
    print("-" * 60)
    for name, mod in modules:
        try:
            found = mod.self_check()
        except Exception as exc:  # noqa: BLE001
            found = [f"自检本身崩了：{type(exc).__name__}: {exc}"]
        status = "通过" if not found else f"不通过（{len(found)} 条）"
        print(f"  {name:10s} {status}")
        for p in found:
            print(f"      [x] {p}")
            problems.append(f"{name}: {p}")

    # 数据集与带
    cfg = load_config(args.config)
    db = _db_path(cfg, args.db)
    print("-" * 60)
    if db.exists():
        import sqlite3

        # closing() 而不是裸的 `with sqlite3.connect(...)`：后者不关连接。
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
            n = conn.execute("SELECT COUNT(*) FROM retail").fetchone()[0]
        print(f"  真值库            {db.name}（{n:,} 行）")
        if n != 392_692:
            problems.append(f"真值库行数 {n} 与预期的 392,692 不一致 —— 口径可能漂了")
    else:
        print(f"  真值库            缺失：{db}")
        print("                    先跑：python tools/build_truth_db.py --download")
        problems.append("真值库不存在，无法录制轨迹")

    tape_dir = ROOT / ((cfg.get("data") or {}).get("tapes_dir", "data/tapes"))
    files = sorted(tape_dir.glob("*.json")) if tape_dir.exists() else []
    # 按**内容**而不是文件名判断它是不是录制带。
    # 目录里可能同时躺着轨迹（`*.trace.json`，键是 steps），
    # 第一版的 glob 把它也数进去了，于是清单里出现一行「0 条记录」——
    # 看起来像是带坏了。**列出来的东西不对，比不列更糟**：
    # 它让人去查一个根本不存在的问题。
    tapes, others = [], []
    for f in files:
        raw = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(raw.get("entries"), list) and isinstance(raw.get("manifest"), dict):
            tapes.append((f, raw))
        else:
            others.append(f.name)
    print(f"  录制带            {len(tapes)} 份")
    for f, raw in tapes:
        lim = (raw.get("manifest") or {}).get("limitations") or []
        print(f"    {f.name}  {len(raw['entries'])} 条"
              f"{'  [限制 ' + str(len(lim)) + ' 条]' if lim else ''}")
    if others:
        print(f"  （同目录另有 {len(others)} 个非录制带文件，未计入："
              f"{'、'.join(others)}）")

    print("-" * 60)
    if problems:
        print(f"自检未通过：{len(problems)} 个问题")
        return 1
    print("自检全部通过")
    return 0


# --------------------------------------------------------------------------- #
# run：录制 or 回放
# --------------------------------------------------------------------------- #


def _system_prompt(args: argparse.Namespace) -> str:
    """生效的 system prompt。

    `--system-suffix` 是「改一个变量，再放一遍」这条主线的最小入口：
    只追加、不替换，于是两次运行的差异只可能来自追加的那一段。
    **如果允许整体替换，实验就从「改一条规则的效果」变成了「换了个人」**，
    后者也能得到分叉，但你没法解释是哪个改动引起的。
    """
    from .agent.core import SYSTEM_PROMPT

    suffix = (getattr(args, "system_suffix", None) or "").strip()
    return f"{SYSTEM_PROMPT}\n{suffix}" if suffix else SYSTEM_PROMPT


def _make_live_llm(args: argparse.Namespace, cfg: dict, tools: dict):
    """造「真实执行」用的那个模型。返回 `(llm, label, extra_limitations)`。

    **`scripted` 与 `real` 走同一条装配路径**，只是喂进去的实现不同。
    这不是为了代码整齐 —— 是为了保证**两条路跑的是同一个 agent、同一套工具、
    同一个 system prompt**。如果真实模型那条路要单独复制一份 agent 循环，
    那么"脚本替身跑出来的结论也适用于真实模型"这句话就不成立了，
    而整个框架的可信度就建立在它上面。

    `label` 会写进轨迹的 `variant` 字段，并在报告里原样回显 ——
    这样一份报告摆在面前时，能一眼看出哪些数字来自替身、哪些来自真实模型。
    """
    from .llm import TASKS, VARIANTS, scripted  # noqa: F401

    if args.llm == "scripted":
        # 真实模型专用的参数在替身路径上是**无意义**的。
        # 静默忽略它们，会让人以为自己改了什么 —— 而实验结论看起来仍然"正常"，
        # 只是解释错了。宁可在这时候拦住。
        stray = [n for n, v in (("--model", args.model), ("--temperature", args.temperature),
                                ("--api-key-file", args.api_key_file),
                                ("--max-tokens", args.max_tokens),
                                ("--system-suffix", getattr(args, "system_suffix", None)))
                 if v is not None]
        if stray:
            raise ValueError(
                f"{'、'.join(stray)} 只在 --llm real 时生效，当前是 scripted 替身。"
                "脚本替身的行为由 --variant 决定，改 system prompt 不会影响它。"
            )
        variant = args.live or args.variant
        if variant not in VARIANTS:
            raise ValueError(f"未知变体 {variant}，可选 {VARIANTS}")
        return scripted(variant, args.task), variant, []

    from .agent import openai_tool_schemas
    from .llm import make_real_llm

    llm = make_real_llm(
        cfg,
        tool_schemas=openai_tool_schemas(tools),
        api_key_file=args.api_key_file,
        model=args.model,
        temperature=args.temperature,
        system_prompt=_system_prompt(args),
        max_tokens=args.max_tokens,
    )
    label = f"real:{llm.model}@t{llm.temperature}"
    limitations = [
        "真实模型的返回**不保证可复现**：temperature=0 只是降低随机性，不等于确定性"
        " —— 而这恰恰是本工具要测量的对象，不是它的缺陷。",
        "轨迹里记录的是该次调用的实际返回；上游若对同一 prompt 做了缓存或路由，"
        "本工具无从知晓，也不假装知道。",
    ]
    # 关于「只记录了模型别名」那条通用限制，由 Recorder.finish() 统一补 ——
    # 不在这里重复一遍。同一条限制出现两次，会让人以为有两个独立的问题。
    return llm, label, limitations


def cmd_run(args: argparse.Namespace) -> int:
    from .agent import ReActAgent, make_tools, tool_schema_hash
    from .player import Player, load_tape, save_tape
    from .recorder import Recorder
    from .signature import content_hash
    from .types import ReplayMode, TapeManifest

    cfg = load_config(args.config)
    db = _db_path(cfg, args.db)
    if not db.exists():
        print(f"真值库不存在：{db}", file=sys.stderr)
        return 2

    from .llm import TASKS

    if args.task not in TASKS:
        print(f"未知任务 {args.task}，可选 {tuple(TASKS)}", file=sys.stderr)
        return 2
    question = TASKS[args.task]["question"]
    tools = make_tools(db)

    try:
        live_llm, label, extra_lims = _make_live_llm(args, cfg, tools)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"模型装配失败：{exc}", file=sys.stderr)
        return 2

    max_steps = args.max_steps or int((cfg.get("agent") or {}).get("max_steps", 8))
    effective_prompt = _system_prompt(args)

    # ── 录制 ──────────────────────────────────────────────────────────
    if not args.tape:
        trace_path = Path(args.out)
        tape_path = Path(args.tape_out) if args.tape_out else _sidecar(args.out, ".tape.json")
        notes = f"llm={args.llm}; label={label}; max_steps={max_steps}"
        if getattr(args, "system_suffix", None):
            notes += f"; system_suffix={args.system_suffix!r}"
        manifest = TapeManifest(
            tape_id=tape_path.stem, task_id=args.task,
            agent_variant=(cfg.get("agent") or {}).get("system_prompt_version", "react-v1"),
            model=getattr(live_llm, "model", label),
            # prompt_hash 之前一直是空的 —— 一个"实现了但没人填"的字段
            # 比没有这个字段更糟：它让人以为 prompt 被固定住了。
            prompt_hash=content_hash(effective_prompt),
            tool_schema_hash=tool_schema_hash(tools),
            dataset_snapshot=db.name,
            notes=notes,
            limitations=list(extra_lims),
        )
        rec = Recorder(manifest)
        agent = ReActAgent(rec.wrap_llm(live_llm), rec.wrap_tools(tools),
                           max_steps=max_steps)
        try:
            trace = agent.run(question, task_id=args.task, run_id=f"r-{args.task}-{label}",
                              variant=label, mode=ReplayMode.LIVE)
        except Exception as exc:  # noqa: BLE001 —— 模型侧的错误要原样暴露
            print(f"录制中止：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        tape = rec.finish()
        save_tape(tape, tape_path)
        # **把 provenance 挂回轨迹。** 之前 trace.manifest 一直是空的，
        # 于是报告拿着一条轨迹，却不知道它是哪个模型、哪版 prompt 跑出来的 ——
        # 而「同名不同源」恰恰是这套工具最想消除的困惑。
        # 轨迹和带各存一份是刻意的：带可以单独分享给别人复现，
        # 轨迹要能脱离带被读懂。
        trace.manifest = manifest.to_dict()
        _write_json(trace.to_dict(), trace_path)
        print("录制完成")
        print(f"  模型   {label}")
        print(f"  录制带 {tape_path}（{len(tape.entries)} 条记录）")
        print(f"  轨迹   {trace_path}（{len(trace.steps)} 步）")
        for lim in tape.manifest.limitations:
            print(f"  [限制] {lim}")
        return 0

    # ── 回放 ──────────────────────────────────────────────────────────
    tape = load_tape(args.tape)
    mode = ReplayMode(args.mode)
    if args.llm == "real":
        # exact 模式下 live 模型永远不会被用到，不构造也不联网；
        # strict / drill 才需要它兜底。提前说明，免得有人以为 exact 在偷偷调模型。
        live = live_llm if mode is not ReplayMode.EXACT else None
        variant_label = label
    else:
        # 替身路径保持原语义：没给 --live 就是「没有兜底模型」，
        # 而不是「用一个默认的 baseline 兜底」—— 后者会让 drill 的
        # 实验条件变得隐晦（"为什么切开后是这个行为？"）。
        live = live_llm if args.live else None
        variant_label = args.live or mode.value
    player = Player(tape, mode, live_llm=live, fork_at=args.fork_at)
    agent = ReActAgent(player.wrap_llm(live), player.wrap_tools(tools),
                       max_steps=max_steps)
    try:
        trace = agent.run(question, task_id=args.task,
                          run_id=f"r-{mode.value}-{variant_label}",
                          variant=variant_label, mode=mode, fork_at=args.fork_at)
    except Exception as exc:  # noqa: BLE001 —— 偏离本身就是结果，要如实报
        print(f"回放中止：{type(exc).__name__}: {exc}", file=sys.stderr)
        if player.deviations:
            print(f"  已记录 {len(player.deviations)} 处偏离：", file=sys.stderr)
            for d in player.deviations:
                print(f"    第 {d['step']} 步 {d['kind']}：{d['detail']}", file=sys.stderr)
        return 3

    # 回放的轨迹也带上来源：不是自己录的带的那份 manifest，而是**它读的那条带**的。
    # 这样「重放出来的这条轨迹是谁的标准答案的复现」是可查的，
    # 而不是只能靠文件名去猜。
    trace.manifest = {**tape.manifest.to_dict(), "replayed_from": str(args.tape)}
    _write_json(trace.to_dict(), args.out)
    print(f"回放完成：{args.out}")
    print(f"  模式 {mode.value} | 模型 {variant_label} | {len(trace.steps)} 步 | "
          f"用带 {player.used_tape} 步 / 真实执行 {player.used_live} 步")
    if player.deviations:
        print(f"  偏离 {len(player.deviations)} 处：")
        for d in player.deviations:
            print(f"    第 {d['step']} 步 {d['kind']}：{d['detail']}")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def _trace_from_file(path: str | Path):
    from .types import ReplayMode, Step, StepKind, Trace

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = [
        Step(seq=int(s["seq"]), kind=StepKind(s["kind"]), payload=s.get("payload") or {},
             source=s.get("source", "live"), tape_seq=s.get("tape_seq"),
             parent_seq=s.get("parent_seq"), meta=s.get("meta") or {})
        for s in raw.get("steps") or []
    ]
    return Trace(
        run_id=raw.get("run_id", ""), task_id=raw.get("task_id", ""),
        variant=raw.get("variant", ""), mode=ReplayMode(raw.get("mode", "live")),
        steps=steps, fork_at=raw.get("fork_at"),
        manifest=raw.get("manifest") or {}, notes=raw.get("notes") or [],
    )


def cmd_compare(args: argparse.Namespace) -> int:
    from .diverge import compare_traces

    a, b = _trace_from_file(args.a), _trace_from_file(args.b)
    v = compare_traces(a, b)
    print(f"分叉判定：{v.severity.value}")
    for r in v.reasons:
        print(f"  · {r}")
    e = v.evidence
    print(f"  对齐：匹配 {e['matched']} / 插入 {e['insertions']} / 删除 {e['deletions']} "
          f"| 决策分叉 {e['decision_mismatches']} 处 | 结论关系 {e['conclusion_relation']}")
    if v.first_divergence:
        d = v.first_divergence
        print(f"  第一处分叉（对齐位置 {d.index}）：")
        print(f"    基线 seq{d.a_seq}: {d.a_summary}")
        print(f"    重放 seq{d.b_seq}: {d.b_summary}")

    # 默认把判定结果落到 `<b 文件名>.verdict.json`。
    # 这样 `compare` 后面接 `gate --dir reports` 就能直接读到输入 ——
    # 否则每跑一次比较都要手工誊一遍路径，人一烦就不跑了，门禁也就没人用。
    out = Path(args.out) if args.out else _sidecar(args.b, ".verdict.json")
    _write_json(v.to_dict(), out)
    print(f"  已写出 {out}")
    return 0


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #


def cmd_gate(args: argparse.Namespace) -> int:
    from .budget import Budget, check, load_budgets
    from .types import Verdict

    cfg = load_config(args.config)
    budgets = load_budgets(ROOT / ((cfg.get("data") or {}).get("budgets_dir", "data/budgets")))
    if args.budget not in budgets:
        print(f"找不到预算 {args.budget!r}；可用：{sorted(budgets)}", file=sys.stderr)
        return 2
    budget: Budget = budgets[args.budget]

    d = Path(args.dir)
    # 去重：`verdict.verdict.json` 同时匹配两个 glob，不去重会被结算两遍。
    files = sorted({*d.glob("*.verdict.json"), *d.glob("verdict*.json")})
    verdicts = []
    for p in files:
        raw = json.loads(p.read_text(encoding="utf-8"))
        verdicts.append(_verdict_from_dict(raw))

    # `--select`：只结算「改动变量恰好是这一组」的比较。
    #
    # 为什么需要它：一个 `reports/` 目录里往往同时躺着好几种实验 ——
    # 同配置重跑（什么都没改）、改提示词、换模型。它们的**合理分叉程度根本不同**，
    # 用一份预算统一判决，结果是换模型那几条把 `prompt_tweak` 预算顶红，
    # 而真正该被拦下的那条反而淹没在噪音里。
    # **按实验类型分开结算，是让门禁可用的前提。**
    dropped = 0
    if args.select:
        # `--select none` 专门指「什么都没改」那组（同一配置重跑）。
        # 没有它的话，这一组永远选不出来 —— 而它恰恰是**唯一能测出
        # 「上游非确定性有多大」**的那组，不是可有可无的。
        want = (set() if args.select.strip().lower() == "none"
                else {s.strip() for s in args.select.split(",") if s.strip()})
        kept = []
        for v in verdicts:
            got = set(((v.evidence or {}).get("provenance") or {}).get("changed") or [])
            if got == want:
                kept.append(v)
        dropped = len(verdicts) - len(kept)
        verdicts = kept
        label_sel = ",".join(sorted(want)) or "无改动"
        if dropped:
            print(f"  已按 --select {label_sel} 过滤掉 {dropped} 条不属于本实验的比较")

    if not verdicts:
        print(f"{d} 下没有符合条件的 *.verdict.json", file=sys.stderr)
        return 2

    res = check(verdicts, budget)
    print(res.summary())
    for v in res.violations:
        print(f"  [!] {v.task_id:14s} {v.rule:24s} {v.detail}")
    if args.out:
        _write_json({"passed": res.passed, "budget": budget.to_dict(),
                     "summary": res.summary(),
                     "violations": [v.to_dict() for v in res.violations]}, args.out)
    return 0 if res.passed else 1


def _verdict_from_dict(raw: dict):
    """把落盘的判定 JSON 读回成 Verdict。

    `alignment.pairs` 不回填 —— 预算结算只看 matched / insertions / deletions 三个计数，
    重建 pairs 既无用处，还会让人误以为这里存了完整的逐步对齐
    （真要展开逐对看，应该去读两侧的轨迹文件，那份才是原始事实）。
    """
    from .types import Alignment, AlignedPair, Divergence, Severity, Verdict

    al = raw.get("alignment") or {}
    v = Verdict(
        task_id=raw.get("task_id", ""),
        baseline_variant=raw.get("baseline_variant", ""),
        replay_variant=raw.get("replay_variant", ""),
        severity=Severity(raw.get("severity", "d0_identical")),
        reasons=raw.get("reasons") or [],
        evidence=raw.get("evidence") or {},
        alignment=Alignment(
            matched=int(al.get("matched", 0)),
            insertions=int(al.get("insertions", 0)),
            deletions=int(al.get("deletions", 0)),
            method=al.get("method", ""),
            pairs=[AlignedPair(a=None, b=None)],
        ),
    )
    for d in raw.get("divergences") or []:
        v.divergences.append(Divergence(**{k: d.get(k) for k in
                                           ("index", "a_seq", "b_seq", "a_summary",
                                            "b_summary", "kind", "reason")}))
    fd = raw.get("first_divergence")
    if fd:
        v.first_divergence = Divergence(**{k: fd.get(k) for k in
                                           ("index", "a_seq", "b_seq", "a_summary",
                                            "b_summary", "kind", "reason")})
    return v


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    out = write_report(Path(args.dir), Path(args.out) if args.out else None,
                       title=args.title)
    print(f"报告已生成：{out}")
    return 0


# --------------------------------------------------------------------------- #
# 参数装配
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replayprobe",
        description="Agent 轨迹回放与分叉诊断：录一次，改一个变量，再放一遍。",
    )
    p.add_argument("--version", action="version", version=f"replayprobe {__version__}")
    p.add_argument("--config", default=None, help="配置文件路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="模块自检 + 数据集体检")
    c.add_argument("--db", default=None)
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="录制一条轨迹，或回放已有的带")
    r.add_argument("--task", default="total", help="任务键（见 llm.TASKS）")
    r.add_argument("--llm", default="scripted", choices=["scripted", "real"],
                   help="用脚本替身（默认，零依赖零联网）还是真实模型执行")
    r.add_argument("--variant", default="baseline",
                   help="--llm scripted 时用哪个行为变体（见 llm.VARIANTS）")
    r.add_argument("--tape", default=None, help="给了就走回放，不给就是录制")
    r.add_argument("--mode", default="exact", choices=["exact", "strict", "drill"])
    r.add_argument("--fork-at", type=int, default=None, help="drill 的切开步号")
    r.add_argument("--live", default=None, help="回放时兜底 / 钻取用的脚本变体")
    r.add_argument("--out", required=True, help="轨迹输出路径（录制 / 回放都是它）")
    r.add_argument("--tape-out", default=None,
                   help="录制时录制带的输出路径；默认 <out>.tape.json")
    r.add_argument("--db", default=None)
    # ── 真实模型相关（--llm real 时生效）──────────────────────────────
    r.add_argument("--model", default=None, help="模型名；默认取 config.json 的 llm.model")
    r.add_argument("--temperature", type=float, default=None)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--api-key-file", default=None,
                   help="API Key 文件路径；也可用环境变量 REPLAYPROBE_API_KEY")
    r.add_argument("--max-steps", type=int, default=None,
                   help="ReAct 循环上限；默认取 config.json 的 agent.max_steps")
    r.add_argument("--system-suffix", default=None,
                   help="追加到 system prompt 末尾的一段规则 —— 用于「改一个变量再放一遍」实验")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("compare", help="比较两条轨迹，输出分叉判定")
    m.add_argument("--a", required=True, help="基线轨迹 JSON")
    m.add_argument("--b", required=True, help="重放轨迹 JSON")
    m.add_argument("--out", default=None,
                   help="判定结果输出路径；默认 <b>.verdict.json（gate 会自动读到）")
    m.set_defaults(func=cmd_compare)

    g = sub.add_parser("gate", help="按分叉预算结算一批比较结果（CI 用）")
    g.add_argument("--dir", default="reports", help="放 verdict JSON 的目录")
    g.add_argument("--budget", default="default")
    g.add_argument("--select", default=None,
                   help="只结算「改动变量恰好等于这组」的比较；逗号分隔，"
                        "如 model 或 prompt_hash。留空则全部结算")
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_gate)

    rp = sub.add_parser("report", help="把轨迹与判定渲染成零依赖 HTML")
    rp.add_argument("--dir", default="reports")
    rp.add_argument("--out", default=None)
    rp.add_argument("--title", default="replayprobe 报告")
    rp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
