"""回放器：把录制带重新喂进 Agent。

## 三种模式，三个不同的实验

混在一起是调试 Agent 时最容易骗到自己的地方：「重放一次看起来一样」
不等于「行为被复现了」。所以模式必须显式选、报告里必须回显。

| 模式 | 带的使用 | 遇到带里没有的请求 | 回答什么问题 |
|---|---|---|---|
| `exact` | 全程用带 | **抛异常** | 这条轨迹本身完好、且可复现吗？ |
| `strict` | 优先用带 | **调 live 并记录偏离点** | 新版本从第几步开始走岔了？ |
| `drill` | 前 `fork_at` 步用带 | 过了切开点后**主动**全走 live | 如果第 N+1 步换个做法，后面会怎样？ |

`strict` 和 `drill` 的区别值得说清楚：**strict 是我不知道会不会偏，让它自己偏；
drill 是我指定从哪儿切。** 前者用于回归检测，后者用于反事实实验。

## 一条铁律：绝不为带里没有的请求即兴编一个成功

这是回放系统最容易犯、后果最隐蔽的错。重放时如果发出的请求带里没有，
**正确反应是停下并标记偏离**，而不是返回一个看起来合理的默认值。

因为一旦开始即兴，重放就会**一路绿灯地跑完**，然后告诉你「一切正常」——
而它其实已经偏离了原轨迹，后面所有比较都失去了参照。
**一个会自己编答案的裁判，比没有裁判更危险。**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .llm import LLM, LLMResponse
from .recorder import llm_key, tool_key
from .types import ReplayMode, StepKind, Tape, TapeEntry


class TapeMismatch(RuntimeError):
    """exact 模式下，运行的请求与带不符 —— 带或代码已经变了。"""


class TapeMiss(RuntimeError):
    """带里没有对应的记录。**不会被静默忽略**，见模块开头。"""


def _resp_from_payload(p: dict) -> LLMResponse:
    return LLMResponse(
        content=p.get("content", "") or "",
        tool_call=p.get("tool_call"),
        finish=p.get("finish", "stop"),
    )


class Player:
    """把带喂进 Agent。内部维护一个单调指针，只往前走。"""

    def __init__(self, tape: Tape, mode: ReplayMode,
                 live_llm: LLM | None = None,
                 live_tools: dict[str, Callable[..., Any]] | None = None,
                 fork_at: int | None = None) -> None:
        if mode is ReplayMode.STRICT and live_llm is None and live_tools is None:
            # strict 允许「不提供 live」——此时遇到偏离就停下并如实报告，
            # 这也是一种合法用法（只想检测，不想继续跑）
            pass
        self.tape = tape
        self.mode = mode
        self.live_llm = live_llm
        self.live_tools = live_tools or {}
        self.fork_at = fork_at
        self.step = 0
        self._ptr = 0
        self._by_seq = tape.by_seq()

        self.deviations: list[dict] = []
        """偏离记录：重放时发出了带里没有的请求。
        **这不是错误，是发现。** 它就是我们要找的「从哪一步开始不一样」。"""

        self.used_tape = 0
        self.used_live = 0

    # ── 内部 ────────────────────────────────────────────────────────

    def _use_tape(self) -> bool:
        if self.mode in (ReplayMode.EXACT, ReplayMode.STRICT):
            return True
        if self.mode is ReplayMode.DRILL:
            return self.fork_at is None or self.step <= self.fork_at
        return False

    def _take(self, kind: StepKind, key: str) -> TapeEntry | None:
        """从指针处向后找第一条 kind 匹配且 key 匹配的记录。

        为什么不是「严格按 seq 取」：一旦中途出现重试或结构变化，
        严格按 seq 会一路错位到底。而按 key 找，能容忍「多用了一次同样的调用」。
        """
        for i in range(self._ptr, len(self.tape.entries)):
            e = self.tape.entries[i]
            if e.kind is not kind:
                continue
            if e.key == key:
                self._ptr = i + 1
                return e
        return None

    def _miss_handle(self, kind: StepKind, key: str) -> None:
        """带里没有这条请求时该怎么办 —— 三种模式三种反应，**绝不即兴**。"""
        msg = f"带里没有对应的 {kind.value} 记录（key={key[:32]}…）"
        if self.mode is ReplayMode.EXACT:
            raise TapeMismatch(msg + "；exact 模式要求逐条命中")
        if self.mode is ReplayMode.DRILL and self._use_tape():
            raise TapeMiss(
                msg + f"；drill 的切开点在第 {self.fork_at} 步，之前必须命中"
            )
        # STRICT：记录偏离，交给调用方决定用 live 兜底
        self.deviations.append({
            "step": self.step, "kind": kind.value, "key": key,
            "detail": "重放发出的请求在带里不存在",
        })

    # ── 对外的两个接缝 ──────────────────────────────────────────────

    def wrap_llm(self, live: LLM | None = None) -> LLM:
        player = self
        fallback = live or self.live_llm

        class _P:
            current_source = "tape"
            current_tape_seq: int | None = None

            @property
            def model(self) -> str:
                return getattr(fallback, "model", "") or getattr(player.live_llm, "model", "")

            def chat(self, messages: list[dict]) -> LLMResponse:
                player.step += 1
                key = llm_key(messages)
                if player._use_tape():
                    entry = player._take(StepKind.LLM_CALL, key)
                    if entry is not None:
                        self.current_source = "tape"
                        self.current_tape_seq = entry.seq
                        player.used_tape += 1
                        return _resp_from_payload(entry.payload)
                    player._miss_handle(StepKind.LLM_CALL, key)
                # 走到这里 = 该走 live（drill 切开后 / strict 的偏离兜底）
                if fallback is None:
                    raise TapeMiss(
                        f"第 {player.step} 步需要真实执行，但没有提供可用的模型"
                    )
                self.current_source = "live"
                self.current_tape_seq = None
                player.used_live += 1
                return fallback.chat(messages)

        return _P()  # type: ignore[return-value]

    def wrap_tools(self, live_tools: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        player = self
        wrapped = _PlayerTools(player, live_tools)
        return wrapped


class _PlayerTools(dict):
    current_source = "tape"
    current_tape_seq: int | None = None

    def __init__(self, player: Player, live_tools: dict[str, Callable[..., Any]]) -> None:
        super().__init__()
        self._player = player
        self._live = live_tools
        for name, fn in live_tools.items():
            self[name] = self._wrap(name, fn)

    def _wrap(self, name: str, live_fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(**kwargs: Any) -> Any:
            p = self._player
            p.step += 1
            key = tool_key(name, kwargs)
            if p._use_tape():
                entry = p._take(StepKind.TOOL_CALL, key)
                if entry is not None:
                    self.current_source = "tape"
                    self.current_tape_seq = entry.seq
                    p.used_tape += 1
                    payload = entry.payload
                    if not payload.get("ok", True):
                        # 把带里那份**原始** payload 挂上去，让 agent 原样记录。
                        # 如果这里只抛一句 RuntimeError，agent 会重建出一份
                        # error 字符串略有差异的 payload —— 而重放的全部意义
                        # 就在于「与带逐字一致」，一点点漂移都是致命的。
                        exc = RuntimeError(payload.get("error") or "工具失败（录自带）")
                        setattr(exc, "replayprobe_payload", payload)
                        raise exc
                    return payload.get("value")
                p._miss_handle(StepKind.TOOL_CALL, key)
            self.current_source = "live"
            self.current_tape_seq = None
            p.used_live += 1
            return live_fn(**kwargs)

        wrapper.__name__ = name
        return wrapper


# --------------------------------------------------------------------------- #
# 带的读写
# --------------------------------------------------------------------------- #


def save_tape(tape: Tape, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tape.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_tape(path: str | Path) -> Tape:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    from .types import TapeManifest

    return Tape(
        manifest=TapeManifest.from_dict(raw.get("manifest") or {}),
        entries=[TapeEntry.from_dict(e) for e in raw.get("entries") or []],
    )
