"""一个最小的 ReAct 循环。

**它做得很薄是故意的。** 这个项目的重点不是「造一个好 Agent」，
而是「让一条已经跑过的轨迹可以被重放和比较」——
Agent 只是轨迹的生产者，做得越简单，轨迹就越好解释。

所以这里没有分层记忆、没有自省、没有多智能体。那些在 faultprobe 里已经做过了。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..llm import LLM
from ..types import ReplayMode, Step, StepKind, Trace

DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = """你是一个数据分析助手。你可以调用工具来查询零售数据。

规则：
1. 需要数据时必须调用工具，不要凭记忆编造数字。
2. 拿到数据后，用一句话给出结论，并把关键数字原样写进结论。
3. 数据里没有的字段，如实说「数据中不存在该字段」，不要猜测。
"""


class ReActAgent:
    """最小 ReAct 循环：模型决定行动 → 执行 → 结果回灌 → 再来一轮。"""

    def __init__(
        self,
        llm: LLM,
        tools: dict[str, Callable[..., Any]],
        max_steps: int = DEFAULT_MAX_STEPS,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(
        self,
        question: str,
        *,
        task_id: str = "",
        run_id: str = "",
        variant: str = "",
        mode: ReplayMode = ReplayMode.LIVE,
        fork_at: int | None = None,
    ) -> Trace:
        trace = Trace(run_id=run_id or "run", task_id=task_id or question[:24],
                      variant=variant, mode=mode, fork_at=fork_at)
        messages: list[dict] = [{"role": "user", "content": question}]
        seq = 0

        for _ in range(self.max_steps):
            # ── 1) 问模型 ───────────────────────────────────────────────
            seq += 1
            resp = self.llm.chat(messages)
            trace.steps.append(Step(
                seq=seq, kind=StepKind.LLM_CALL, payload=resp.to_payload(),
                source=getattr(self.llm, "current_source", "live"),
                tape_seq=getattr(self.llm, "current_tape_seq", None),
                meta=resp.to_meta(),
            ))

            if not resp.tool_call:
                seq += 1
                trace.steps.append(Step(
                    seq=seq, kind=StepKind.FINAL, payload={"text": resp.content or ""},
                    source=getattr(self.llm, "current_source", "live"),
                ))
                return trace

            # ── 2) 执行工具 ─────────────────────────────────────────────
            name = resp.tool_call.get("name") or ""
            args = resp.tool_call.get("arguments") or {}
            seq += 1
            payload = self._call_tool(name, args)
            trace.steps.append(Step(
                seq=seq, kind=StepKind.TOOL_CALL, payload=payload,
                source=getattr(self.tools, "current_source", "live"),
                tape_seq=getattr(self.tools, "current_tape_seq", None),
            ))

            # ── 3) 结果回灌 ─────────────────────────────────────────────
            body = payload.get("value") if payload.get("ok") else payload.get("error")
            messages.append({"role": "assistant", "content": resp.content or f"调用 {name}"})
            messages.append({
                "role": "user",
                "content": f"工具 {name} 返回：\n```json\n"
                           f"{json.dumps(body, ensure_ascii=False, default=str)[:2000]}\n```",
            })

        # 步数耗尽仍未收敛 —— 这本身是一种要能被比较的结局
        seq += 1
        trace.steps.append(Step(
            seq=seq, kind=StepKind.ERROR,
            payload={"type": "StepLimitExceeded",
                     "message": f"达到最大步数 {self.max_steps} 仍未给出结论"},
            source="live",
        ))
        trace.notes.append("未收敛")
        return trace

    def _call_tool(self, name: str, args: dict) -> dict:
        fn = self.tools.get(name) if isinstance(self.tools, dict) else None
        if fn is None:
            known = sorted(self.tools) if isinstance(self.tools, dict) else []
            return {"name": name, "arguments": args, "ok": False,
                    "error": f"未知工具 {name}；已注册：{known}"}
        try:
            value = fn(**args)
        except Exception as exc:  # noqa: BLE001 —— 工具故障要如实入轨迹
            # 如果包装层（录制器 / 回放器）已经给出了权威 payload，就用它的。
            # **不要在 agent 这边重建一份** —— 重建会产生与录制带不一致的
            # error 字符串，于是重放时凭空多出一处假分叉。
            given = getattr(exc, "replayprobe_payload", None)
            if isinstance(given, dict):
                return given
            return {"name": name, "arguments": args, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        return {"name": name, "arguments": args, "ok": True, "value": value}
