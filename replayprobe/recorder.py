"""录制器：把一次真实运行固化成录制带。

## 录制的是什么，不是什么

录的是**每一个决策接缝上的「输入 → 输出」**：模型收到什么消息、回了什么；
工具收到什么参数、返回了什么。

录的**不是**「Agent 的意图」。推断的东西不进带 ——
一旦把推断录进去，推断错了就永远查不出来，因为带会说「当初就是这么决策的」。

## 为什么必须记录不可变指纹

带里存了 `model` / `prompt_hash` / `tool_schema_hash`。这不是形式主义：

**如果工具实现换了，旧带就不再可比。** 但如果没有指纹，这种不可比是**静默的** ——
你会拿着一份对不上的标准答案去判分，然后奇怪为什么全是红的。
所以 `TapeManifest.limitations` 这个字段存在，就是为了如实记录
「哪些东西我没法保证不可变」。
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Callable

from .llm import LLM, LLMResponse
from .signature import content_hash, normalize_arg
from .types import StepKind, Tape, TapeEntry, TapeManifest


def llm_key(messages: list[dict]) -> str:
    """模型调用的输入指纹。

    **用规范化后的参数算 key，是有意为之**：key 的职责是「找回对应的记录」，
    不是「判断有没有分叉」。分叉判断由签名层做，那里保留全部原始信息。
    两件事用同一个指纹，会让「顺手改了个空格」既找不到记录、又报不出分叉，
    是最糟的组合。
    """
    return "llm:" + content_hash(normalize_arg(messages))


def tool_key(name: str, arguments: dict) -> str:
    return f"tool:{name}:" + content_hash(normalize_arg(arguments))


class _WrappedLLM:
    """录制期的 LLM 外壳。它只做一件事：透传 + 落一条记录。"""

    current_source = "live"
    current_tape_seq: int | None = None

    def __init__(self, inner: LLM, recorder: "Recorder") -> None:
        self._inner = inner
        self._rec = recorder

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "")

    def chat(self, messages: list[dict]) -> LLMResponse:
        resp = self._inner.chat(messages)
        self._rec._record(
            StepKind.LLM_CALL,
            key=llm_key(messages),
            payload=resp.to_payload(),
            meta=resp.to_meta(),
            input_obj=messages,
        )
        return resp


class _WrappedTools(dict):
    """录制期的工具注册表。`dict` 子类，所以 `self.tools.get(...)` 照常用。"""

    current_source = "live"
    current_tape_seq: int | None = None

    def __init__(self, tools: dict[str, Callable[..., Any]], recorder: "Recorder") -> None:
        super().__init__()
        self._rec = recorder
        for name, fn in tools.items():
            self[name] = self._wrap(name, fn)

    def _wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(**kwargs: Any) -> Any:
            payload: dict[str, Any] = {"name": name, "arguments": kwargs}
            try:
                value = fn(**kwargs)
            except Exception as exc:  # noqa: BLE001 —— 失败也要入轨迹
                payload.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                self._rec._record(
                    StepKind.TOOL_CALL, key=tool_key(name, kwargs),
                    payload=payload, input_obj=kwargs,
                )
                # 把权威 payload 挂在异常上再原样抛出。
                #
                # **为什么不用 agent 那边的 try/except 去重建一份 payload。**
                # 重建过一次就多了一处可能不一致的地方：录制层会记成
                # `TypeError: xxx`，而 agent 可能重建成 `RuntimeError: xxx`，
                # 于是重放时这一条对不上，凭空多出一处假分叉。
                # 失败路径的一致性必须和成功路径一样有保证 ——
                # 而"两处各自构造同一份数据"是最容易悄悄漂移的写法。
                setattr(exc, "replayprobe_payload", payload)
                raise

            payload.update({"ok": True, "value": value})
            self._rec._record(
                StepKind.TOOL_CALL, key=tool_key(name, kwargs),
                payload=payload, input_obj=kwargs,
            )
            return value

        wrapper.__name__ = name
        return wrapper


class Recorder:
    """一次录制的状态机。用法：

        rec = Recorder(manifest)
        agent = ReActAgent(rec.wrap_llm(live_llm), rec.wrap_tools(tools))
        trace = agent.run("...")
        rec.finish().save("data/tapes/xxx.json")
    """

    def __init__(self, manifest: TapeManifest) -> None:
        self.manifest = manifest
        self.entries: list[TapeEntry] = []
        self._seq = 0

    def wrap_llm(self, live: LLM) -> LLM:
        self.manifest.model = self.manifest.model or getattr(live, "model", "")
        return _WrappedLLM(live, self)  # type: ignore[return-value]

    def wrap_tools(self, tools: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        return _WrappedTools(tools, self)

    def _record(self, kind: StepKind, *, key: str, payload: dict,
                meta: dict | None = None, input_obj: Any = None) -> TapeEntry:
        self._seq += 1
        entry = TapeEntry(
            seq=self._seq, kind=kind, key=key, payload=payload,
            input_hash=content_hash(normalize_arg(input_obj)) if input_obj is not None else "",
            output_hash=content_hash(payload),
            meta=meta or {},
        )
        self.entries.append(entry)
        return entry

    def finish(self) -> Tape:
        self.manifest.tape_id = self.manifest.tape_id or f"tape-{self._seq:04d}"
        self.manifest.created_at = self.manifest.created_at or \
            _dt.datetime.now().isoformat(timespec="seconds")
        if not self.manifest.tool_schema_hash:
            self.manifest.limitations.append(
                "未记录工具定义指纹 —— 工具实现变更后本条带不再可比"
            )
        if not self.manifest.model_revision:
            self.manifest.limitations.append(
                "只记录了模型别名，未记录具体服务版本 —— 别名可能指向不同权重"
            )
        return Tape(manifest=self.manifest, entries=list(self.entries))
