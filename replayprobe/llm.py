"""模型接入层：一个协议，两个实现。

- `ScriptedLLM`：**确定性**的脚本替身，让整个框架在**没有网络、没有 Key** 的情况下
  也能完整跑一遍。这不是玩具 —— 它是 CI 里唯一能跑的东西，
  也是所有单元测试的基础。一个"需要联网才能验证自己是对的"的框架，
  没人会在 PR 上跑它。
- `OpenAIChatLLM`：真实 API（OpenAI 兼容端点），用来录制真实轨迹。

**回放时那个「从录制带取答案」的实现不在这里** —— 它不是模型，
是记录的重放，住在 `player.py`。混在一起会让人以为重放也要调模型。

## 脚本替身的一个设计要点

`ScriptedLLM` 的分支依据是**第几次调用**（call_index），不是输入内容的 hash。
这是刻意的：真实模型的非确定性恰恰**不能**用输入 hash 模拟出来 ——
输入相同、输出不同，才是我们要研究的现象。用 call_index 分叉，
能精确控制"从第几步开始不一样"，这正是分叉诊断实验需要的。

三个内置变体对应三种真实的改动：

| 变体 | 模拟的改动 | 分叉位置 |
|---|---|---|
| `baseline` | 什么都没改 | — |
| `late_fork` | 模型变得更谨慎，多查一步（新探索） | 第 2 步起 |
| `scope_confusion` | 换了个更弱的模型，口径理解错 | 第 1 步起 |
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class LLMResponse:
    """一次模型调用的返回。字段与 `Step.payload`（LLM_CALL）严格对应。"""

    content: str = ""
    tool_call: dict | None = None
    """`{"name": str, "arguments": dict}`。None 表示这轮不给工具调用。"""

    finish: str = "stop"
    """"stop" 或 "tool_use"。"""

    model: str = ""
    """**注意：`model` 和 `usage` 属于元信息，不进 `to_payload()`。**

    第一版把它们塞进了 payload，自检立刻报错：`late_fork` 与 `baseline`
    的第 1 步明明都是同一次调用，却因为 `model` 字段一个是
    `scripted-baseline`、一个是 `scripted-late_fork` 而被判成不同。

    这在真实场景里是个坑：模型名常带日期后缀（`qwen-plus-2026-09-25`），
    升个版本号就会让所有 `full_signature` 失效，于是**满屏 D1**，
    而行为其实一模一样。**元信息不该参与内容比较。**
    """

    usage: dict = field(default_factory=dict)
    """与 model 同理，只进 meta。"""

    finish_reason: str = ""
    """供应商返回的**原始** `finish_reason`（`stop` / `length` / `tool_calls` …）。

    它只进 meta，不进 payload。理由是同一个语义在不同供应商那里取值不同 ——
    有的是 `tool_calls`、有的是 `stop` 加一个非空 tool_calls 数组。
    把它塞进 payload 会让「同一件事」因供应商不同而签名不同，**又是满屏假分叉**。

    但它必须被记下来：`finish_reason == "length"` 意味着响应被截断，
    这样的轨迹根本不该进入比较 —— 这是"数据采集阶段的已知坏样本"，
    要能被事后筛掉，而不是混进统计里污染结论。
    """

    def to_payload(self) -> dict:
        """内容部分。**只有这部分参与签名比较。**"""
        return {
            "content": self.content,
            "tool_call": self.tool_call,
            "finish": self.finish,
        }

    def to_meta(self) -> dict:
        """元信息部分。进录制带的 meta，供追溯用，不参与比较。"""
        meta = {"model": self.model, "usage": dict(self.usage)}
        if self.finish_reason:
            meta["finish_reason"] = self.finish_reason
        return meta


class LLM(Protocol):
    def chat(self, messages: list[dict]) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# 确定性脚本替身
# --------------------------------------------------------------------------- #


class ScriptedLLM:
    """按 call_index 分叉的确定性替身。

    同一个实例跑两次，逐步完全一致 —— 这是 `exact` 重放能自证的前提。
    """

    def __init__(self, policy: Callable[[int, list[dict]], LLMResponse],
                 model: str = "scripted-v1") -> None:
        self._policy = policy
        self.model = model
        self.calls = 0

    def chat(self, messages: list[dict]) -> LLMResponse:
        self.calls += 1
        resp = self._policy(self.calls, messages)
        resp.model = self.model
        return resp

    def reset(self) -> None:
        self.calls = 0


def _act(name: str, arguments: dict, content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content or f"调用 {name}",
        tool_call={"name": name, "arguments": arguments},
        finish="tool_use",
    )


def _answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_call=None, finish="stop")


TOTAL_SQL = "SELECT ROUND(SUM(Amount), 2) AS total FROM retail"
UK_SQL = "SELECT ROUND(SUM(Amount), 2) AS total FROM retail WHERE Country = 'United Kingdom'"
QTY_SQL = "SELECT ROUND(SUM(Quantity), 0) AS units FROM retail"

TASKS: dict[str, dict] = {
    "total": {
        "question": "全量总销售额是多少美元？",
        "sql": TOTAL_SQL,
        "answer": "全量总销售额为 8,887,208.89 美元，共 18,532 笔订单。",
        "wrong_sql": UK_SQL,
        "wrong_answer": "总销售额为 7,285,024.64 美元。",
        "cross_check_sql": QTY_SQL,
    },
    "orders": {
        "question": "一共有多少笔订单？",
        "sql": "SELECT COUNT(DISTINCT InvoiceNo) AS orders FROM retail",
        "answer": "一共有 18,532 笔订单。",
        # 把「行数」当成「订单数」—— 这是真实世界里最常犯的口径错误之一：
        # 392,692 行里有大量同一订单的多条明细。
        "wrong_sql": "SELECT COUNT(*) AS rows FROM retail",
        "wrong_answer": "一共有 392,692 笔订单。",
        "cross_check_sql": "SELECT COUNT(DISTINCT CustomerID) AS customers FROM retail",
    },
    "customers": {
        "question": "一共有多少位客户？",
        "sql": "SELECT COUNT(DISTINCT CustomerID) AS customers FROM retail",
        "answer": "一共有 4,338 位客户。",
        # 忘了 DISTINCT —— 于是把「记录条数」当成了「客户人数」
        "wrong_sql": "SELECT COUNT(CustomerID) AS customers FROM retail",
        "wrong_answer": "一共有 392,692 位客户。",
        "cross_check_sql": "SELECT COUNT(DISTINCT InvoiceNo) AS orders FROM retail",
    },
}
"""三个任务。**所有数字都来自真值库实跑，没有一个是编的。**

`wrong_sql` / `wrong_answer` 不是随便写的错 —— 它们对应两类真实高频的口径错误
（把行数当订单数、忘了 DISTINCT）。这样演示里的「结论分叉」才是可信的，
而不是一个没人会犯的假想错误。
"""

VARIANTS = ("baseline", "late_fork", "scope_confusion")
"""三个行为变体。**名字即用途**，报告里会原样回显。

- `baseline`      一轮取数后直接给结论
- `late_fork`     第 1 轮相同，第 2 轮多核对一次（新探索）
- `scope_confusion` 第 1 轮就用错口径 —— 典型的分叉最早点
"""


def _turn_of(messages: list[dict], idx: int) -> int:
    """从消息长度推出「这是第几轮对话」。

    **为什么不能直接用调用计数。**

    钻取回放时，切到真实执行的那一次调用，计数器是从 0 开始的 ——
    但它在上下文里已经是第 N 轮了。如果脚本替身只看计数器，
    切开后它会吐出一个跟上下文无关的答案（比如又调一次第一步查过的 SQL），
    结果 drill 实验看起来「什么都没变」，**而这是假的**。

    真实模型只看 messages，所以替身也必须只看 messages。
    消息布局：[user] + 每轮 [assistant, user] 两跳。
    """
    if not messages:
        return idx
    return max(1, (len(messages) - 1) // 2 + 1)


def make_policy(variant: str, task: str = "total") -> Callable[[int, list[dict]], LLMResponse]:
    """造一个脚本策略。`task` 决定查什么，`variant` 决定行为怎么变。"""
    if variant not in VARIANTS:
        raise ValueError(f"未知变体 {variant}，可选 {VARIANTS}")
    if task not in TASKS:
        raise ValueError(f"未知任务 {task}，可选 {tuple(TASKS)}")
    spec = TASKS[task]

    def baseline(n: int, messages: list[dict]) -> LLMResponse:
        if _turn_of(messages, n) == 1:
            return _act("run_sql", {"sql": spec["sql"]}, "我先查一下。")
        return _answer(spec["answer"])

    def late_fork(n: int, messages: list[dict]) -> LLMResponse:
        # 第 1 轮与 baseline 完全一致，第 2 轮开始多绕一次（新探索）
        turn = _turn_of(messages, n)
        if turn == 1:
            return _act("run_sql", {"sql": spec["sql"]}, "我先查一下。")
        if turn == 2:
            return _act("run_sql", {"sql": spec["cross_check_sql"]}, "再交叉验证一下。")
        return _answer(spec["answer"])

    def scope_confusion(n: int, messages: list[dict]) -> LLMResponse:
        # 第 1 轮就分叉：口径理解错了
        if _turn_of(messages, n) == 1:
            return _act("run_sql", {"sql": spec["wrong_sql"]}, "我来查一下。")
        return _answer(spec["wrong_answer"])

    return {"baseline": baseline, "late_fork": late_fork,
            "scope_confusion": scope_confusion}[variant]


def scripted(variant: str = "baseline", task: str = "total") -> ScriptedLLM:
    return ScriptedLLM(make_policy(variant, task), model=f"scripted-{variant}")


# --------------------------------------------------------------------------- #
# 真实 API（OpenAI 兼容）
# --------------------------------------------------------------------------- #


class OpenAIChatLLM:
    """OpenAI 兼容端点的最小客户端。只用标准库。

    只实现了「取一条回复」这一件事 —— 不是 SDK，也不打算是。
    录制真实轨迹时才需要它，其余场景用 `ScriptedLLM` 就够了。

    ## 工具调用解析：两条路都走

    - **原生**：请求里带 `tools`，从 `message.tool_calls` 读。真实模型首选这条。
    - **兜底**：从 content 里找 `<tool_call>{...}</tool_call>` 标签。

    两条都保留是刻意的：不同供应商、不同模型版本对 function calling 的
    支持程度不一样，而**协议不匹配的失败是静默的** ——
    模型明明调了工具，我们却只看到一段纯文本，于是要么把工具调用当成
    「任务完成」记进轨迹，要么整条轨迹失去意义。两条路都走，
    并且**两条都走不通时明确报错**，比赌一条要可靠。
    """

    def __init__(self, api_key: str, base_url: str, model: str,
                 temperature: float = 0.0, timeout: int = 60,
                 system_prompt: str = "",
                 tool_schemas: list[dict] | None = None,
                 max_tokens: int | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas or []
        self.max_tokens = max_tokens
        self.calls = 0
        """调用计数。**只用于排查，不用于决策** ——
        任何按「第几次调用」分叉的逻辑都必须住在 ScriptedLLM 里，
        真实调用拿到的是上下文，不是编号。"""

    def chat(self, messages: list[dict]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ([{"role": "system", "content": self.system_prompt}]
                         if self.system_prompt else []) + messages,
            "temperature": self.temperature,
        }
        if self.tool_schemas:
            payload["tools"] = self.tool_schemas
            payload["tool_choice"] = "auto"
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        self.calls += 1
        body = self._post(payload)
        return self._parse(body)

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"连不上 {self.base_url}：{exc.reason}") from exc

    def _parse(self, body: dict) -> LLMResponse:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or "stop"

        tool_call = _parse_native_tool_calls(message.get("tool_calls"))
        if tool_call is None:
            # 原生没给，再试标签协议。注意 `_parse_inline_tool_call`
            # 在遇到未闭合标签时会抛错 —— 这是刻意的，见它的 docstring。
            tool_call = _parse_inline_tool_call(content)

        return LLMResponse(
            content=content,
            tool_call=tool_call,
            finish="tool_use" if tool_call else "stop",
            model=body.get("model", self.model),
            usage=body.get("usage") or {},
            finish_reason=finish_reason,
        )


def _parse_native_tool_calls(raw: Any) -> dict | None:
    """从 OpenAI 风格的 `message.tool_calls` 里取**第一个**工具调用。

    只取第一个，是因为 `ReActAgent` 一次只执行一个工具。如果模型一次返回多个，
    这里**不静默丢弃**：多出来的会通过 `_extra_tool_calls` 计数反映在 meta 里，
    免得"模型想调两个、我们只跑了一个"这种偏差无声无息地进了轨迹。
    """
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0] or {}
    fn = first.get("function") or {}
    name = fn.get("name")
    if not name:
        return None
    args = fn.get("arguments")
    if isinstance(args, str):
        # 有的端点在参数不是合法 JSON 时会把原文塞回来 —— 如实报错，别猜
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具 {name} 的 arguments 不是合法 JSON：{exc}；原文={args[:200]!r}") from exc
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def count_tool_calls(raw: Any) -> int:
    """返回 `message.tool_calls` 的条数。用于记录「模型一次想调几个工具」。"""
    return len(raw) if isinstance(raw, list) else 0


def _parse_inline_tool_call(content: str) -> dict | None:
    """从纯文本回复里解析工具调用。

    这里刻意沿用 `nanocc` 那套 `<tool_call>{...}</tool_call>` 标签协议 ——
    不是因为它好，而是因为它**简单到不会失败**。
    它的已知缺陷（截断时闭合标签丢失，会被误判成「任务完成」）在
    `docs/真实案例对照.md` 里有完整记录，本项目在自己的解析里补了闭环检查。
    """
    import re

    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content or "", re.DOTALL)
    if not m:
        # 有开标签没闭合 → 明确报错，**绝不当作「任务完成」**
        if "<tool_call>" in (content or ""):
            raise ValueError("检测到未闭合的 <tool_call> 标签 —— 响应疑似被截断，拒绝解析")
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"<tool_call> 内的 JSON 无法解析：{exc}") from exc
    if not isinstance(data, dict) or "name" not in data:
        return None
    return {"name": data.get("name"), "arguments": data.get("arguments") or {}}


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_api_key(path: str | None = None) -> tuple[str, str]:
    """读取 API Key。返回 `(key, base_url_hint)`。

    只从文件或环境变量读，绝不把 Key 写进仓库 —— 这一点在 `.gitignore` 里也守了一道。

    **指定的路径不存在时直接报错，不静默回落。**
    第一版是「文件不存在就去看环境变量」，看起来更宽容，实际很坏：
    路径写错了一个字母，程序照样跑起来，用的是另一个 Key（或空 Key），
    于是你录了一整批轨迹才发现模型根本不是你想的那个。
    宽容在这里等于**把配置错误伪装成正常运行**。

    ## 为什么第二个返回值是空的（这是修过的）

    第一版在读到文件时**硬编码返回 dashscope 的端点**。结果是：
    只要用了 `--api-key-file`，配置文件里的 `base_url` 就被静默忽略，
    想换端点（自建代理、其他兼容服务）怎么改配置都没用。

    根因是把两件事绑在了一起：**读取凭据**和**决定端点**。
    一个 Key 文件不知道自己属于哪个端点 —— 那是调用方的事。
    所以这里返回的是「环境变量里有没有提示」，没有就交回调用方去决定。
    """
    import os
    from pathlib import Path

    hint = os.environ.get("REPLAYPROBE_BASE_URL", "").strip()
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"API Key 文件不存在：{p}")
        key = p.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"API Key 文件是空的：{p}")
        return key, hint
    return os.environ.get("REPLAYPROBE_API_KEY", "").strip(), hint


def make_real_llm(
    cfg: dict,
    *,
    tool_schemas: list[dict] | None = None,
    api_key_file: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    system_prompt: str = "",
    max_tokens: int | None = None,
) -> "OpenAIChatLLM":
    """按配置造一个真实模型客户端。优先级：命令行参数 > 配置文件 > 默认值。

    这个函数只负责**装配**，不负责**判断能不能用** ——
    "Key 缺失" 是配置问题（抛错），"模型不按协议回话" 是实验发现（录进轨迹）。
    把两者混在一起，会让配置错误看起来像模型行为，这正是本项目最想避免的混淆。
    """
    llm_cfg = (cfg or {}).get("llm") or {}
    key_file = api_key_file or llm_cfg.get("api_key_file") or None
    key, base_url = load_api_key(key_file)
    if not key:
        raise RuntimeError(
            "没有可用的 API Key：用 --api-key-file 指定文件，"
            "或设置环境变量 REPLAYPROBE_API_KEY。"
        )
    return OpenAIChatLLM(
        api_key=key,
        base_url=base_url or llm_cfg.get("base_url") or DEFAULT_BASE_URL,
        model=model or llm_cfg.get("model") or "qwen-plus",
        temperature=(llm_cfg.get("temperature", 0.0)
                     if temperature is None else temperature),
        timeout=int(llm_cfg.get("timeout", 60)),
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
        max_tokens=max_tokens,
    )


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #


def self_check() -> list[str]:
    problems: list[str] = []

    # ① 确定性：同一个变体跑两次必须逐步一致
    for variant in VARIANTS:
        a, b = scripted(variant), scripted(variant)
        for _ in range(4):
            ra, rb = a.chat([]), b.chat([])
            if ra.to_payload() != rb.to_payload():
                problems.append(f"变体 {variant} 不确定：同一位置两次输出不同")
                break

    # ② 变体之间必须在预期位置分叉
    base_steps, fork_steps, scope_steps = [], [], []
    for variant, bucket in (("baseline", base_steps), ("late_fork", fork_steps),
                            ("scope_confusion", scope_steps)):
        llm = scripted(variant)
        for _ in range(3):
            bucket.append(llm.chat([]).to_payload())

    if base_steps[0] == fork_steps[0] and base_steps[0] == scope_steps[0]:
        problems.append("三个变体的第 1 步居然完全相同 —— 分叉演示失去了意义")
    if base_steps[0] != fork_steps[0]:
        problems.append("late_fork 的第 1 步应当与 baseline 一致（它是晚期分叉）")

    # ③ 未闭合标签必须报错，不能被当成「任务完成」—— 这是抄来的教训
    try:
        _parse_inline_tool_call('好的，我来查。\n<tool_call>\n{"name": "run_sql"')
    except ValueError:
        pass
    else:
        problems.append("未闭合的 tool_call 标签没报错 —— 那正是 nanocc 踩过的坑")

    # ④ 元信息不得参与内容比较。
    # 第一版把 model 塞进了 payload，结果 late_fork 和 baseline 的第 1 步
    # 明明是同一件事，却因为模型名不同而被判成差异。真实场景里模型名
    # 常带日期后缀，升个版本号就会让 full_signature 集体失效、满屏 D1。
    p = scripted("baseline").chat([]).to_payload()
    if "model" in p or "usage" in p:
        problems.append("model / usage 混进了 payload —— 模型版本号一变就满屏假分叉")

    return problems
