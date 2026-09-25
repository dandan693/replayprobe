"""真实模型连通性探针：一次调用，把**原始返回**原样打出来。

**为什么要有这个小工具。**

接真实模型时最常见的失败不是"连不上"，而是**连上了但协议对不上**：
模型把工具调用写成了纯文本、返回了 `finish_reason=length` 被截断、
或者干脆不用工具直接编了个数字。这些情况在完整链路里会表现为
"轨迹有点怪"，很难归因；而在一次裸调用里，一眼就能看清。

所以：**先用它确认协议，再动主链路。** 它只发一次请求，成本可以忽略。

用法：

    python tools/probe_real_llm.py --api-key-file "C:/path/to/key.txt"
    python tools/probe_real_llm.py --model qwen-plus --temperature 0

Key 的读取顺序：`--api-key-file` → 环境变量 `REPLAYPROBE_API_KEY`。
**Key 永远不会被打印出来**（只回显长度和首尾各 4 位）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replayprobe.agent import make_tools, openai_tool_schemas  # noqa: E402
from replayprobe.agent.core import SYSTEM_PROMPT  # noqa: E402
from replayprobe.llm import TASKS, OpenAIChatLLM, load_api_key  # noqa: E402

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _mask(key: str) -> str:
    if len(key) < 12:
        return f"<长度 {len(key)}>"
    return f"{key[:8]}…{key[-4:]}（长度 {len(key)}）"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真实模型连通性探针（单次调用）")
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--task", default="total", choices=sorted(TASKS))
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)

    key, base_url = load_api_key(args.api_key_file)
    if not key:
        print("没拿到 Key。用 --api-key-file 指定，或设环境变量 REPLAYPROBE_API_KEY。",
              file=sys.stderr)
        return 2
    base_url = args.base_url or base_url or DEFAULT_BASE_URL
    print(f"Key     {_mask(key)}")
    print(f"端点    {base_url}")
    print(f"模型    {args.model}  temperature={args.temperature}")
    print("-" * 62)

    db = Path(args.db) if args.db else ROOT / "data/truth/retail_truth.db"
    if not db.exists():
        print(f"真值库不存在：{db}\n先跑：python tools/build_truth_db.py --download",
              file=sys.stderr)
        return 2
    tools = make_tools(db)
    schemas = openai_tool_schemas(tools)

    print(f"工具 schema（{len(schemas)} 个）：")
    for s in schemas:
        fn = s["function"]
        params = ", ".join(fn["parameters"]["properties"]) or "无"
        print(f"  {fn['name']}({params})")

    question = TASKS[args.task]["question"]
    llm = OpenAIChatLLM(key, base_url, args.model,
                        temperature=args.temperature,
                        system_prompt=SYSTEM_PROMPT,
                        tool_schemas=schemas)

    print("-" * 62)
    print(f"问题    {question}")
    resp = llm.chat([{"role": "user", "content": question}])

    print("-" * 62)
    print("原始返回体检：")
    print(f"  model        {resp.model!r}")
    print(f"  finish       {resp.finish!r}")
    print(f"  finish_reason {resp.finish_reason!r}   （供应商原始值，只进 meta 不进签名）")
    print(f"  content      {len(resp.content)} 字")
    print(f"  tool_call    {json.dumps(resp.tool_call, ensure_ascii=False)}")
    print(f"  usage        {json.dumps(resp.usage, ensure_ascii=False)}")
    print("-" * 62)
    print("content 原文：")
    print(resp.content or "<空>")

    print("-" * 62)
    ok = True
    if resp.finish == "length":
        print("[!] 响应被长度截断 —— 这条轨迹不可用，换更短的 prompt 或加大 max_tokens")
        ok = False
    if not resp.tool_call:
        print("[!] 模型没有调用工具。要么它直接答了（对数据任务是坏事），")
        print("    要么工具调用写成了纯文本而没被解析出来。看上面的 content 原文。")
        ok = False
    else:
        tc = resp.tool_call
        fn = tools.get(tc.get("name"))
        if fn is None:
            print(f"[!] 模型调了不存在的工具 {tc.get('name')!r}")
            ok = False
        else:
            try:
                value = fn(**(tc.get("arguments") or {}))
                preview = json.dumps(value, ensure_ascii=False, default=str)[:200]
                print(f"[ok] 工具 {tc['name']} 实跑成功：{preview}")
            except Exception as exc:  # noqa: BLE001 —— 参数错了也要如实报
                print(f"[!] 工具 {tc['name']} 实跑失败：{type(exc).__name__}: {exc}")
                ok = False

    print("-" * 62)
    print("探针结论：" + ("协议可用，可以录真实轨迹了。" if ok else "先修上面的问题，别急着录。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
