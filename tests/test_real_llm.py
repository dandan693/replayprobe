"""真实模型接入层的测试：**全程不打网络**。

这里检验的全是"协议对不上"这类故障 —— 而它们的可怕之处在于**静默**：
模型明明调了工具，我们只看到一段纯文本，于是把这次调用当成"任务完成"
记进轨迹，整条轨迹看起来正常，其实什么都没做。

所以这一层每条测试都在问同一个问题：
**出错的时候，它是报错，还是装作没事？**
"""

from __future__ import annotations

import unittest

from replayprobe.agent import make_tools, openai_tool_schemas, tool_schema_hash
from replayprobe.llm import (
    LLMResponse,
    OpenAIChatLLM,
    count_tool_calls,
    load_api_key,
    make_real_llm,
)

DB = ":memory:"  # 只用来构造工具注册表，本文件不执行任何 SQL


def _llm(**kw) -> OpenAIChatLLM:
    return OpenAIChatLLM(api_key="sk-test", base_url="https://example.invalid/v1",
                         model="qwen-plus", **kw)


def _body(*, content="", tool_calls=None, finish_reason="stop", usage=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "model": "qwen-plus-2026-09-25",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _native(name="run_sql", arguments='{"sql": "SELECT SUM(Amount) FROM retail"}'):
    return [{"id": "call_1", "type": "function",
             "function": {"name": name, "arguments": arguments}}]


class TestToolSchemas(unittest.TestCase):
    def test_shape_is_openai_compatible(self):
        schemas = openai_tool_schemas(make_tools(DB))
        self.assertEqual({s["function"]["name"] for s in schemas},
                         {"run_sql", "get_schema", "calc"})
        for s in schemas:
            self.assertEqual(s["type"], "function")
            fn = s["function"]
            self.assertTrue(fn["description"], f"{fn['name']} 缺描述")
            self.assertEqual(fn["parameters"]["type"], "object")

    def test_param_types_are_mapped_from_annotations(self):
        by_name = {s["function"]["name"]: s["function"] for s in openai_tool_schemas(make_tools(DB))}
        # 三个工具的参数都是 str，映射必须是 string 而不是默认的 object
        for name, param in (("run_sql", "sql"), ("get_schema", "table"), ("calc", "expression")):
            prop = by_name[name]["parameters"]["properties"][param]
            self.assertEqual(prop["type"], "string", f"{name}.{param} 类型映射错了")
            self.assertTrue(prop["description"])

    def test_every_param_documented_in_code_is_used(self):
        """`PARAM_DOCS` 里的键必须真的对得上某个工具的某个参数。

        否则改一次参数名，描述就会静默消失 —— 模型照样跑，只是 SQL 质量变差，
        而没人会收到任何报错。
        """
        from replayprobe.agent.tools_builtin import PARAM_DOCS

        actual = set()
        for s in openai_tool_schemas(make_tools(DB)):
            fn = s["function"]
            for p in fn["parameters"]["properties"]:
                actual.add((fn["name"], p))
        self.assertEqual(set(PARAM_DOCS), actual,
                         "PARAM_DOCS 与实际参数集合不一致（改了参数名却忘了改描述？）")

    def test_nothing_is_required_since_all_params_have_defaults(self):
        for s in openai_tool_schemas(make_tools(DB)):
            self.assertEqual(s["function"]["parameters"]["required"], [])


class TestNativeToolCallParsing(unittest.TestCase):
    def test_native_tool_call_is_parsed(self):
        r = _llm()._parse(_body(tool_calls=_native(), finish_reason="tool_calls"))
        self.assertEqual(r.tool_call, {"name": "run_sql",
                                       "arguments": {"sql": "SELECT SUM(Amount) FROM retail"}})
        self.assertEqual(r.finish, "tool_use")

    def test_no_tool_call_means_stop(self):
        r = _llm()._parse(_body(content="答案是 42。"))
        self.assertIsNone(r.tool_call)
        self.assertEqual(r.finish, "stop")

    def test_arguments_as_dict_is_accepted(self):
        """有的端点直接给 dict 而不是 JSON 字符串。两条都要认。"""
        r = _llm()._parse(_body(tool_calls=[{"function": {
            "name": "calc", "arguments": {"expression": "1+1"}}}]))
        self.assertEqual(r.tool_call["arguments"], {"expression": "1+1"})

    def test_empty_arguments_string_becomes_empty_dict(self):
        r = _llm()._parse(_body(tool_calls=_native(arguments="")))
        self.assertEqual(r.tool_call["arguments"], {})

    def test_malformed_arguments_raises_instead_of_guessing(self):
        """参数不是合法 JSON 时必须报错。

        猜一个"看起来合理"的参数塞回去，会让轨迹记录下一次**根本没发生过的**
        工具调用 —— 而这比崩溃难查得多，因为它不报错。
        """
        with self.assertRaises(ValueError) as ctx:
            _llm()._parse(_body(tool_calls=_native(arguments="{sql: broken")))
        self.assertIn("JSON", str(ctx.exception))

    def test_unnamed_tool_call_is_ignored(self):
        self.assertIsNone(_llm()._parse(_body(tool_calls=[{"function": {}}])).tool_call)

    def test_count_tool_calls_reports_parallel_calls(self):
        """模型一次想调多个工具时，这个数字进 meta ——

        我们只执行第一个，但不能让这件事**无声无息**。
        """
        raw = _native() + _native(name="get_schema", arguments="{}")
        self.assertEqual(count_tool_calls(raw), 2)
        self.assertEqual(count_tool_calls(None), 0)

    def test_inline_tag_is_used_when_native_is_absent(self):
        """兜底路径：供应商不支持原生 function calling 时，标签协议仍要能走。"""
        content = '我来查。\n<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT 1"}}</tool_call>'
        r = _llm()._parse(_body(content=content))
        self.assertEqual(r.tool_call["name"], "run_sql")
        self.assertEqual(r.finish, "tool_use")

    def test_native_wins_over_inline(self):
        r = _llm()._parse(_body(
            content='<tool_call>{"name": "get_schema", "arguments": {}}</tool_call>',
            tool_calls=_native()))
        self.assertEqual(r.tool_call["name"], "run_sql")


class TestMetaNeverLeaksIntoContent(unittest.TestCase):
    """元信息不进 payload —— 否则模型版本号一变就满屏假分叉。"""

    def test_finish_reason_is_meta_only(self):
        r = _llm()._parse(_body(tool_calls=_native(), finish_reason="tool_calls"))
        self.assertEqual(r.finish_reason, "tool_calls")
        self.assertNotIn("finish_reason", r.to_payload())
        self.assertIn("finish_reason", r.to_meta())

    def test_model_and_usage_are_meta_only(self):
        r = _llm()._parse(_body(content="hi"))
        for k in ("model", "usage"):
            self.assertNotIn(k, r.to_payload())
            self.assertIn(k, r.to_meta())

    def test_same_content_different_provider_metadata_has_same_payload(self):
        """同一件事、两个供应商 → payload 必须相同。

        `finish_reason` 有的供应商给 `tool_calls`、有的给 `stop`，
        这正是它会污染签名的原因。
        """
        a = _llm()._parse(_body(tool_calls=_native(), finish_reason="tool_calls"))
        b = _llm()._parse(_body(tool_calls=_native(), finish_reason="stop"))
        self.assertEqual(a.to_payload(), b.to_payload())


class TestRequestPayload(unittest.TestCase):
    def _capture(self, llm: OpenAIChatLLM) -> dict:
        sent = {}
        llm._post = lambda payload: (sent.update(payload), _body(content="ok"))[1]
        llm.chat([{"role": "user", "content": "问题"}])
        return sent

    def test_tools_are_sent_when_provided(self):
        sent = self._capture(_llm(tool_schemas=openai_tool_schemas(make_tools(DB))))
        self.assertEqual(len(sent["tools"]), 3)
        self.assertEqual(sent["tool_choice"], "auto")

    def test_no_tools_key_when_none_provided(self):
        sent = self._capture(_llm())
        self.assertNotIn("tools", sent)
        self.assertNotIn("tool_choice", sent)

    def test_system_prompt_is_prepended_not_replacing(self):
        sent = self._capture(_llm(system_prompt="规则一"))
        self.assertEqual(sent["messages"][0],
                         {"role": "system", "content": "规则一"})
        self.assertEqual(sent["messages"][1]["role"], "user")

    def test_temperature_is_actually_sent(self):
        """这是一个真实踩过的坑：界面上配了温度，请求里却没带上，
        于是"温度没影响"这个结论是假的。"""
        self.assertEqual(self._capture(_llm(temperature=1.0))["temperature"], 1.0)


class TestApiKeyLoading(unittest.TestCase):
    def test_missing_file_raises_instead_of_falling_back(self):
        """指定的路径不存在时必须报错。

        第一版是"文件不存在就去看环境变量"，看起来更宽容，实际很坏：
        路径写错一个字母，程序照样跑，用的是另一个 Key（或空 Key），
        于是录了一整批轨迹才发现模型不是你想的那个。
        **宽容在这里等于把配置错误伪装成正常运行。**
        """
        with self.assertRaises(FileNotFoundError):
            load_api_key("C:/definitely/not/here/key.txt")

    def test_empty_file_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.txt"
            p.write_text("   \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_api_key(str(p))

    def test_missing_key_everywhere_raises_with_actionable_message(self):
        import os

        saved = os.environ.pop("REPLAYPROBE_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                make_real_llm({}, tool_schemas=[])
            self.assertIn("REPLAYPROBE_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["REPLAYPROBE_API_KEY"] = saved

    def test_cli_args_override_config(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.txt"
            p.write_text("sk-from-file", encoding="utf-8")
            cfg = {"llm": {"model": "from-config", "temperature": 0.3,
                           "base_url": "https://config.example/v1"}}
            llm = make_real_llm(cfg, tool_schemas=[], api_key_file=str(p),
                                model="from-arg", temperature=1.0)
            self.assertEqual(llm.api_key, "sk-from-file")
            self.assertEqual(llm.model, "from-arg")
            self.assertEqual(llm.temperature, 1.0)
            llm2 = make_real_llm(cfg, tool_schemas=[], api_key_file=str(p))
            self.assertEqual(llm2.model, "from-config")
            self.assertEqual(llm2.temperature, 0.3)
            self.assertEqual(llm2.base_url, "https://config.example/v1")

    def test_key_file_does_not_dictate_the_endpoint(self):
        """Key 文件不该决定端点 —— 那是调用方的事。

        第一版在读到文件时硬编码返回 dashscope 的端点，于是只要用了
        `--api-key-file`，配置里的 `base_url` 就被静默忽略，
        想换端点怎么改配置都没用。根因是把「读凭据」和「定端点」绑成了同一件事。
        """
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.txt"
            p.write_text("sk-x", encoding="utf-8")
            _, hint = load_api_key(str(p))
            self.assertEqual(hint, "", "读 Key 文件不该附带端点信息")

            saved = os.environ.pop("REPLAYPROBE_BASE_URL", None)
            try:
                # 文件里的 Key + 配置里的端点：端点必须来自配置
                llm = make_real_llm({"llm": {"base_url": "https://conf.example/v1"}},
                                    tool_schemas=[], api_key_file=str(p))
                self.assertEqual(llm.base_url, "https://conf.example/v1")
                # 都没给时退到 dashscope 兜底，而不是留空
                llm2 = make_real_llm({}, tool_schemas=[], api_key_file=str(p))
                self.assertEqual(llm2.base_url,
                                 "https://dashscope.aliyuncs.com/compatible-mode/v1")
            finally:
                if saved is not None:
                    os.environ["REPLAYPROBE_BASE_URL"] = saved

    def test_env_hint_is_the_second_element_only(self):
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.txt"
            p.write_text("sk-x", encoding="utf-8")
            saved = os.environ.get("REPLAYPROBE_BASE_URL")
            os.environ["REPLAYPROBE_BASE_URL"] = "https://env.example/v1"
            try:
                key, hint = load_api_key(str(p))
                self.assertEqual(key, "sk-x")
                self.assertEqual(hint, "https://env.example/v1")
            finally:
                if saved is None:
                    os.environ.pop("REPLAYPROBE_BASE_URL", None)
                else:
                    os.environ["REPLAYPROBE_BASE_URL"] = saved


class TestResponseDataclass(unittest.TestCase):
    def test_truncated_response_is_visible_in_meta(self):
        """`finish_reason == "length"` 的轨迹不该混进统计。

        它必须能被事后筛掉 —— 一条被截断的响应，其"内容"根本就不是
        模型想说的完整意思，拿它去判分等于拿半句话当结论。
        """
        r = LLMResponse(content="被截断的一", finish_reason="length")
        self.assertEqual(r.to_meta()["finish_reason"], "length")


class TestToolSchemaHashCoversDefaults(unittest.TestCase):
    def test_changing_a_default_changes_the_hash(self):
        def tool_a(sql: str = "") -> str:
            return sql

        def tool_b(sql: str = "SELECT 1") -> str:
            return sql

        self.assertNotEqual(tool_schema_hash({"t": tool_a}),
                            tool_schema_hash({"t": tool_b}),
                            "改了参数默认值却算出同一个指纹 —— 那是静默失配")


if __name__ == "__main__":
    unittest.main()
