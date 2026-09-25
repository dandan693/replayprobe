"""数据契约：录制带、轨迹、步、分叉。

这三张结构是整个框架的"事实来源"。对齐、分级、预算、报告全部读它，
所以字段定下来就要克制改动 —— **报告里的每一个数字都必须能回溯到某一条 Trace**。

写在这里的三条纪律（都是从现有工程实践里学来的硬经验，免得后来忘）：

1. **排序绝不用时间戳。** 并行工具完成的顺序可能和发起的顺序不同。
   所以每一步带一个单调递增的 `seq`，以及指向因果父节点的 `parent_seq`。
   按时间戳排序会得到一份"看起来对"的轨迹 —— 这种错最难查，
   因为它不报错，只是把因果讲反了。

2. **轨迹里只放客观发生了的事。** 不放任何"我认为它想干什么"的推断。
   推断住在 `signature.py` 和 `diverge.py` 里，这样推理错了只改一处。

3. **每个输入输出都存 canonical hash。** 用途有二：
   (a) 检测录制带被意外改动（重放对不上时，先排除"带本身被动过"）；
   (b) 内容寻址的接口预留 —— 同一个 prompt 重复出现时理论上只存一份，
       本项目暂未实现去重，但 hash 先留着。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class StepKind(str, enum.Enum):
    """轨迹里的一步是什么类型。"""

    LLM_CALL = "llm_call"
    """一次模型调用。这是**唯一真正的非确定性来源** —— 工具和数据库都是确定的。"""

    TOOL_CALL = "tool_call"
    """一次工具调用（执行 SQL、查元信息、算术）。"""

    FINAL = "final"
    """最终答复。对齐与分级的收口处。"""

    ERROR = "error"
    """崩溃 / 超时 / 未收敛。"""


class ReplayMode(str, enum.Enum):
    """回放的三种模式。

    这三者的区别不是"实现细节"，而是**三个不同的实验**。
    把它们混为一谈，是调试 Agent 时最容易骗到自己的地方：
    「重放一次看起来一样」不等于「行为被复现了」。
    """

    LIVE = "live"
    """真实执行，全程不打带 —— 这是**录制**时的模式。

    它和前三个不是并列关系，而是「录制」与「回放」的分界线：
    录制产物（带）是 live 跑出来的，回放才有 exact / drill / strict 三种模式。

    报告里必须显式区分 live 与 tape —— 拿 live 的结果去断言确定性是无效实验。
    """

    EXACT = "exact"
    """精确重放：模型和工具的结果**全部**取自录制带，不打网络。
    用来回答"这条轨迹本身是不是完好的"。跑 N 次必须 100% 一致。"""

    DRILL = "drill"
    """钻取重放（反事实）：前 N 步取自录制带，**第 N+1 步开始真实执行**。
    用来回答"如果第 N+1 步换了个做法，后面会怎样"。
    这是本项目真正的实验工具 —— 控制变量法的最小实现。"""

    STRICT = "strict"
    """严格重放：一切取自录制带，但每步都比对决策签名，对不上立刻中止并报出位置。
    用来当 CI 门禁 —— 它不关心结果，只关心"是不是同一条路"。"""


class Severity(str, enum.Enum):
    """分叉严重度。这是本项目的核心立场：**分叉不是一个布尔值**。

    D0 和 D1 是"没坏"，D2 要人看，D3/D4 是"坏了"。
    把 D1 和 D3 都记成"分叉了"，等于把噪音和信号混进同一个指标 ——
    结果就是门禁要么太松（天天过），要么太紧（天天红），最后没人看。
    """

    D0_IDENTICAL = "d0_identical"
    """完全一致：逐步的**完整签名**（含措辞）都相同。"""

    D1_WORDING = "d1_wording"
    """表述分叉：决策签名完全一致，但完整签名有差异。
    典型是措辞变了、SQL 的等价写法变了。**无害，且是常态** ——
    因为温度不为零时措辞本来就会变。这一档的存在是为了不误报。"""

    D2_PATH = "d2_path"
    """路径分叉：从某一步开始决策签名不同（换了工具 / 换了实质参数），
    但**最终结论一致**。需要人判断 —— 可能只是多绕了一步，也可能藏着问题。"""

    D3_CONCLUSION = "d3_conclusion"
    """结论分叉：最终答复的实质内容不同。这是"坏"。"""

    D4_SAFETY = "d4_safety"
    """安全分叉：一条拒答、另一条给出了具体答案；或一条触发了本该被拦下的动作。
    最严重 —— 它意味着**防护策略本身变了个样**，而不是结果算错了。"""


SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.D0_IDENTICAL,
    Severity.D1_WORDING,
    Severity.D2_PATH,
    Severity.D3_CONCLUSION,
    Severity.D4_SAFETY,
)
"""从轻到重。**顺序即语义** —— 预算比较、报告排序全靠它，不要随意调换。"""


def severity_rank(s: Severity | str) -> int:
    """取严重度的序号，越小越轻。用于预算比较与排序。"""
    if isinstance(s, str):
        s = Severity(s)
    return SEVERITY_ORDER.index(s)


# --------------------------------------------------------------------------- #
# 录制带
# --------------------------------------------------------------------------- #


@dataclass
class TapeEntry:
    """录制带里的一条记录 —— 相当于"标准答案"里的一行。

    一条 entry 只回答一件事：**当输入长这样时，当时的输出是那样。**
    它不解释为什么，也不判断对不对。
    """

    seq: int
    """单调序号。**不是时间戳**，见模块开头的纪律 1。"""

    kind: StepKind
    key: str
    """输入指纹。LLM 调用用 `模型+消息摘要`，工具调用用 `工具名+参数`。
    重放时靠它找回对应的输出。"""

    payload: dict
    """输出原文。LLM 调用存 content / tool_call，工具调用存 value / error。"""

    input_hash: str = ""
    output_hash: str = ""
    parent_seq: int | None = None
    """因果父节点。并行工具场景下，靠它才能还原"是谁触发了谁"。"""

    meta: dict = field(default_factory=dict)
    """补充信息：耗时、token 数、model id、prompt 版本 hash 等。
    **不放决策依据** —— 决策依据属于 Step，不属于带。"""

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind.value,
            "key": self.key,
            "payload": self.payload,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parent_seq": self.parent_seq,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(raw: dict) -> "TapeEntry":
        return TapeEntry(
            seq=int(raw["seq"]),
            kind=StepKind(raw["kind"]),
            key=raw["key"],
            payload=raw.get("payload") or {},
            input_hash=raw.get("input_hash", ""),
            output_hash=raw.get("output_hash", ""),
            parent_seq=raw.get("parent_seq"),
            meta=raw.get("meta") or {},
        )


@dataclass
class TapeManifest:
    """一份录制带的身份信息。

    **版本名不算身份。** 这是从调研里学到的第二条硬经验：
    日志里写 `model=gpt-x, prompt=repair-v4` 是弱证据 ——
    别名可能指向不同的权重，prompt 可能被就地改过。
    所以这里存的是**不可变指纹**：内容 hash。
    """

    tape_id: str
    task_id: str
    created_at: str = ""
    agent_variant: str = ""
    model: str = ""
    model_revision: str = ""
    prompt_hash: str = ""
    tool_schema_hash: str = ""
    dataset_snapshot: str = ""
    notes: str = ""
    limitations: list[str] = field(default_factory=list)
    """**凡不能做到不可变标识的东西，如实写在这里。**
    不要假装一个可变别名是稳定产物 —— 这是诚实与自我欺骗的分界线。"""

    def to_dict(self) -> dict:
        return {
            "tape_id": self.tape_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "agent_variant": self.agent_variant,
            "model": self.model,
            "model_revision": self.model_revision,
            "prompt_hash": self.prompt_hash,
            "tool_schema_hash": self.tool_schema_hash,
            "dataset_snapshot": self.dataset_snapshot,
            "notes": self.notes,
            "limitations": list(self.limitations),
        }

    @staticmethod
    def from_dict(raw: dict) -> "TapeManifest":
        return TapeManifest(
            **{k: v for k, v in raw.items() if k in TapeManifest.__annotations__}
        )


@dataclass
class Tape:
    """一份完整的录制带：身份 + 若干条记录。"""

    manifest: TapeManifest
    entries: list[TapeEntry] = field(default_factory=list)

    def by_seq(self) -> dict[int, TapeEntry]:
        return {e.seq: e for e in self.entries}

    def of_kind(self, kind: StepKind) -> list[TapeEntry]:
        return [e for e in self.entries if e.kind is kind]

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
        }


# --------------------------------------------------------------------------- #
# 轨迹
# --------------------------------------------------------------------------- #


@dataclass
class Step:
    """实际跑出来的一步。可能取自录制带，也可能是真实执行产生的。"""

    seq: int
    kind: StepKind
    payload: dict

    source: str = "tape"
    """`tape` = 取自录制带（确定性）；`live` = 真实执行（非确定性）。
    报告里要显式区分 —— **拿 live 的结果去断言确定性，是无效实验**。"""

    tape_seq: int | None = None
    """取自带的哪一条。None 表示 live。"""

    parent_seq: int | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "kind": self.kind.value,
            "payload": self.payload,
            "source": self.source,
            "tape_seq": self.tape_seq,
            "parent_seq": self.parent_seq,
            "meta": self.meta,
        }


@dataclass
class Trace:
    """一条完整轨迹 = 一次运行的全部步。"""

    run_id: str
    task_id: str
    variant: str
    mode: ReplayMode
    steps: list[Step] = field(default_factory=list)

    fork_at: int | None = None
    """DRILL 模式下的切开点：`seq <= fork_at` 取自录制带，之后 live。"""

    manifest: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def transcript_steps(self) -> list[Step]:
        """去掉 ERROR 的步骤序列，用于对齐 —— 崩溃不适合参与逐步比对。"""
        return [s for s in self.steps if s.kind is not StepKind.ERROR]

    def final(self) -> Step | None:
        for s in reversed(self.steps):
            if s.kind is StepKind.FINAL:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "variant": self.variant,
            "mode": self.mode.value,
            "fork_at": self.fork_at,
            "manifest": self.manifest,
            "notes": list(self.notes),
            "steps": [s.to_dict() for s in self.steps],
        }


# --------------------------------------------------------------------------- #
# 对齐与分叉
# --------------------------------------------------------------------------- #


@dataclass
class AlignedPair:
    """对齐后的一对位置。任一侧可以为 None，表示"这侧多了一步"。"""

    a: Step | None
    b: Step | None

    def is_insertion(self) -> bool:
        """B 侧多出一步（A 没有对应）。典型场景：换了做法后多了一次重试。"""
        return self.a is None and self.b is not None

    def is_deletion(self) -> bool:
        return self.b is None and self.a is not None

    def is_pair(self) -> bool:
        return self.a is not None and self.b is not None


@dataclass
class Alignment:
    """两条轨迹的对齐结果。

    `pairs` 是逐步对应关系，`misaligned_steps` 是插入/删除的步数 ——
    这两个数字本身就是指标：**插入删除多，说明两条轨迹的"形状"变了**，
    而不只是"某个决策变了"。
    """

    pairs: list[AlignedPair] = field(default_factory=list)
    matched: int = 0
    """决策签名一致的步数。"""

    insertions: int = 0
    deletions: int = 0
    method: str = "decision-signature-lcs"
    """用了哪种对齐法。`seq-index` 是不对齐的朴素做法，留作对照实验。"""

    def shape_changed(self) -> bool:
        return self.insertions > 0 or self.deletions > 0


@dataclass
class Divergence:
    """一处分叉。"""

    index: int
    """在对齐序列中的位置（0 起）。"""

    a_seq: int | None
    b_seq: int | None
    a_summary: str
    b_summary: str
    kind: str
    """分叉发生在哪一步的类型上。"""

    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "a_seq": self.a_seq,
            "b_seq": self.b_seq,
            "a_summary": self.a_summary,
            "b_summary": self.b_summary,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass
class Verdict:
    """一次"基线 vs 重放"比较的完整结论。"""

    task_id: str
    baseline_variant: str
    replay_variant: str
    severity: Severity = Severity.D0_IDENTICAL

    first_divergence: Divergence | None = None
    divergences: list[Divergence] = field(default_factory=list)
    alignment: Alignment = field(default_factory=Alignment)

    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "baseline_variant": self.baseline_variant,
            "replay_variant": self.replay_variant,
            "severity": self.severity.value,
            "first_divergence": self.first_divergence.to_dict() if self.first_divergence else None,
            "divergences": [d.to_dict() for d in self.divergences],
            "alignment": {
                "matched": self.alignment.matched,
                "insertions": self.alignment.insertions,
                "deletions": self.alignment.deletions,
                "method": self.alignment.method,
            },
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }
