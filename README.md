# replayprobe

**Agent 轨迹回放与分叉诊断。录一次，改一个变量，再放一遍。**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

![deps](https://img.shields.io/badge/dependencies-0-brightgreen)

![tests](https://img.shields.io/badge/tests-182%20passed-brightgreen)



![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 一句话说清它解决什么

Agent 上线之后，你会不断改东西：换模型、改提示词、升工具版本、调温度。

改完怎么看有没有改坏？

现在的做法基本是两种：**要么重跑一遍，凭印象看日志**（人眼只能看几条）；  
**要么跑评测集看分数**（分数没掉，就认为没问题）。

但这两种都漏掉了同一类问题：**分数没掉，路走歪了。**

举个真实场景。你把提示词里一句话改了一下，重跑 50 条任务，  
准确率从 46 掉到 45 —— 在噪声范围内，看起来没问题。但其中 12 条任务里，  
Agent 从"一次取数直接回答"变成了"先查表结构、再取数、再交叉验证、最后回答"。  
结论是对的，成本是原来的三倍，延迟翻了两番。**分数看不出来，人也不可能一条条看。**

replayprobe 就是为这一类问题做的：它不关心"答得对不对"（那是评测集的事），  
它只回答一个问题 ——

> **这次改动之后，Agent 走的还是同一条路吗？如果不是，是从哪一步开始不一样的？**

---

## 30 秒上手（零第三方依赖）

```bash
git clone https://github.com/dandan693/replayprobe && cd replayprobe
python tools/build_truth_db.py --download   # 自己取原始数据 + 建真值库（约 24 MB）
python -m replayprobe check                 # 先证明工具自己是好的
python tools/demo_replay.py                 # 五段式端到端演示，退出码 0
```

**没有任何第三方依赖。** 不装 pytest、不装 pandas、不引 CDN。  
这不是洁癖：**一个门禁如果 clone 之后跑不起来，就没有人会跑它；  
而一个没人跑的门禁，等于不存在。**

只有第一步需要联网，而且只下一次：UCI 现在给的是 xlsx（23.7 MB，装在  
`online+retail.zip` 里），脚本自己下载、自己解析（零依赖）、缓存进 `data/raw/`，  
之后重建真值库是纯本地的 30 秒。手头已经有数据就跳过下载：

```bash
python tools/build_truth_db.py --src "/path/to/Online Retail.xlsx"   # .xlsx 和 .csv 都吃
```

> **这一段原来是个坑，写在这里是因为它很典型。**  
> 上一版的「30 秒上手」第一步是 `python tools/build_truth_db.py`，而那个脚本  
> **只会找本地文件、不会下载** —— 新 clone 的人走到第一步必然失败。  
> 这正好犯了本仓库自己反复警告的那条：**文档承诺的和实际能跑的，不是同一件事。  
> 而一个跑不起来的第一步，会让后面所有"你可以自己验证"的承诺一起失效。**  
> 现在取数由 `replayprobe/dataset.py` 负责，并且"下载到的数据对不对"也一起管住了 ——  
> 后面的 **392,692 行 + 7 项口径自检**就是它的验证器。

**而且不联网也能重放一条真实模型的轨迹** —— 仓库里存了一条  
`qwen-plus-latest` 的录制带，可以直接逐字放出来：

```bash
python -m replayprobe run \
  --tape data/tapes/real_qwen-plus-latest_t0_total.tape.json \
  --mode exact --out reports/real_exact.json
#   模式 exact | 4 步 | 用带 3 步 / 真实执行 0 步

python -m replayprobe compare \
  --a data/tapes/real_qwen-plus-latest_t0_total.trace.json --b reports/real_exact.json
#   分叉判定：d0_identical
```

这条轨迹是怎么来的、以及它上面跑出来的全部实验，见  
[`docs/真实实验记录.md`](docs/真实实验记录.md)。

完整的命令行链路：

```bash
# ① 录制：真实跑一遍，把每个决策接缝固化成「带」
python -m replayprobe run --task total --variant baseline --out reports/base.json
#   录制带 reports/base.tape.json（3 条记录）
#   轨迹   reports/base.json（4 步）

# ② 精确重放：全程只用带，不打网络
python -m replayprobe run --tape reports/base.tape.json --mode exact --out reports/exact.json
#   模式 exact | 4 步 | 用带 3 步 / 真实执行 0 步

# ③ 钻取重放：前 2 步照旧，第 3 步起让模型自由发挥（反事实实验）
python -m replayprobe run --tape reports/base.tape.json --mode drill --fork-at 2 \
                          --live late_fork --task total --out reports/drill.json
#   模式 drill | 6 步 | 用带 2 步 / 真实执行 3 步

# ④ 比较两条轨迹，拿到分叉判定
python -m replayprobe compare --a reports/base.json --b reports/drill.json
#   分叉判定：d2_path
#     · 路径上多出 2 步，且其中包含基线里没有过的**新决策** —— 模型获取了不同的信息
#     已写出 reports/drill.verdict.json

# ⑤ 按预算结算（CI 用退出码）
python -m replayprobe gate --dir reports --budget default ; echo $?
#   预算 [default] 未通过（2 条违例）
#     任务 1 条 · 分叉率 100.0%（d2_path 及以上计入）· 匹配率 66.7%
#     分布 d2_path=1
#   退出码 1
#   （上面这一批的目录里只有 ①~③ 产出的三个文件，所以是「1 条任务」）

# ⑥ 出报告（零依赖手写 HTML/SVG）
python -m replayprobe report --dir reports
```

---

## 核心机制

### 1. 录制带（Tape）

录的是**每一个决策接缝上的「输入 → 输出」**：模型收到什么消息、回了什么；  
工具收到什么参数、返回了什么。

```
TapeManifest  身份 + 不可变指纹（model_revision / prompt_hash / tool_schema_hash）
TapeEntry     seq / kind / key / payload / input_hash / output_hash / parent_seq / meta
```

三条纪律写在 `types.py` 的模块注释里，都有出处：

- **排序绝不用时间戳。** 并行工具完成的顺序可能和发起顺序不同。按时间戳排序  
  会得到一份"看起来对"的轨迹 —— 这种错最难查，因为它不报错，只是把因果讲反了。  
  所以用单调 `seq` + 指向因果父节点的 `parent_seq`。
- **轨迹里只放客观发生了的事。** 不放任何"我认为它想干什么"的推断 ——  
  推断错了就永远查不出来，因为带会说"当初就是这么决策的"。
- **每个输入输出都存 canonical hash。** 用来先排除"带本身被动过"。

**版本名不算身份。** 日志里写 `model=gpt-x, prompt=repair-v4` 是弱证据 ——  
别名可能指向不同权重，prompt 可能被就地改过。所以存的是内容 hash。  
凡不能保证不可变的，如实写进 `TapeManifest.limitations`。

### 2. 决策签名（Signature）：本项目的技术支点

**把模型的自由文本压成可比较的规范化指纹。全程纯字符串处理，零模型调用。**

三级分辨率，是**严格的粗细阶梯**：

| 级别         | 比什么                      | 相同意味着            |
| ---------- | ------------------------ | ---------------- |
| `full`     | 整个 payload，含措辞           | 连字都一样 → D0       |
| `decision` | 行动 + **工具名** + 规范化参数     | 决策一样，只是说法不同 → D1 |
| `intent`   | 只比动作形状（做题 / 作答 / 拒答 / 崩） | 换的是手段，不是方向       |

规范化强度是一条刻度，不是一个开关。**本项目的取舍：只做表面规范化  
（空白 / 大小写 / 标点 / 末尾分号），不做语义规范化。**  
理由是：表面规范化是确定性的且**可解释**（能说出改了哪几个字符），  
而语义规范化必须引入 SQL 解析器（重）或模型判断（不可复现）。

> **宁可误报，也不引入不可复现的判据。** 因为本项目的存在意义就是"可复现"，  
> 用一个不可复现的判据去判它，自相矛盾。而且门禁的可信度是一次性的 ——  
> 人们不会信任一个"有时候会误判"的门禁。

### 3. 序列对齐：为什么不能按序号对

改一次提示词，最常见的后果**不是**"某一步做错了"，而是"**某一步多做了一次**"。  
一旦中间多出一步，后面所有步号就整体错位。

**这个错误方向特别恶劣：它看起来像真的发现了问题。**  
报告里长长一串分叉，没人会怀疑是工具算错了 ——  
他们只会得出"这次改动影响很大"，然后动手去改那些根本没坏的地方。

所以改用**基于决策签名的序列对齐**（`difflib.SequenceMatcher`，标准库），  
并**刻意保留**朴素实现 `align_by_index` 作对照组：

```
序号对齐（朴素做法）报出 2 处决策分叉
签名对齐（本项目）  报出 0 处决策分叉

真实情况：只有 1 处新决策，其余"分叉"全是多了一步之后整体错位造成的假象。
```

这个数字是 `python tools/demo_replay.py` 里第 ③b 段实跑出来的，不是编的。

### 4. 五档分叉量表：分叉不是一个布尔值

如果你只回答"分叉了 / 没分叉"，会立刻遇到两难：阈值调松 → 温度不为零时  
措辞本来就会变，于是**什么都算分叉**，门禁天天红，两周内团队就学会加 `--skip`；  
调紧 → 真的换了工具、真的算错了也可能被抹平。

**这不是调参问题，是模型缺了维度。** 所以拆成五档：

| 档      | 含义                   | 该不该拦 CI             |
| ------ | -------------------- | ------------------- |
| **D0** | 连措辞都一样               | 不该                  |
| **D1** | 措辞变了，决策没变            | **不该** —— 这是常态，不是问题 |
| **D2** | 路径变了，结论没变            | 看预算                 |
| **D3** | 结论变了                 | 该拦                  |
| **D4** | 结局性质变了（拒答↔给答案、正常↔崩溃） | **必须拦**             |

两个容易被做错的地方，这里专门处理了：

**① 「多答了一个数」不等于「结论被推翻」。** 用集合运算替代语义判断：

```
{8887208.89}  vs  {8887208.89, 18532}   →  原数都还在，只是信息量变了 → D2
{8887208.89}  vs  {4443606.45}          →  原来的数没了，被替换了     → D3
```

**② 「重试」不等于「新探索」。** 多调了一次同样的工具 → D1（常态）；  
调了一个基线里从没出现过的工具 → D2（真的换了获取信息的路径）。  
不做这个区分，改一次提示词后模型偶然多试一次 SQL 就会淹没在 D2 里 ——  
**那是这个量表最容易死掉的方式。**

### 5. 分叉预算：把"允许分叉多少"变成 CI 能判的退出码

现有的观测工具（Langfuse / Phoenix / LangSmith / Braintrust）都能让你**看**  
两条轨迹的差异。但看完之后呢？差异是"可接受"还是"不可接受"，  
**只能靠人看**。于是它永远停在调试阶段，进不了流水线 ——  
因为流水线需要的是一个**能返回退出码的判断**。

这个项目补的就是这一环：

```json
{
  "name": "prompt_tweak",
  "max_severity": "d2_path",
  "divergence_floor": "d2_path",
  "allow_fork_after": 2,
  "max_divergence_ratio": 0.3,
  "min_matched_ratio": 0.5,
  "forbid_crash": true,
  "forbid_safety_fork": true
}
```

叫「预算」是因为它和"允许花多少钱"是同一种东西：**有额度**（D1 无限、D2 少量、  
D3/D4 零）、**可耗尽**、**必须显式声明**。

注意额度是**按批**算的：单条任务 D2 很正常，但一批 50 条里 30 条都是 D2，  
那就是改动本身有问题 —— 所以比率类配置必须存在。

`divergence_floor` 是**后来补的**，补的原因很能说明问题。`model_swap` 预算  
原本写的是「路径允许大改」（上限 D3）+「分叉率上限 50%」。拿真实模型一跑，  
两组比较全是 D2（SQL 写法不同、结论一模一样），分叉率 100% → 被判违规。

问题不在阈值，在于**两个旋钮在互相打架**：`max_severity` 说"D2 允许"，  
分叉率却把 D2 算作违规。于是这份预算永远无法通过 —— 而它恰恰是  
"换便宜模型值不值得"这个问题唯一的可执行表达。

修法不是把 0.5 调成 1.0（那叫把门禁调绿灯），而是**把混在一起的两个概念拆开**：

- `max_severity` 管「最坏那一条能坏到什么程度」
- `divergence_floor` 管「从哪一档起算『它被碰到了』」—— 这是**计数口径**

据此，`Budget.audit()` 会拦下 `floor` 低于 `max_severity` 的配置：  
**被明确允许的等级，不该又被算进违规率。**  
一个自相矛盾的预算必须当场报错，不能默默执行 —— 它会以"测试通过"的形式藏起来。

### 6. 三种回放模式，三个不同的实验

混在一起是调试 Agent 时最容易骗到自己的地方：  
「重放一次看起来一样」**不等于**「行为被复现了」。

| 模式       | 带的使用            | 遇到带里没有的请求           | 回答什么问题               |
| -------- | --------------- | ------------------- | -------------------- |
| `exact`  | 全程用带            | **抛异常**             | 这条轨迹本身完好、且可复现吗？      |
| `strict` | 优先用带            | **调 live 并记录偏离点**   | 新版本从第几步开始走岔了？        |
| `drill`  | 前 `fork_at` 步用带 | 过了切开点后**主动**全走 live | 如果第 N+1 步换个做法，后面会怎样？ |

`strict` 和 `drill` 的区别值得说清：**strict 是我不知道会不会偏、让它自己偏；  
drill 是我指定从哪儿切。** 前者用于回归检测，后者用于反事实实验。

### 7. 一条铁律：绝不为带里没有的请求即兴编一个成功

这是回放系统最容易犯、后果最隐蔽的错。一旦开始即兴，重放就会**一路绿灯地跑完**，  
然后告诉你"一切正常" —— 而它其实已经偏离了原轨迹，后面所有比较都失去了参照。

**一个会自己编答案的裁判，比没有裁判更危险。**

```
exact  -> 如实抛错：带里没有对应的 llm_call 记录（key=llm:02d7682fc6cb0825…）
strict -> 记录 1 处偏离并用真实执行兜底，跑完 4 步
```

---

## 接真实模型：实测跑出来的三个结论

脚本替身（`ScriptedLLM`）是 CI 的基础，但**真实模型才是要诊断的对象**。  
接法是原生 function calling，不依赖标签协议：

```bash
# ① 先验协议：一次裸调用，把原始返回打出来（成本可忽略）
python tools/probe_real_llm.py --api-key-file /path/to/key.txt --model qwen-plus-latest

# ② 录制真实轨迹
python -m replayprobe run --task total --llm real --model qwen-plus-latest \
    --temperature 0 --api-key-file /path/to/key.txt --out reports/t0_1.json

# ③ 改一个变量再放一遍（这里改的是 system prompt）
python -m replayprobe run --task total --llm real --model qwen-plus-latest --temperature 0 \
    --api-key-file /path/to/key.txt \
    --system-suffix "4. 给出结论前，必须先调用 get_schema 确认表结构。" \
    --out reports/prompt_plus.json

# ④ 按实验类型分别结算（--select 按"改动了哪个变量"筛选）
python -m replayprobe gate --dir reports --budget default      --select none
python -m replayprobe gate --dir reports --budget prompt_tweak --select prompt_hash
python -m replayprobe gate --dir reports --budget model_swap   --select model
```

Key 只从文件或环境变量读，绝不进仓库；带里也不会出现 Key。  
真实模型录出来的带会自动声明三条「不可保证项」（模型别名可能换权重、  
temperature=0 不等于确定性、上游可能有缓存）。

### 实测结果（qwen-plus-latest / qwen-turbo，2026-09-25）

同一份代码、同一个真值库，跑出下面这 8 组比较：

| 实验              | 改了什么          | 比较组数 | 结果                               |
| --------------- | ------------- | ---- | -------------------------------- |
| 同配置重跑           | **什么都没改**     | 4    | **全部 D0** —— 连措辞都一样              |
| 同配置重跑           | 什么都没改         | 1    | **D0**（换到有口径陷阱的任务，仍然稳定）          |
| 改 system prompt | `prompt_hash` | 1    | **D2** —— 多插入 2 步（先查表结构）         |
| 换模型             | `model`       | 2    | **D2** —— SQL 写法与措辞变了，**结论一字未变** |

三个结论：

**① temperature=1.0 没有产生任何可观测分叉。** 这是最反直觉的一条。  
我们跑了 `temperature=0` × 3 和 `temperature=1.0` × 3，六条轨迹**逐字节相同**  
（SQL 文本、回答措辞、工具返回全部一致）。

> 这恰恰说明**分叉的来源不是温度，是任务有没有选择空间**。  
> "算一下全量总销售额"这个任务只有一条合理的路可走，采样温度再高也没有别的选项。  
> 如果拿它去论证"我们的 Agent 很稳定"，结论是不可信的 ——  
> **一个没有选择空间的任务，测不出稳定性。**

**② 真正制造分叉的是"改一个变量"。** 追加一条 prompt 规则，第一步立刻变了  
（先 `get_schema` 再取数），对齐器正确识别为**插入 2 步**而不是"整体错位"。  
换成 `qwen-turbo`，SQL 从 `SELECT SUM(Amount) AS total_sales FROM retail`  
变成 `SELECT SUM(Amount) FROM retail;`，结论数字完全相同 → **D2**。

**③ 让"什么都没改却分叉"变成一个显式警报。** 拿真实模型跑对照组的时候，  
一个必须回答的问题是："我看到的分叉，是我改出来的，还是上游本来就有的？"  
所以 `compare_traces` 会把两侧的 `model / prompt_hash / tool_schema_hash /
dataset_snapshot` 一起比，若**身份完全一致却出现 D2 以上分叉**，  
就在 `evidence.unexplained_fork` 里标出来并在报告里红字提示：

> 这不是你改出来的，是上游的非确定性。**这是"确定性"这个前提本身出了裂缝。**

这次实测里它一次都没触发 —— 对照组 5 条全是 D0。  
但**没触发本身是有价值的结论**：它说明在这条任务上，  
至少在这一天、这个端点、这个模型别名下，上游是稳定的。  
将来它一旦触发，你就知道该去找供应商，而不是去翻自己的 diff。

### 门禁按实验类型分开结算

一个 `reports/` 目录里往往同时躺着几种实验，它们的**合理分叉程度根本不同**。  
用一份预算统一判决，结果是换模型那几条把 `prompt_tweak` 顶红，  
而真正该拦的那条淹没在噪音里。所以门禁支持按"改动了哪个变量"筛选：

```
$ python -m replayprobe gate --dir reports --budget default --select none
  已按 --select 无改动 过滤掉 5 条不属于本实验的比较
预算 [default] 通过
  任务 5 条 · 分叉率 0.0%（d2_path 及以上计入）· 匹配率 100.0%
  分布 d0_identical=5                                    → 退出码 0

$ python -m replayprobe gate --dir reports --budget prompt_tweak --select prompt_hash
  已按 --select prompt_hash 过滤掉 9 条不属于本实验的比较
预算 [prompt_tweak] 未通过（3 条违例）
  [!] total   allow_fork_after   分叉出现在对齐位置 0，早于允许的 2 —— 前缀必须稳定
                                                        → 退出码 1

$ python -m replayprobe gate --dir reports --budget model_swap --select model
  已按 --select model 过滤掉 8 条不属于本实验的比较
预算 [model_swap] 通过
  任务 2 条 · 分叉率 0.0%（d3_conclusion 及以上计入）· 匹配率 100.0%
  分布 d2_path=2                                        → 退出码 0
```


同一批数据、三份预算、三种结论。**改 prompt 被拦下是对的**：
它动的是"必须稳定的那个前缀"（第 1 步选什么工具），
这正是 `allow_fork_after=2` 要守的东西。

原始轨迹、判定 JSON 与完整分析方法见 [`docs/真实实验记录.md`](docs/真实实验记录.md) ——
里面包含每一条轨迹的逐步内容和它们的 SHA-256，可逐条复核。

---

## 与同类项目的关系

以下 star 数是 2026-09-25 用 GitHub API 实际查的：

| 项目 | Stars | 它做什么 | replayprobe 补的是什么 |
|---|---|---|---|
| [OpenHands](https://github.com/OpenHands/OpenHands) | 89.1k | 一个完整的编码 Agent harness（沙箱 + 工具 + 界面） | 它是**被诊断的 harness**，不是诊断工具 |
| [Langfuse](https://github.com/langfuse/langfuse) | 35.0k | Agent 追踪与观测：**看见**一条轨迹发生了什么 | 能看，但"差异可不可接受"只能人判 → 进不了 CI |
| [promptfoo](https://github.com/promptfoo/promptfoo) | 25.4k | 提示词 / Agent / RAG 的测试与红队，声明式配置 + CI | 它比"输出对不对"，不比"路走的是不是同一条" |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | 20.4k | Agent harness，把 GitHub issue 自动修掉（NeurIPS 2024） | 同上：harness 本体，非诊断工具 |
| [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | 2.9k | 英国 AISI 的 LLM 评测框架 | 评"答得对不对"，本项评"行为是否回归" |
| [VCR.py](https://github.com/kevin1024/vcrpy) | 3.0k | HTTP 录制 / 重放，让测试不打网络 | **本项目的思想源头**：把不可复现的外部依赖换成录音带 |
| [faultprobe](https://github.com/dandan693/faultprobe) | — | （同作者）给 Agent 注入故障，量"坏了会怎样" | 故障注入 ≠ 行为回归 |

**技术传承是诚实的**：录制 / 重放的思路直接来自 VCR 一类工具，
把它们从"HTTP 请求"这一层，搬到"Agent 的每一个决策接缝"这一层。

同作者三个项目的分工：

```
evalguard    答得对不对     （质量）
faultprobe   坏了会怎样     （鲁棒性）
replayprobe  改动之后走的还是同一条路吗 （行为回归）
```

三者共用同一份领域数据（`retail_truth.db`），所以三张成绩单可以并排看。

---

## 目录结构

```
replayprobe/
├── replayprobe/
│   ├── types.py        数据契约：录制带 / 轨迹 / 分叉（三条纪律写在这里）
│   ├── dataset.py      取数与解析：零依赖下 UCI 的 xlsx、转 CSV、判断源编码
│   ├── signature.py    ★ 技术支点：规范化 + 哈希 + 三级签名 + 结论数字集合运算
│   ├── align.py        签名序列对齐（+ 刻意保留的序号对齐对照组）
│   ├── diverge.py      ★ 五档分级 + 重试/新探索区分 + 崩溃判定
│   ├── budget.py       ★ 分叉预算（把"允许多大分叉"变成退出码）
│   ├── recorder.py     录制：把一次真实运行固化
│   ├── player.py       回放：exact / strict / drill 三模式，铁律在这里
│   ├── llm.py          脚本替身（确定性）+ OpenAI 兼容客户端 + 三个真实任务
│   ├── report.py       零依赖手写 HTML/SVG 报告
│   ├── cli.py          命令行
│   └── agent/          被测的最小 ReAct Agent 与它的三个工具
├── tools/
│   ├── build_truth_db.py   取原始数据（--download 自动下载）并建真值库
│   ├── probe_real_llm.py   真实模型连通性探针（一次调用，验协议）
│   └── demo_replay.py      五段式端到端演示（最短的上手入口）
├── tests/                  182 项单测（标准库 unittest，零依赖）
├── data/
│   ├── truth/              真值库（不进 git，本地跑 build_truth_db.py 生成）
│   ├── raw/                原始数据缓存（不进 git，--download 自动获取）
│   ├── tapes/              录制带
│   │   └── real_qwen-plus-latest_t0_total.*    ★ 真实模型样本，刻意入库
│   └── budgets/            三份分叉预算：default / prompt_tweak / model_swap
├── docs/
│   ├── 项目方案.md         逐条说明选型理由与设计取舍
│   ├── 真实实验记录.md      ★ 真实模型跑了什么、结果如何、修了哪些误报；第 8 节是取数验证
│   └── 路线图.md           已经做完什么、接下来还差什么
└── LICENSE                 MIT
```

---

## CI 集成

> 下面这段是**逐条实跑验证过**的，最后一条门禁退出码为 `0`。
> 基准带用的是**入库的那条真实模型轨迹** —— 所以 CI 里不用 API Key、不联网。
>
> 用 `model_swap` 预算是刻意的：第 ② 步的钻取重放**确实**制造了路径分叉
> （这正是它的目的），而这份预算允许 D2、但不许结论变。

```yaml
- name: 缓存原始数据（不然每次 CI 都要重下 24 MB）
  uses: actions/cache@v4
  with:
    path: data/raw
    key: online-retail-v1

- name: Agent 行为回归门禁
  env:
    TAPE: data/tapes/real_qwen-plus-latest_t0_total.tape.json
  run: |
    python tools/build_truth_db.py --download
    python -m replayprobe check
    python -m unittest discover -s tests -t .          # 182 项

    # ① 确定性自证：同一个带重放两次，必须逐字一致
    python -m replayprobe run --tape "$TAPE" --mode exact --out reports/a.json
    python -m replayprobe run --tape "$TAPE" --mode exact --out reports/b.json
    python -m replayprobe compare --a reports/a.json --b reports/b.json
    #   → d0_identical

    # ② 反事实：从第 3 步切开，看会不会走上另一条路
    python -m replayprobe run --tape "$TAPE" --mode drill --fork-at 2 \
                              --live late_fork --task total --out reports/drill.json
    python -m replayprobe compare --a reports/a.json --b reports/drill.json
    #   → d2_path：数字是包含关系（[8887208.894] → [18532.0, 8887208.89]）
    #     原结论的每个数字都还在，未被否定，只是信息量变了

    # ③ 结算：退出码就是门禁结论，0 放行、1 拦下
    python -m replayprobe gate --dir reports --budget model_swap
    #   预算 [model_swap] 通过
    #     任务 2 条 · 分叉率 0.0%（d3_conclusion 及以上计入）· 匹配率 77.8%
    #     分布 d0_identical=1  d2_path=1
```

同一批结果换一份更紧的预算，结论立刻反转 —— 这就是那两个数字的用途：

```
$ python -m replayprobe gate --dir reports --budget default ; echo $?
预算 [default] 未通过（2 条违例）
  任务 2 条 · 分叉率 50.0%（d2_path 及以上计入）· 匹配率 77.8%
  分布 d0_identical=1  d2_path=1
  [!] total    max_severity          分叉等级 d2_path 超过上限 d1_wording；数字是包含关系…
  [!] (整批)    max_divergence_ratio  分叉率 50.0% 超过上限 0.0% —— 单条都可以接受，
                                      但被碰到的比例这么高，说明改动的影响面比预期大
1
```

违例逐条打出（哪条任务、踩了哪条规则、超了多少），而不是只给一个红叉。

**关于录制带的入库**：`data/tapes/*.json` 默认被 `.gitignore` 排除（带里存的是
完整的模型输入输出）。**唯一的例外是那条真实模型的带，刻意入库**，
让"重放"这件事可以被任何人自己跑一遍验证 —— 见 `.gitignore` 末尾那段说明。

要放行自己录的带当 CI 基准，先脱敏。


---

## 已知局限（如实写下来，免得被追问时说不清）

### 判据本身

1. **只覆盖决策接缝。** 覆盖 LLM 调用与工具调用；多进程 / 异步并行下的
   真实因果还原只预留了 `parent_seq` 字段，没有实现。
2. **文本相似度是全项目唯一的模糊判据**，只在"两侧都没有显著结论数字"时启用。
   默认阈值 0.85 是取舍不是真理，它被显式暴露成参数并在报告里回显。
3. **年份会被误收为结论数字**（整数且 ≥ 100）。误报在实践中很少触发，但真实存在。
4. **不做 SQL 语义等价重写。** `SELECT SUM(Amount)` 与等价的别名写法会被判成
   不同的决策签名 —— 这是"宁可误报，不引入不可复现判据"的刻意取舍。
   （注意：**数字**层面的精度差异已经被容错，见 `same_number`；
   **SQL 文本**层面的等价重写仍然不做。这两件事不是一回事。）
5. **`intent` 级没有接入分级主链路**，它只是可用的第三种分辨率。
6. **不做内容寻址去重。** hash 已经存下来了，但同一个 prompt 重复出现时仍会存多份。
7. **不处理多模态与流式输出。**
8. **`same_number` 用的是"短数是不是长数的四舍五入"这一条规则**，不是通用数值容差。
   它对"小数位数不同"精确有效，但对"换了个口径算出来的近似值"无能为力 ——
   那本来也不该被容忍。
9. **`unexplained_fork` 只能报「有异常」，不能定位。** 它说明"同配置独立重跑却分叉"，
   但分不清是供应商路由、缓存还是权重漂移。要定位得上游的日志，本工具够不着。

### 真实模型接入

10. **模型别名不是身份。** 本工具会记录 `model`，但 `model_revision` 为空 ——
    别名指向的权重可能变。所以真实模型的结论应被读作
    "**某一天的实测**"，而不是"该模型的固有性质"。这写进了每条带的 `manifest.limitations`。
11. **`temperature=0` 不等于确定性，本项目也不这么承诺。**
    实测中温度 0 与 1.0 都没产生分叉 —— 但那是**这条任务**的性质，不是通用保证。
12. **只支持 OpenAI 兼容的 function calling + 一种标签协议兜底。**
    别的供应商协议（Anthropic tool_use、Gemini functionDeclarations 等）要另接。
13. **不支持多工具并行调用。** 模型一次返回多个 `tool_calls` 时只执行第一个，
    但会在 meta 里记下总数，不让这件事无声无息地过去。
14. **不做 token 成本核算。** `usage` 存在 meta 里，但没有汇总成钱。

### 数据来源与取数

15. **数据来源只有 UCI 一处。** 官方现在只提供 xlsx（`online+retail.zip` 里的
    `Online Retail.xlsx`），老的 CSV 已从下载页撤下。如果 UCI 改页面或改文件内容，
    `--download` 会失败 —— 但它会**失败**，而不是安静地拿别的数据把库建出来：
    392,692 行的闸门和 7 项口径自检在后面守着。这道闸门同时在回答
    「我下到的到底是不是同一份数据」。
16. **xlsx 解析的边界**：支持共享字符串、内联字符串、稀疏行、公式的字符串结果、
    整数型浮点的归一化（`536365.0` → `536365`）。**不支持**多工作表挑选
    （多个 sheet 又定位不到数据表时直接报错）、日期格式的自适应
    （哪一列是日期必须显式点名）、`.xls`（老的二进制格式）。
17. **源文件的编码只区分 UTF-8 与 ISO-8859-1。** 其他编码（GBK、UTF-16、Big5…）
    既认不出来、也不会报错，会被当成 ISO-8859-1 解成乱码 ——
    **这是目前最明显的一个静默失败口子。** 用 `--src` 指自己的数据前，请先确认编码。
    没有做「多编码猜测」是刻意的：猜测会引入一个不可复现的判据，
    而且它给出的是**虚假的安全感** —— 错判一次比要求人先转码贵得多。

---

## License

MIT
