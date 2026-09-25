"""零依赖 HTML 报告。

和 faultprobe 一样手写 HTML/SVG，不引 CDN、不装模板引擎。
**一个报告生成器如果需要联网才能渲染，那它在演示现场就会挂。**

报告要回答的只有一个问题：**这次改动，让 Agent 走上另一条路了吗？
如果有，是从哪一步开始的？**
所以版面上最重要的不是总数，而是「第一处分叉」的左右对照 ——
左边是基线怎么做的，右边是这次怎么做，中间是分叉在哪。
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

SEVERITY_STYLE = {
    "d0_identical": ("#0F6E56", "#E1F5EE", "完全一致"),
    "d1_wording": ("#185FA5", "#E6F1FB", "仅措辞不同"),
    "d2_path": ("#854F0B", "#FAEEDA", "路径分叉"),
    "d3_conclusion": ("#993C1D", "#FAECE7", "结论分叉"),
    "d4_safety": ("#A32D2D", "#FCEBEB", "安全分叉"),
}
ORDER = list(SEVERITY_STYLE)


def _style(sev: str) -> tuple[str, str, str]:
    return SEVERITY_STYLE.get(sev, ("#5F5E5A", "#F1EFE8", sev))


def load_verdicts(directory: Path) -> list[dict]:
    out = []
    for p in sorted(directory.glob("*.verdict.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def _kpi(label: str, value: str, tone: str = "") -> str:
    color = {"good": "#0F6E56", "warn": "#854F0B", "bad": "#A32D2D"}.get(tone, "#2C2C2A")
    return (
        f'<div style="flex:1;min-width:150px;background:#FBFAF7;border:1px solid #E4E2DA;'
        f'border-radius:12px;padding:14px 16px">'
        f'<div style="font-size:12px;color:#6B6A64">{html.escape(label)}</div>'
        f'<div style="font-size:24px;font-weight:500;color:{color};margin-top:6px">'
        f'{html.escape(value)}</div></div>'
    )


def _dist_bar(dist: dict[str, int], total: int) -> str:
    if not total:
        return ""
    rows = []
    for sev in ORDER:
        n = dist.get(sev, 0)
        if not n:
            continue
        stroke, fill, label = _style(sev)
        pct = n / total * 100
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
            f'<div style="width:92px;font-size:12px;color:{stroke}">{html.escape(label)}</div>'
            f'<div style="flex:1;background:#F1EFE8;border-radius:6px;height:16px;overflow:hidden">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{stroke};opacity:.85"></div></div>'
            f'<div style="width:64px;font-size:12px;color:#5F5E5A;text-align:right">'
            f'{n} · {pct:.0f}%</div></div>'
        )
    return "".join(rows)


VAR_LABEL = {"model": "模型", "prompt_hash": "提示词", "tool_schema_hash": "工具定义",
             "dataset_snapshot": "数据集", "mode": "运行方式"}


def _prov_block(v: dict) -> str:
    """「这次比较改了什么」—— 报告里必须有，否则 8 组结果没法归因。"""
    prov = (v.get("evidence") or {}).get("provenance") or {}
    a, b = prov.get("baseline") or {}, prov.get("replay") or {}
    changed = prov.get("changed") or []
    parts = []
    if a.get("model") or b.get("model"):
        parts.append(f'{html.escape(str(a.get("model") or "?"))} → '
                     f'{html.escape(str(b.get("model") or "?"))}')
    if changed:
        chips = "".join(
            f'<span style="background:#FAEEDA;color:#854F0B;border:1px solid #E9C97F;'
            f'border-radius:6px;padding:1px 7px;font-size:11px;margin-left:4px">'
            f'{html.escape(VAR_LABEL.get(c, c))}</span>' for c in changed)
        parts.append("改动了：" + chips)
    else:
        parts.append('<span style="color:#888780">未检测到变量改动</span>')

    unexplained = (v.get("evidence") or {}).get("unexplained_fork")
    warn = ""
    if unexplained:
        warn = ('<div style="margin-top:6px;background:#FCEBEB;border:1px solid #F7C1C1;'
                'border-radius:8px;padding:8px 10px;font-size:12px;color:#A32D2D">'
                '什么都没改，却分叉了 —— 这是上游的非确定性，不是你引入的。</div>')
    return (f'<div style="margin-top:8px;font-size:12px;color:#5F5E5A">'
            f'{" · ".join(parts)}</div>{warn}')


def _fusion_block(v: dict) -> str:
    """第一处分叉的左右对照 —— 报告里最该被看见的一块。"""
    fd = v.get("first_divergence")
    if not fd:
        return '<div style="font-size:13px;color:#6B6A64">没有分叉点。</div>'
    return (
        f'<div style="display:flex;gap:12px;align-items:stretch;margin-top:8px">'
        f'<div style="flex:1;background:#E1F5EE;border:1px solid #9FE1CB;border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:11px;color:#0F6E56;margin-bottom:4px">'
        f'基线 · seq{fd.get("a_seq")}</div>'
        f'<div style="font-size:12px;color:#04342C;font-family:ui-monospace,Consolas,monospace;'
        f'word-break:break-all">{html.escape(str(fd.get("a_summary") or ""))}</div></div>'
        f'<div style="align-self:center;font-size:12px;color:#888780">对比</div>'
        f'<div style="flex:1;background:#FCEBEB;border:1px solid #F7C1C1;border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:11px;color:#A32D2D;margin-bottom:4px">'
        f'重放 · seq{fd.get("b_seq")}</div>'
        f'<div style="font-size:12px;color:#501313;font-family:ui-monospace,Consolas,monospace;'
        f'word-break:break-all">{html.escape(str(fd.get("b_summary") or ""))}</div></div></div>'
    )


def render(verdicts: list[dict], title: str = "replayprobe 报告") -> str:
    total = len(verdicts)
    dist: dict[str, int] = {}
    for v in verdicts:
        s = v.get("severity", "")
        dist[s] = dist.get(s, 0) + 1

    diverged = sum(n for s, n in dist.items()
                   if ORDER.index(s) >= ORDER.index("d2_path")) if dist else 0
    healthy = dist.get("d0_identical", 0) + dist.get("d1_wording", 0)
    divergence_ratio = diverged / total if total else 0.0

    # 「没改任何变量却分叉」的条数。这是报告里唯一一个**指向工具之外**的指标：
    # 它说的不是"你的改动有问题"，而是"上游的确定性假设有裂缝"。
    unexplained = sum(1 for v in verdicts
                      if (v.get("evidence") or {}).get("unexplained_fork"))

    models: set[str] = set()
    with_limits = 0
    for v in verdicts:
        prov = (v.get("evidence") or {}).get("provenance") or {}
        for side in ("baseline", "replay"):
            info = prov.get(side) or {}
            if info.get("model"):
                models.add(info["model"])
            if side == "replay" and (info.get("limitations") or 0) > 0:
                with_limits += 1
    model_list = sorted(models)

    cards = []
    for v in verdicts:
        sev = v.get("severity", "")
        stroke, fill, label = _style(sev)
        e = v.get("evidence") or {}
        reasons = "".join(
            f'<li style="margin:2px 0">{html.escape(r)}</li>' for r in (v.get("reasons") or [])
        )
        cards.append(f"""
<section style="background:#fff;border:1px solid #E4E2DA;border-radius:14px;padding:18px 20px;margin:14px 0">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <span style="font-family:ui-monospace,Consolas,monospace;font-size:14px;color:#2C2C2A">
      {html.escape(str(v.get('task_id') or ''))}</span>
    <span style="background:{fill};color:{stroke};border:1px solid {stroke};
      border-radius:999px;padding:2px 10px;font-size:12px">{html.escape(label)}</span>
    <span style="font-size:12px;color:#6B6A64">
      {html.escape(str(v.get('baseline_variant') or ''))} → {html.escape(str(v.get('replay_variant') or ''))}</span>
  </div>
  <ul style="margin:10px 0 0 18px;padding:0;font-size:13px;color:#2C2C2A;line-height:1.7">{reasons}</ul>
  {_prov_block(v)}
  {_fusion_block(v)}
  <div style="margin-top:10px;font-size:12px;color:#6B6A64">
    对齐：匹配 {e.get('matched', 0)} 步 · 插入 {e.get('insertions', 0)} · 删除 {e.get('deletions', 0)}
    · 决策分叉 {e.get('decision_mismatches', 0)} 处 · 结论关系 {html.escape(str(e.get('conclusion_relation', '')))}
  </div>
</section>""")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title></head>
<body style="margin:0;background:#F7F6F2;font-family:system-ui,'Segoe UI','Microsoft YaHei',sans-serif;color:#2C2C2A">
<div style="max-width:960px;margin:0 auto;padding:32px 20px 64px">

  <h1 style="font-size:20px;font-weight:500;margin:0 0 4px">{html.escape(title)}</h1>
  <div style="font-size:13px;color:#6B6A64;margin-bottom:20px">
    生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
    回答一个问题：这次改动，让它走上另一条路了吗？从哪一步开始？</div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">
    {_kpi('比较条数', str(total))}
    {_kpi('未分叉（D0+D1）', str(healthy), 'good' if healthy == total and total else '')}
    {_kpi('出现分叉（D2+）', str(diverged), 'bad' if diverged else 'good')}
    {_kpi('分叉率', f'{divergence_ratio * 100:.0f}%',
          'bad' if divergence_ratio > 0.2 else ('good' if total else ''))}
    {_kpi('未解释的分叉', str(unexplained),
          'bad' if unexplained else ('good' if total else ''))}
  </div>

  <section style="background:#fff;border:1px solid #E4E2DA;border-radius:14px;padding:18px 20px">
    <div style="font-size:14px;font-weight:500;margin-bottom:8px">这次比较涉及什么</div>
    <div style="font-size:13px;color:#2C2C2A;line-height:1.9">
      出现的模型：{html.escape('、'.join(model_list)) or '（未记录）'}<br>
      声明了「不可保证项」的轨迹：{with_limits} / {total} 条
    </div>
    <div style="font-size:12px;color:#6B6A64;margin-top:8px;line-height:1.7">
      「不可保证项」是录制带自己声明的不确定性（模型别名可能换权重、上游可能缓存…）。
      <strong>一个不承认自己有不可保证项的重放系统，是在假装确定性。</strong>
      真实模型录出来的带一定带这几条 —— 它们不是缺陷说明，是结论的适用边界。
    </div>
  </section>

  <section style="background:#fff;border:1px solid #E4E2DA;border-radius:14px;padding:18px 20px">
    <div style="font-size:14px;font-weight:500;margin-bottom:8px">分叉等级分布</div>
    {_dist_bar(dist, total) or '<div style="font-size:13px;color:#6B6A64">没有数据。</div>'}
    <div style="font-size:12px;color:#6B6A64;margin-top:10px;line-height:1.7">
      D0/D1 是常态（措辞本来就会变），D2 要人看一眼，D3/D4 该拦下来。
      <strong>把 D1 和 D3 记成同一件事，是这类门禁最常见的死法。</strong>
    </div>
  </section>

  {''.join(cards) or ''}

  <div style="font-size:12px;color:#888780;margin-top:24px;line-height:1.7">
    判据全程不调模型：签名相等、集合运算、文本相似度。
    唯一的模糊阈值是「无显著数字时的文本相似度」，它会回显在每条证据里。
  </div>
</div></body></html>"""


def write_report(directory: Path, out: Path | None = None,
                 title: str = "replayprobe 报告") -> Path:
    verdicts = load_verdicts(directory)
    html_text = render(verdicts, title)
    target = out or (directory / "report.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_text, encoding="utf-8")
    return target
