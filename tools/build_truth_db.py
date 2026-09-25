#!/usr/bin/env python3
"""从原始数据重建真值库 `retail_truth.db`。

只用标准库（csv + sqlite3）。跑一次约 20 秒，产物约 69 MB。

## 为什么这个脚本必须存在，而且必须可复现

`replayprobe` 里所有"结论数字"——总销售额 8,887,208.89、订单 18,532、
客户 4,338——都来自这张表。**如果这张表的构建过程不可复现，
那么"录制带里记的数字是对的"这句话就没有依据。**

所以这个脚本不只建表，还会在最后**自检**：把关键指标复算一遍，
对不上就非零退出。理由和整个项目一致 ——
**一个安静地给出错误数字的构建脚本，比一个报错的脚本危险得多。**

## 原始数据从哪来

UCI 现在**只提供 xlsx**（在 `online+retail.zip` 里，23.7 MB），
老的那份 CSV 早已不在下载页面上。所以本脚本给三条路：

    python tools/build_truth_db.py --download   # ① 自己下（推荐，一步到位）
    python tools/build_truth_db.py              # ② 用 data/raw/ 下已有的源文件
    python tools/build_truth_db.py --src <文件>  # ③ 指一份自己的（.csv 或 .xlsx 都行）

xlsx 的解析在 `replayprobe/dataset.py` 里，零第三方依赖。
**「clone 之后第一步就跑不起来」是这类项目最常见的死法**，
所以这条路必须由脚本自己走通，而不是写在 README 里让人手工准备。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # 让直接 `python tools/build_truth_db.py` 也能 import

from replayprobe.dataset import (       # noqa: E402
    DEFAULT_TIMEOUT, DatasetError, fetch_online_retail, sniff_encoding, xlsx_to_csv,
)

# 期望的最终行数。写死在代码里是刻意的 —— 它是一道**闸门**，
# 不是一条注释。清洗口径一旦被无意改动，这里会立刻拦住。
EXPECTED_ROWS = 392_692

# UCI 原版的数据行数（不含表头）。只用来提示"下载到的数据对不对"，
# **不作为闸门** —— 因为用 --src 指定自己的数据时这个数字本来就可能不同。
EXPECTED_RAW_ROWS = 541_909

RAW_DIR = ROOT / "data" / "raw"
CACHED_CSV = RAW_DIR / "online_retail.csv"

SCHEMA = """
CREATE TABLE retail (
    InvoiceNo   TEXT,
    StockCode   TEXT,
    Description TEXT,
    Quantity    INTEGER,
    InvoiceDate TEXT,
    UnitPrice   REAL,
    CustomerID  INTEGER,
    Country     TEXT,
    Amount      REAL,
    YearMonth   TEXT
)
"""

INDEXES = [
    ("idx_retail_customer", "retail(CustomerID)"),
    ("idx_retail_invoice", "retail(InvoiceNo)"),
    ("idx_retail_country", "retail(Country)"),
    ("idx_retail_ym", "retail(YearMonth)"),
]

# 原始表用 ISO-8859-1（不是 UTF-8），而 dataset.py 转出来的是 UTF-8。
# 读取时自动判断，见 replayprobe.dataset.sniff_encoding()。

MISSING_HELP = """\
找不到原始数据 —— 这个脚本不会凭空变出数据来。

三选一：

  ① 让它自己下（推荐，约 24 MB，UCI 官方源）：
       python tools/build_truth_db.py --download

  ② 手工下载后放到 data/raw/ 下（.csv 和 .xlsx 都能识别）：
       data/raw/online_retail.csv       或       data/raw/Online Retail.xlsx

  ③ 本机已经有源文件，直接指给它：
       python tools/build_truth_db.py --src /path/to/online_retail.csv

  数据来源：UCI Machine Learning Repository — Online Retail Data Set
            https://archive.ics.uci.edu/dataset/352/online+retail
"""


# ── 取数 ────────────────────────────────────────────────────────────────

def _candidates() -> list[Path]:
    """源文件候选。**只放相对路径** —— 写死某个人的本机绝对路径，
    对别人是噪音，对自己换台机器就是坑。要指自定义位置请用 --src
    或环境变量 REPLAYPROBE_RAW。
    """
    out: list[Path] = []
    env = os.environ.get("REPLAYPROBE_RAW")
    if env:
        out.append(Path(env))
    out.append(CACHED_CSV)
    out.append(RAW_DIR / "Online Retail.xlsx")
    return out


def _as_csv(p: Path, *, force: bool = False) -> Path:
    """源是 xlsx 就先转成 CSV 缓存；是 CSV 就原样返回。"""
    if p.suffix.lower() != ".xlsx":
        return p
    if CACHED_CSV.exists() and not force:
        print(f"[i] 复用已由 {p.name} 转换出来的 {CACHED_CSV.name}")
        return CACHED_CSV
    print(f"[i] 源是 xlsx，先转成 CSV（零依赖解析，约 40 秒）：{p}")
    try:
        xlsx_to_csv(p, CACHED_CSV, date_columns=("InvoiceDate",),
                    log=lambda s: print(f"    {s}"))
    except DatasetError as exc:
        sys.exit(f"xlsx 转换失败：\n{exc}")
    return CACHED_CSV


def resolve_src(explicit: str | None, *, download: bool,
                force: bool, download_timeout: int = DEFAULT_TIMEOUT) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"找不到指定的源文件：{p}")
        return _as_csv(p, force=force)

    if download:
        print("[i] 从 UCI 下载原始数据…")
        try:
            return fetch_online_retail(RAW_DIR, force=force,
                                       timeout=download_timeout,
                                       log=lambda s: print(f"    {s}"))
        except DatasetError as exc:
            sys.exit(f"取数失败：\n{exc}")

    for p in _candidates():
        if p.exists():
            print(f"[i] 用本地源文件：{p}")
            return _as_csv(p, force=force)

    sys.exit(MISSING_HELP)


def parse_dt(raw: str) -> tuple[str, str] | None:
    """把原始日期归一成 (datetime, year-month)。

    原始 CSV 是 `12/1/10 8:26`（月/日/两位年，无零填充），
    dataset.py 转出来的 CSV 是 `2010-12-01 08:26:00`。两种都要能解析。

    必须显式解析 —— 交给字符串切片会悄悄算错月份，而那种错**不会报错**，
    只会让 YearMonth 分组结果整体偏移。
    """
    s = (raw or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = _dt.datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%Y-%m")
        except ValueError:
            continue
    return None


# ── 建库 ────────────────────────────────────────────────────────────────

def build(src: Path, out: Path) -> int:
    t0 = time.time()
    encoding = sniff_encoding(src)
    print(f"源文件   {src}")
    print(f"编码     {encoding}")
    print(f"输出     {out}")
    print("-" * 62)

    rows_raw = 0
    dup = 0
    no_customer = 0
    bad_value = 0
    bad_date = 0
    kept: list[tuple] = []
    seen: set[tuple] = set()

    with open(src, newline="", encoding=encoding) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            sys.exit("源文件是空的")
        # 原始表无 BOM 但有列名；去掉 BOM 与空白再比对，避免因不可见字符误判
        cols = [c.strip().lstrip("\ufeff").strip('"') for c in header]
        need = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice",
                "CustomerID", "Country"}
        missing = need - set(cols)
        if missing:
            sys.exit(f"源文件缺少必需列：{sorted(missing)}；实际列：{cols}")
        idx = {c: i for i, c in enumerate(cols)}

        for rec in reader:
            if not rec or len(rec) < len(cols):
                continue
            rows_raw += 1

            # ① 去重（按整行原值）
            key = tuple((rec[idx[c]] or "").strip() for c in cols)
            if key in seen:
                dup += 1
                continue
            seen.add(key)

            # ② 客户号缺失
            cust = (rec[idx["CustomerID"]] or "").strip()
            if not cust:
                no_customer += 1
                continue

            # ③ 非正数量 / 非正单价（退货与异常单价）
            try:
                qty = int(float(rec[idx["Quantity"]]))
                price = float(rec[idx["UnitPrice"]])
            except ValueError:
                bad_value += 1
                continue
            if qty <= 0 or price <= 0:
                bad_value += 1
                continue

            dt = parse_dt(rec[idx["InvoiceDate"]])
            if dt is None:
                bad_date += 1
                continue
            invoice_date, year_month = dt

            desc = (rec[idx["Description"]] or "").strip() if "Description" in idx else ""
            kept.append((
                (rec[idx["InvoiceNo"]] or "").strip(),
                (rec[idx["StockCode"]] or "").strip(),
                desc, qty, invoice_date, price,
                int(float(cust)),
                (rec[idx["Country"]] or "").strip(),
                qty * price,          # Amount：**不四舍五入**，保留浮点乘积原值
                year_month,
            ))

    print(f"原始行数            {rows_raw:,}")
    print(f"① 去重              -{dup:,}  →  {rows_raw - dup:,}")
    print(f"② 客户号缺失        -{no_customer:,}  →  {rows_raw - dup - no_customer:,}")
    print(f"③ 非正数量/单价     -{bad_value:,}  →  {len(kept):,}")
    if bad_date:
        print(f"   其中日期无法解析  -{bad_date:,}（已单独计数，见下方自检）")
    print(f"最终行数            {len(kept):,}  （保留率 {len(kept) / rows_raw:.1%}）")
    if rows_raw != EXPECTED_RAW_ROWS:
        # 只提示不拦：用 --src 指定自己的数据时，这个数字本来就可能不同。
        print(f"[i] 原始行数 {rows_raw:,} 与 UCI 原版的 {EXPECTED_RAW_ROWS:,} 不同 —— "
              f"如果这份源不是 UCI 原版，忽略本条。")
    print("-" * 62)

    if len(kept) != EXPECTED_ROWS:
        print(f"[!] 行数 {len(kept):,} 与预期的 {EXPECTED_ROWS:,} 不一致。", file=sys.stderr)
        print("    清洗口径可能被改动过 —— 先查清原因，不要直接把期望值改掉。", file=sys.stderr)
        if rows_raw == EXPECTED_RAW_ROWS:
            print("    原始行数是对的，所以问题出在清洗这一步，而不是数据来源。",
                  file=sys.stderr)
        else:
            print("    原始行数也不对 —— 先确认数据来源，再怀疑清洗口径。",
                  file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # closing() 而不是裸的 `with sqlite3.connect(...)`：后者只管事务，不关连接。
    with closing(sqlite3.connect(out)) as conn:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO retail VALUES (?,?,?,?,?,?,?,?,?,?)", kept)
        for name, target in INDEXES:
            conn.execute(f"CREATE INDEX {name} ON {target}")
        conn.commit()

    print(f"已写出 {out}（{out.stat().st_size / 1e6:.1f} MB，{time.time() - t0:.1f}s）")

    # ── 自检：把结论数字复算一遍 ──────────────────────────────────────
    print("-" * 62)
    print("口径自检")
    with closing(sqlite3.connect(f"file:{out}?mode=ro", uri=True)) as conn:
        checks = [
            ("行数", "SELECT COUNT(*) FROM retail", EXPECTED_ROWS),
            ("总销售额", "SELECT ROUND(SUM(Amount),2) FROM retail", 8_887_208.89),
            ("订单数", "SELECT COUNT(DISTINCT InvoiceNo) FROM retail", 18_532),
            ("客户数", "SELECT COUNT(DISTINCT CustomerID) FROM retail", 4_338),
            ("英国占比", "SELECT ROUND(SUM(CASE WHEN Country='United Kingdom' "
                        "THEN Amount ELSE 0 END)/SUM(Amount)*100,2) FROM retail", 81.97),
            # 以下两项验证"清洗后这些情形已不存在"——口径里因此**不需要**
            # 查询期过滤条件。这两条数字是 schema.py 那张口径表的依据。
            ("退货记录数(应为 0)", "SELECT COUNT(*) FROM retail WHERE Quantity<=0", 0),
            ("空客户号(应为 0)", "SELECT COUNT(*) FROM retail "
                                "WHERE CustomerID IS NULL", 0),
        ]
        bad = 0
        for label, sql, want in checks:
            got = conn.execute(sql).fetchone()[0]
            ok = abs(got - want) < 1e-6 if isinstance(want, float) else got == want
            print(f"  {'OK ' if ok else '[!]'} {label:20s} {got!r}"
                  f"{'' if ok else f'  ← 期望 {want!r}'}")
            if not ok:
                bad += 1

    if bad:
        print(f"\n自检未通过：{bad} 项。**不要把期望值改掉去迁就结果** —— "
              f"先查清清洗口径哪里变了。", file=sys.stderr)
        return 3
    print("\n自检全部通过。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="从原始数据重建真值库",
        epilog="原始数据来源：UCI Machine Learning Repository — Online Retail Data Set")
    ap.add_argument("--src", default=None,
                    help="源文件路径（.csv 或 .xlsx 都行）")
    ap.add_argument("--out", default=str(ROOT / "data" / "truth" / "retail_truth.db"),
                    help="输出的 sqlite 路径")
    ap.add_argument("--download", action="store_true",
                    help="自己从 UCI 下载原始数据（约 24 MB，缓存到 data/raw/）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有缓存，强制重新下载与转换")
    ap.add_argument("--download-timeout", type=int, default=DEFAULT_TIMEOUT,
                    metavar="秒",
                    help=f"单次 socket 操作的超时（默认 {DEFAULT_TIMEOUT}）。"
                         f"它是每一步的超时、不是总时长；链路特别慢就调大")
    args = ap.parse_args(argv)
    return build(resolve_src(args.src, download=args.download, force=args.force,
                             download_timeout=args.download_timeout),
                 Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
