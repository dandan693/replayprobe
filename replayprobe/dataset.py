"""零依赖地取回 UCI Online Retail 原始数据，并转成 CSV。

## 为什么这个模块存在（它补的是一个真实存在的坑）

`tools/build_truth_db.py` 的输入是一份 44 MB 的 CSV，而 **UCI 现在只提供 xlsx**：
`online+retail.zip` 里只有一个 `Online Retail.xlsx`（23.7 MB，实测确认）。

早先的 README 把 `python tools/build_truth_db.py` 写成「30 秒上手」的第一步，
而那个脚本**只找本地文件、不会下载** —— 于是新 clone 的人走到第一步必然失败。
这正是本仓库自己反复警告的那种问题：**文档承诺的和实际能跑的，不是同一件事。**

所以这里补齐的是「第一公里」：把原始数据弄到手。

## 为什么不用 pandas / openpyxl

理由和整个仓库一致：**一个门禁如果 clone 之后跑不起来，就没有人会跑它。**
xlsx 本身就是一个 zip 加几个 XML，标准库够用。代价是要自己处理三件事，
而这三件事全都是「写错了不会报错、只会安静地算错」的类型：

1. **列位置只能按单元格的 `r` 属性（如 `C7`）算，不能按出现顺序算。**
   稀疏行会跳过空单元格；按顺序读，空值之后的整行会整体串列 ——
   而串出来的表**看起来是满的**，不报错，只是每一列的含义都错了。
2. **共享字符串表（`sharedStrings.xml`）**：`t="s"` 的单元格里存的是**索引**，
   不是文本本身。忘了查表，商品名那一列就会变成一列整数。
3. **日期是序列号**：`InvoiceDate` 存的是 `36660.3513888…`（自 1899-12-30 起的天数），
   不是日期字符串。当成普通数字读下去，下游所有按时间分组的结论会整体偏移。

另外还有一个只有真的跑过才会发现的坑，写在 `_normalize_number()` 里：
**Excel 会把纯数字的 InvoiceNo 存成浮点**，于是同一张表里
「字符串写法」和「数值写法」并存。它不影响数值计算，但会影响
`COUNT(DISTINCT InvoiceNo)`。

本模块的纪律和其他模块一致：**要么完整，要么报错**，绝不返回半个结果。
"""

from __future__ import annotations

import codecs
import csv
import datetime as _dt
import http.client
import io
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable, Iterator

__all__ = [
    "DatasetError",
    "UCI_ZIP_URL",
    "UCI_XLSX_URL",
    "col_index",
    "excel_serial_to_text",
    "sniff_encoding",
    "xlsx_rows",
    "xlsx_to_csv",
    "download_zip",
    "extract_single",
    "fetch_online_retail",
]

# UCI 的两个入口。zip 是首选（一次就能拿全）；xlsx 是直链，留作 zip 挂掉时的退路。
# 两个地址都实测 HTTP 200（见 docs/真实实验记录.md 第 8 节）。
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_XLSX_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "00352/Online%20Retail.xlsx")

USER_AGENT = "replayprobe/0.1 (+https://github.com/dandan693/replayprobe)"

# Excel 的日期原点是 1899-12-30 而不是 1900-01-01。
# 差这一天是故意的：Excel 为了兼容 Lotus 1-2-3，把 1900 当成闰年
# （序列号 60 对应一个**并不存在**的 1900-02-29）。所以 61 以后要减掉这一天。
# 本数据集的日期都在 2010–2011（序列号 > 40000），落在受影响区间之后。
EXCEL_EPOCH = _dt.datetime(1899, 12, 30)

# 能安全转换的最小序列号。见 excel_serial_to_text() 的说明。
EXCEL_MIN_SERIAL = 61

XLSX_SUFFIX = ".xlsx"

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class DatasetError(RuntimeError):
    """原始数据取回或解析失败。

    单独一个异常类型是刻意的：调用方要能把「数据没到手」和「数据解析坏了」
    跟其他错误区分开 —— 前者重试就好，后者说明代码要改。
    """


def _local(tag: str) -> str:
    """去掉 XML 命名空间，只留标签名。"""
    return tag.rsplit("}", 1)[-1]


# ── 单元格引用 ──────────────────────────────────────────────────────────

def col_index(ref: str) -> int:
    """把 `AB12` 这类单元格引用里的列部分转成 0 基索引；没有列字母时返回 -1。

    这就是第 1 条纪律的实现：**列位置来自 `r` 属性，不是来自出现顺序。**
    """
    n = 0
    seen = False
    for ch in ref:
        up = ch.upper()
        if "A" <= up <= "Z":
            n = n * 26 + (ord(up) - 64)
            seen = True
        else:
            break
    return n - 1 if seen else -1


def _normalize_number(raw: str) -> str:
    """把「数值相等但写法不同」的数字归一化：`536365.0` → `536365`。

    为什么必须做（这是实测才发现的）：Excel 会把纯数字的 InvoiceNo 存成浮点，
    于是 xlsx 里读出来是 `536365.0`，而原始 CSV 里是 `536365`。
    两者**数值相同、字符串不同** —— 而下游是按字符串做
    `COUNT(DISTINCT InvoiceNo)` 的，于是同一个订单会被算成两个。
    这类错**不会报错**，只会让「订单数」这个结论悄悄变大。

    只在「是整数、且没有科学计数法」时动手，其余原样保留：
    `2.55` 保持 `2.55`，`1.5e-3` 保持原样 —— 我们不知道原始 CSV 怎么写它，
    能不碰就不碰。
    """
    s = raw.strip()
    if not s or "." not in s or "e" in s or "E" in s:
        return raw
    try:
        f = float(s)
    except ValueError:
        return raw
    if abs(f) >= 1e15:          # 超出 double 精确整数范围，别硬转
        return raw
    return str(int(f)) if f == int(f) else raw


def excel_serial_to_text(raw: str) -> str | None:
    """把 Excel 的日期序列号转成 `YYYY-MM-DD HH:MM:SS`；不在可转范围内则返回 None。

    **这不是一个通用日期转换器** —— 它只需要覆盖本数据集的日期范围
    （2010–2011，序列号 40000 上下）。超出范围时**返回 None 而不是猜**：

    下界取 61 是刻意的。Excel 有个 1900 闰年 bug，序列号 60 对应一个
    **并不存在的日期**（1900-02-29），它之前和之后要用不同的原点。
    拿同一个原点硬转那一段，结果是安静地差一天 —— 所以这里直接不转。
    返回 None 是安全的：调用方会把原值留着，下游解析时必然失败，
    最终被真值库的行数闸门拦下。**它会炸，但不会给出错的数字。**

    返回 None 的另一种情况是「这一格根本不是日期」，那是合法输入，不是错误。
    """
    s = (raw or "").strip()
    if not s:
        return None
    try:
        serial = float(s)
    except ValueError:
        return None
    if not EXCEL_MIN_SERIAL <= serial < 2_958_466.0:    # 1900-03-01 ~ 9999-12-31
        return None
    days = int(serial)
    secs = int(round((serial - days) * 86400))
    try:
        dt = EXCEL_EPOCH + _dt.timedelta(days=days, seconds=secs)
    except (OverflowError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── xlsx 读取 ──────────────────────────────────────────────────────────

def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """读共享字符串表。缺这个文件是合法的（没有文本单元格时就不会有）。"""
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    out: list[str] = []
    for _, el in ET.iterparse(io.BytesIO(data), events=("end",)):
        if _local(el.tag) != "si":
            continue
        # 一个 <si> 可能被拆成多个 <r><t>（富文本分段），必须拼起来，
        # 只取第一个 <t> 会得到被截断的商品名。
        out.append("".join(t.text or "" for t in el.iter() if _local(t.tag) == "t"))
        el.clear()
    return out


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    """定位第一个工作表。走 workbook → rels 这条正路，而不是猜文件名。"""
    sheets = sorted(n for n in zf.namelist()
                    if n.startswith("xl/worksheets/") and n.endswith(".xml"))
    if not sheets:
        raise DatasetError(
            f"这个 xlsx 里没有工作表（xl/worksheets/*.xml）。"
            f"实际内容：{zf.namelist()}")

    rels: dict[str, str] = {}
    try:
        for r in ET.fromstring(zf.read("xl/_rels/workbook.xml.rels")):
            rels[r.get("Id")] = r.get("Target") or ""
    except (KeyError, ET.ParseError):
        rels = {}

    try:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
    except (KeyError, ET.ParseError):
        return sheets[0]

    for sh in wb.iter():                      # 只看第一个 sheet
        if _local(sh.tag) != "sheet":
            continue
        target = rels.get(sh.get(f"{_REL_NS}id") or "")
        if target:
            target = target.lstrip("/")
            cand = target if target.startswith("xl/") else f"xl/{target}"
            if cand in zf.namelist():
                return cand
        break

    if len(sheets) > 1:
        # 有多个 sheet 又定位不到：宁可报错，也不要安静地读错那张表。
        raise DatasetError(
            f"xlsx 里有 {len(sheets)} 个工作表，但无法从 workbook.xml 确定哪张是数据表："
            f"{sheets}")
    return sheets[0]


def _cell_text(c: ET.Element, shared: list[str]) -> str:
    """取一个 `<c>` 单元格的文本值。"""
    t = c.get("t")

    if t == "inlineStr":
        return "".join(x.text or "" for x in c.iter() if _local(x.tag) == "t")

    v = None
    for x in c:
        if _local(x.tag) == "v":
            v = x.text
            break
    if v is None:
        return ""

    if t == "s":                              # 共享字符串：v 是索引，不是文本
        try:
            return shared[int(v)]
        except (ValueError, IndexError):
            raise DatasetError(
                f"共享字符串索引越界：{v!r}（表里有 {len(shared)} 条）") from None
    if t == "b":
        return "1" if v.strip() == "1" else "0"
    if t == "e":
        raise DatasetError(f"单元格是错误值 {v!r}；数据不干净，别猜它的含义")
    # t in (None, "n", "str")：数字，或公式的字符串结果
    return _normalize_number(v)


def _row_values(row_el: ET.Element, shared: list[str]) -> list[str]:
    cells: dict[int, str] = {}
    nxt = 0
    for c in row_el:
        if _local(c.tag) != "c":
            continue
        idx = col_index(c.get("r") or "")
        if idx < 0:
            idx = nxt                          # 标准要求有 r；没有时只能按顺序容错
        nxt = idx + 1
        cells[idx] = _cell_text(c, shared)
    if not cells:
        return []
    return [cells.get(i, "") for i in range(max(cells) + 1)]


def xlsx_rows(path: str | Path) -> Iterator[list[str]]:
    """逐行产出 xlsx 的单元格文本（**保真**：数字原样透传，不做日期转换）。

    用 `iterparse` 增量解析。整个 sheet 的 XML 解压后有上百 MB，
    一次性 `fromstring` 会吃掉一两个 GB —— 在一台普通笔记本上就是「跑不动」。
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"找不到 xlsx：{path}")
    if not zipfile.is_zipfile(path):
        raise DatasetError(f"这不是一个 xlsx（xlsx 本质是 zip）：{path}")

    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        sheet = _first_sheet_path(zf)
        with zf.open(sheet) as fh:
            context = ET.iterparse(fh, events=("start", "end"))
            try:
                _, root = next(context)
            except StopIteration:
                raise DatasetError(f"{path} 的工作表是空的") from None
            for event, el in context:
                if event != "end" or _local(el.tag) != "row":
                    continue
                yield _row_values(el, shared)
                # 关键：不清空就会把 54 万个行元素的空壳攒在内存里。
                root.clear()


# ── 下载 ────────────────────────────────────────────────────────────────

class _Truncated(RuntimeError):
    """内部用：服务端声明的长度和实收长度对不上。

    单独做一个类型，是为了让它和网络异常走同一条重试路径 ——
    截断绝大多数时候也是**瞬时**的，不该一断就判死刑。
    """


def _download_once(part: Path, url: str, timeout: int,
                   log: Callable[[str], None] | None) -> int:
    """下载一次到 `part`，返回收到的字节数。失败往上抛，由调用方决定是否重试。"""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # 注意：UCI 走的是 **chunked 编码**（实测响应头里没有 Content-Length），
        # 所以 total 常常是 0，下面那句长度校验对这份数据其实是跳过的。
        # 真正兜住截断的是 is_zipfile —— zip 的中央目录在文件末尾，
        # 被截断的 zip 一定读不出来。
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last_pct = -1
        with open(part, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if log and total:
                    pct = got * 100 // total
                    if pct != last_pct and pct % 5 == 0:
                        last_pct = pct
                        log(f"  {got / 1e6:.0f}/{total / 1e6:.0f} MB（{pct}%）")
    if total and got != total:
        raise _Truncated(f"下载不完整：收到 {got:,} 字节，声明 {total:,} 字节")
    return got


def download_zip(dest: str | Path, *, url: str = UCI_ZIP_URL, timeout: int = 600,
                 force: bool = False, retries: int = 3, backoff: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] | None = None) -> Path:
    """下载 zip 到 `dest`（已存在就复用）。返回 zip 路径。

    下载**先写 .part，校验完整后再改名**。「看起来下载完了但其实是半个文件」
    是这类脚本最经典的静默失败：zipfile 可能连读都不报错。

    ## 为什么要重试（这行代码是被真实故障逼出来的）

    原先这里只捕获 `(URLError, TimeoutError, OSError)`。但 UCI 是 chunked 编码，
    连接中途断掉时抛的是 **`http.client.IncompleteRead`** ——
    它继承自 `HTTPException`，**不是 `OSError` 的子类**，所以整个 except 抓不住它。
    后果有两个，都是坏的：半截 `.part` 留在磁盘上没人清；
    使用者看到的是裸堆栈，而不是那句"可以手动下载后重跑"。

    这个 bug 只在**网络真的抖了一下**的时候才出现，本机跑十次也未必遇上一次 ——
    是新 clone 的实测把它逼出来的。所以现在的纪律是：
    **断一次不算失败，自动重来；断够 `retries` 次才算失败，并且不留残骸。**
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        if log:
            log(f"复用已下载的 {dest.name}（{dest.stat().st_size / 1e6:.1f} MB）")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    # 清掉上一次崩在半路留下的 .part。不清的话它会一直躺在那里，
    # 让下一个人怀疑"到底下载了没有"。
    if part.exists():
        part.unlink(missing_ok=True)
        if log:
            log(f"清掉上次留下的半截文件 {part.name}")

    if log:
        log(f"下载 {url}")

    last_exc: BaseException | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            _download_once(part, url, timeout, log)
            last_exc = None
            break
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException, _Truncated) as exc:
            # HTTPException 单独列出来，就是为了 IncompleteRead —— 见 docstring。
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt < max(1, retries):
                wait = backoff ** attempt
                if log:
                    log(f"  第 {attempt} 次断了（{type(exc).__name__}），"
                        f"{wait:.0f}s 后重试")
                sleep(wait)

    if last_exc is not None:
        raise DatasetError(
            f"下载失败（重试 {max(1, retries)} 次都没成功）：{url}\n"
            f"  {type(last_exc).__name__}: {last_exc}\n"
            f"  可以手动下载后放到 {dest} 再重跑（脚本会自动复用）。") from last_exc

    size = part.stat().st_size
    if not zipfile.is_zipfile(part):
        part.unlink(missing_ok=True)
        raise DatasetError(
            f"下载回来的不是 zip（收到 {size:,} 字节）：{url}\n"
            f"  可能是重定向到了错误页、网络中间有代理，或者连接在末尾被截断了。"
            f"\n  可以手动下载后放到 {dest} 再重跑（脚本会自动复用）。")
    part.replace(dest)
    if log:
        log(f"已保存 {dest}（{size / 1e6:.1f} MB）")
    return dest


def extract_single(zip_path: str | Path, dest_dir: str | Path, *,
                   suffix: str = XLSX_SUFFIX,
                   log: Callable[[str], None] | None = None) -> Path:
    """把 zip 里**唯一**一个指定后缀的文件解出来。不唯一就报错。"""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(suffix)
                 and not n.endswith("/")]
        if not names:
            raise DatasetError(
                f"{zip_path.name} 里没有 {suffix} 文件。实际内容：{zf.namelist()}")
        if len(names) > 1:
            raise DatasetError(
                f"期望恰好 1 个 {suffix}，实际 {len(names)} 个：{names}\n"
                f"  先确认压缩包内容，不要猜哪一个是数据文件。")
        dest = Path(dest_dir) / Path(names[0]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(names[0]) as src, open(dest, "wb") as fh:
            shutil.copyfileobj(src, fh, 1 << 20)
    if log:
        log(f"已解出 {dest}（{dest.stat().st_size / 1e6:.1f} MB）")
    return dest


# ── 转换 ────────────────────────────────────────────────────────────────

def xlsx_to_csv(src: str | Path, dest: str | Path, *, date_columns: tuple[str, ...] = (),
                log: Callable[[str], None] | None = None) -> int:
    """把 xlsx 转成 UTF-8 CSV，返回**数据行数**（不含表头）。

    `date_columns` 里的列会做 Excel 序列号 → 日期字符串的转换；其余列一律保真透传。
    输出用 UTF-8，而 `build_truth_db.py` 读的时候会先试 UTF-8 再回落 ISO-8859-1 ——
    这样手工下载的老 CSV（ISO-8859-1）和这里转出来的新 CSV 都能吃。
    """
    src, dest = Path(src), Path(dest)
    rows = xlsx_rows(src)
    try:
        header = next(rows)
    except StopIteration:
        raise DatasetError(f"{src} 没有表头行") from None

    cols = [h.strip().lstrip("\ufeff") for h in header]
    date_idx: dict[int, str] = {}
    for name in date_columns:
        if name not in cols:
            raise DatasetError(
                f"表头里没有日期列 {name!r}。实际列：{cols}\n"
                f"  列名变了就必须显式改这里，不能默默跳过 —— 跳过的后果是"
                f"整列日期变成一串数字，且不报错。")
        date_idx[cols.index(name)] = name

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    n = 0
    unconverted = 0
    width = len(cols)
    with open(part, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in rows:
            if len(row) < width:
                row = row + [""] * (width - len(row))
            for i in date_idx:
                raw = row[i].strip()
                if not raw:
                    continue
                conv = excel_serial_to_text(raw)
                if conv is None:
                    unconverted += 1        # 保留原样，交给下游的 parse_dt
                else:
                    row[i] = conv
            w.writerow(row[:width])
            n += 1
            if log and n % 100_000 == 0:
                log(f"  已转换 {n:,} 行")
    part.replace(dest)
    if log:
        log(f"已写出 {dest}（{n:,} 行，{dest.stat().st_size / 1e6:.1f} MB）")
    if unconverted:
        # 不抛错：可能只是个别空值写法不同。但**必须说出来** ——
        # 如果整列都没转成功，下游的行数闸门会拦住，这里先给一条线索。
        print(f"[!] 有 {unconverted:,} 个日期格没能按 Excel 序列号解析，已按原样写出",
              file=sys.stderr)
    return n


# ── 读源 CSV 的编码判断 ──────────────────────────────────────────────────

LEGACY_ENCODING = "ISO-8859-1"


def sniff_encoding(path: str | Path) -> str:
    """判断源 CSV 的编码，返回 `"utf-8"` 或 `"ISO-8859-1"`。

    **顺序不能反。** ISO-8859-1 能解码任意字节序列，所以先试它永远不会失败 ——
    一个 UTF-8 文件会被"成功"解成乱码，而且**不报错**。
    所以必须先用 UTF-8 走一遍完整校验，失败才回落。

    用增量解码器而不是读一段头部来试，是因为头部可能在多字节字符中间截断，
    那样会把一个正常的 UTF-8 文件误判成 ISO-8859-1 —— 同样是静默的错。

    （文件恰好全是 ASCII 时两者结果完全相同，判成哪个都无所谓。）
    """
    dec = codecs.getincrementaldecoder("utf-8")()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            try:
                dec.decode(chunk)
            except UnicodeDecodeError:
                return LEGACY_ENCODING
    return "utf-8"


# ── 一站式取数 ──────────────────────────────────────────────────────────

def fetch_online_retail(raw_dir: str | Path, *, force: bool = False,
                        date_columns: tuple[str, ...] = ("InvoiceDate",),
                        log: Callable[[str], None] | None = None) -> Path:
    """保证 `raw_dir` 下有一份可用的 `online_retail.csv`，返回它。

    步骤：zip（缓存）→ xlsx（解出）→ csv（转换）。每一步都已存在就跳过，
    所以第二次跑是秒级的。`force=True` 会从头重来。
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "online_retail.csv"
    if csv_path.exists() and not force:
        if log:
            log(f"复用 {csv_path}")
        return csv_path

    zip_path = download_zip(raw_dir / "online+retail.zip", force=force, log=log)
    xlsx_path = extract_single(zip_path, raw_dir, log=log)
    xlsx_to_csv(xlsx_path, csv_path, date_columns=date_columns, log=log)
    return csv_path
