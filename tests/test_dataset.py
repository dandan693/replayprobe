"""原始数据层的测试：**全程不打网络**。

被测试的是 `replayprobe/dataset.py` —— 把 UCI 的 xlsx 变成 CSV 的那一层。
这一层的每个坑都是「写错了不会报错、只会安静地算错」的类型：

- 按出现顺序而不是按 `r` 属性读列 → 稀疏行整体串列，而串出来的表是"满"的；
- 忘了查共享字符串表 → 商品名那列变成一列整数；
- 把日期序列号当数字读 → 下游所有按时间分组的结论整体偏移；
- 编码判反了 → 文本变乱码，而 ISO-8859-1 解码**永不失败**。

所以这个文件的重点是**用最小的构造输入**把这些坑一个个钉住：
xlsx 由测试现场用 `zipfile` 手写，不依赖任何外部文件、也不下载。
"""

from __future__ import annotations

import contextlib
import http.client
import io
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

from replayprobe.dataset import (
    DatasetError,
    col_index,
    download_zip,
    excel_serial_to_text,
    extract_single,
    fetch_online_retail,
    sniff_encoding,
    xlsx_rows,
    xlsx_to_csv,
)

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"

_WORKBOOK = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<workbook xmlns="{_MAIN}" xmlns:r="{_REL}">'
    '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
)


# ── 现场构造 xlsx ────────────────────────────────────────────────────────
#
# 不用 openpyxl（零依赖），也不把样例文件塞进仓库（那会变成"改不动、看不懂"的
# 二进制黑盒）。xlsx 就是一个 zip，直接写出来最短。

def _sheet(rows_xml: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="{_MAIN}"><sheetData>{rows_xml}</sheetData></worksheet>')


def _row(r: int, cells: list[str]) -> str:
    return f'<row r="{r}">' + "".join(cells) + "</row>"


def _num(ref: str, v) -> str:
    return f'<c r="{ref}"><v>{v}</v></c>'


def _shared(ref: str, i: int) -> str:
    return f'<c r="{ref}" t="s"><v>{i}</v></c>'


def _inline(ref: str, text: str) -> str:
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _write_xlsx(path: Path, **kw) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        _fill_xlsx(zf, **kw)
    return path


def _xlsx_bytes(**kw) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        _fill_xlsx(zf, **kw)
    return buf.getvalue()


def _fill_xlsx(zf: zipfile.ZipFile, *, shared: tuple[str, ...] = (),
               sheet_xml: str = "", workbook: str = _WORKBOOK,
               rels: str | None = None, extra: dict[str, str] | None = None) -> None:
    if rels is None:
        rels = (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Relationships xmlns="{_PKG}">'
            f'<Relationship Id="rId1" Type="{_REL}/worksheet" '
            f'Target="worksheets/sheet1.xml"/>'
            f'<Relationship Id="rId2" Type="{_REL}/sharedStrings" '
            f'Target="sharedStrings.xml"/></Relationships>'
        )
    sst = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<sst xmlns="{_MAIN}" count="{len(shared)}" uniqueCount="{len(shared)}">'
           + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
    zf.writestr("xl/workbook.xml", workbook)
    zf.writestr("xl/_rels/workbook.xml.rels", rels)
    zf.writestr("xl/worksheets/sheet1.xml", sheet_xml or _sheet(""))
    if shared:
        zf.writestr("xl/sharedStrings.xml", sst)
    for name, body in (extra or {}).items():
        zf.writestr(name, body)


# ── 单元格引用 ──────────────────────────────────────────────────────────

class TestColIndex(unittest.TestCase):
    def test_basic_letters(self):
        self.assertEqual(col_index("A1"), 0)
        self.assertEqual(col_index("B1"), 1)
        self.assertEqual(col_index("Z9"), 25)
        self.assertEqual(col_index("AA1"), 26)
        self.assertEqual(col_index("AB1"), 27)
        self.assertEqual(col_index("BA2"), 52)

    def test_lowercase_is_tolerated(self):
        self.assertEqual(col_index("c7"), 2)

    def test_no_letters_returns_minus_one(self):
        # 调用方据此回落到"按顺序"的容错路径，而不是把列号算成 -1 之后悄悄错位。
        self.assertEqual(col_index("12"), -1)
        self.assertEqual(col_index(""), -1)


# ── 数字归一化 ──────────────────────────────────────────────────────────

class TestNumberNormalization(unittest.TestCase):
    """`536365.0` → `536365` 这件事，看着琐碎，但它决定订单数对不对。"""

    def test_integer_valued_float_loses_the_tail(self):
        # Excel 会把纯数字的 InvoiceNo 存成浮点。原始 CSV 里是 `536365`，
        # 而下游按**字符串**做 COUNT(DISTINCT InvoiceNo) —— 不归一化就是两个订单。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_num("A1", "536365.0")])))
            self.assertEqual(list(xlsx_rows(p)), [["536365"]])

    def test_real_price_is_not_touched(self):
        # 实测：UCI 的 xlsx 里 UnitPrice 存的就是 2.5499999999999998，不是 2.55。
        # **必须原样透传** —— 任何格式化都可能让 Amount 的乘积漂移。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_num("A1", "2.5499999999999998")])))
            self.assertEqual(list(xlsx_rows(p)), [["2.5499999999999998"]])

    def test_scientific_notation_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_num("A1", "1.5E-3")])))
            self.assertEqual(list(xlsx_rows(p)), [["1.5E-3"]])

    def test_huge_values_are_left_alone(self):
        # 超出 double 的精确整数范围就不该硬转，否则会把大数改坏。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_num("A1", "1.0E16")])))
            self.assertEqual(list(xlsx_rows(p)), [["1.0E16"]])


# ── 日期序列号 ──────────────────────────────────────────────────────────

class TestExcelSerial(unittest.TestCase):
    def test_real_value_from_the_dataset(self):
        # 实测值：40513.351388888892 就是 2010-12-01 08:26（原 CSV 写作 `12/1/10 8:26`）。
        self.assertEqual(excel_serial_to_text("40513.351388888892"),
                         "2010-12-01 08:26:00")

    def test_epoch_anchor(self):
        # 序列号 61 = 1900-03-01，即 1899-12-30 这个原点有效的第一段。
        self.assertEqual(excel_serial_to_text("61"), "1900-03-01 00:00:00")

    def test_ambiguous_excel_epoch_is_refused_not_guessed(self):
        """序列号 60 之前是 Excel 的 1900 闰年 bug 区。

        60 对应一个**并不存在**的 1900-02-29，它前后的日期原点不同。
        这里刻意返回 None（= 不转）而不是猜一个原点 —— 猜错的表现是
        **安静地差一天**，而一天之差会挪动整个年月的分组。
        """
        for ambiguous in ("1", "59", "60"):
            self.assertIsNone(excel_serial_to_text(ambiguous),
                              f"序列号 {ambiguous} 落在歧义区，不该被猜着转换")

    def test_rounds_to_whole_seconds(self):
        self.assertEqual(excel_serial_to_text("40513.5"), "2010-12-01 12:00:00")

    def test_non_numbers_return_none(self):
        # 返回 None 而不是抛错：「这一格不是日期」是合法情况，
        # 调用方会把原值留着交给下游再试一次。
        for bad in ("", "   ", "2010-12-01", "abc", None):
            self.assertIsNone(excel_serial_to_text(bad))

    def test_out_of_range_returns_none(self):
        for bad in ("0", "-5", "1e12"):
            self.assertIsNone(excel_serial_to_text(bad))


# ── xlsx 读取 ───────────────────────────────────────────────────────────

class TestXlsxRows(unittest.TestCase):
    def test_header_and_shared_strings_and_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(
                Path(d) / "t.xlsx",
                shared=("InvoiceNo", "StockCode", "UnitPrice", "85123A"),
                sheet_xml=_sheet(
                    _row(1, [_shared("A1", 0), _shared("B1", 1), _shared("C1", 2)])
                    + _row(2, [_num("A2", 536365), _shared("B2", 3), _num("C2", "2.55")])
                ),
            )
            self.assertEqual(list(xlsx_rows(p)), [
                ["InvoiceNo", "StockCode", "UnitPrice"],
                ["536365", "85123A", "2.55"],
            ])

    def test_missing_cell_does_not_shift_columns(self):
        """第 1 号纪律：列位置由 `r` 决定，不是由出现顺序决定。

        这一行只有 A 和 C 两格。按顺序读会得到 `['1','3']`（或把 3 放到 B 列）——
        整行含义都错了，而表看起来还是"满"的。这是本层最贵的一个坑。
        """
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_num("A1", 1), _num("C1", 3)])))
            self.assertEqual(list(xlsx_rows(p)), [["1", "", "3"]])

    def test_cells_without_ref_fall_back_to_order(self):
        # 标准要求单元格带 r；真遇到不带的，只能按顺序读 —— 但要**能读**，
        # 而不是把它当成第 -1 列。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet('<row r="1"><c><v>7</v></c><c><v>8</v></c></row>'))
            self.assertEqual(list(xlsx_rows(p)), [["7", "8"]])

    def test_inline_string(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_inline("A1", "WHITE LANTERN")])))
            self.assertEqual(list(xlsx_rows(p)), [["WHITE LANTERN"]])

    def test_empty_row_yields_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet('<row r="1"></row>' + _row(2, [_num("A2", 5)])))
            self.assertEqual(list(xlsx_rows(p)), [[], ["5"]])

    def test_shared_string_index_out_of_range_is_an_error(self):
        # 越界说明文件坏了。宁愿报错，也不要安静地返回空字符串 ——
        # 那会让整列商品名变空，而下游只会觉得"这批数据描述缺失"。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx", shared=("only",),
                            sheet_xml=_sheet(_row(1, [_shared("A1", 9)])))
            with self.assertRaises(DatasetError):
                list(xlsx_rows(p))

    def test_error_cell_is_an_error_not_a_guess(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, ['<c r="A1" t="e"><v>#REF!</v></c>'])))
            with self.assertRaises(DatasetError):
                list(xlsx_rows(p))

    def test_not_a_zip_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nope.xlsx"
            p.write_bytes(b"definitely not a zip")
            with self.assertRaises(DatasetError):
                list(xlsx_rows(p))

    def test_missing_file_is_an_error(self):
        with self.assertRaises(DatasetError):
            list(xlsx_rows(Path("no/such/file.xlsx")))

    def test_workbook_without_worksheet_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.xlsx"
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr("xl/workbook.xml", _WORKBOOK)
            with self.assertRaises(DatasetError):
                list(xlsx_rows(p))

    def test_ambiguous_multiple_sheets_is_an_error(self):
        """有多个工作表又定位不到数据表时，**报错而不是猜**。

        猜错的后果是安静地读错一张表：行数、列名可能都对得上，只有内容不是你要的。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.xlsx"
            workbook = ('<?xml version="1.0"?>'
                        f'<workbook xmlns="{_MAIN}"><sheets>'
                        '<sheet name="A" sheetId="1"/>'
                        '<sheet name="B" sheetId="2"/></sheets></workbook>')
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr("xl/workbook.xml", workbook)
                zf.writestr("xl/worksheets/sheet1.xml", _sheet(_row(1, [_num("A1", 1)])))
                zf.writestr("xl/worksheets/sheet2.xml", _sheet(_row(1, [_num("A1", 2)])))
            with self.assertRaises(DatasetError) as cm:
                list(xlsx_rows(p))
            self.assertIn("2 个工作表", str(cm.exception))

    def test_sheet_is_found_through_rels_not_by_guessing(self):
        # 工作表不叫 sheet1.xml 时，必须靠 workbook → rels 找到它。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(
                Path(d) / "t.xlsx",
                sheet_xml="",  # 占位，随后被覆盖
                rels=(f'<?xml version="1.0"?><Relationships xmlns="{_PKG}">'
                      f'<Relationship Id="rId1" Type="{_REL}/worksheet" '
                      f'Target="worksheets/data000.xml"/></Relationships>'),
                extra={"xl/worksheets/data000.xml": _sheet(_row(1, [_num("A1", 42)]))},
            )
            self.assertEqual(list(xlsx_rows(p)), [["42"]])

    def test_is_lazy(self):
        # 真实的 sheet1.xml 解压后 184 MB，一次性加载会吃掉一两个 GB。
        # 断言它是生成器：取一行就返回，而不是先把整份读进内存。
        with tempfile.TemporaryDirectory() as d:
            p = _write_xlsx(Path(d) / "t.xlsx",
                            sheet_xml=_sheet(_row(1, [_num("A1", 1)]) + _row(2, [_num("A2", 2)])))
            gen = xlsx_rows(p)
            try:
                self.assertEqual(next(gen), ["1"])
            finally:
                # 生成器还握着 zip 的句柄。Windows 上不关掉，临时目录就删不掉 ——
                # 在 Linux 上这条会静默通过，所以显式关。
                gen.close()


# ── xlsx → CSV ─────────────────────────────────────────────────────────

class TestXlsxToCsv(unittest.TestCase):
    def test_date_column_is_converted_and_row_count_returned(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(
                d / "t.xlsx",
                shared=("InvoiceNo", "InvoiceDate"),
                sheet_xml=_sheet(
                    _row(1, [_shared("A1", 0), _shared("B1", 1)])
                    + _row(2, [_num("A2", 536365), _num("B2", "40513.351388888892")])
                ),
            )
            n = xlsx_to_csv(src, d / "out.csv", date_columns=("InvoiceDate",))
            self.assertEqual(n, 1)
            text = (d / "out.csv").read_text(encoding="utf-8")
            self.assertIn("2010-12-01 08:26:00", text)
            self.assertNotIn("40513", text)

    def test_non_date_columns_are_left_untouched(self):
        # 只有显式点名的列才转。没点名就原样 —— 猜"哪列是日期"迟早会猜错。
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(d / "t.xlsx",
                              sheet_xml=_sheet(_row(1, [_num("A1", "40513.35")])))
            xlsx_to_csv(src, d / "out.csv")
            self.assertIn("40513.35", (d / "out.csv").read_text(encoding="utf-8"))

    def test_missing_date_column_is_an_error(self):
        # 列名变了必须炸出来。默默跳过 = 整列日期变成一串数字，且不报错。
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(d / "t.xlsx", shared=("A",),
                              sheet_xml=_sheet(_row(1, [_shared("A1", 0)])))
            with self.assertRaises(DatasetError) as cm:
                xlsx_to_csv(src, d / "out.csv", date_columns=("InvoiceDate",))
            self.assertIn("InvoiceDate", str(cm.exception))

    def test_short_rows_are_padded_to_header_width(self):
        # 尾部单元格被省略时，行会短于表头。补齐成空串，而不是让列数参差。
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(d / "t.xlsx", shared=("a", "b", "c"),
                              sheet_xml=_sheet(
                                  _row(1, [_shared("A1", 0), _shared("B1", 1), _shared("C1", 2)])
                                  + _row(2, [_num("A2", 1)])))
            xlsx_to_csv(src, d / "out.csv")
            lines = (d / "out.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[1].count(","), 2)

    def test_unparseable_date_is_kept_and_reported(self):
        # 转不了的日期按原样写出（下游还会再试一次），但要**在 stderr 说一句** ——
        # 整列没转成功的话最终行数闸门会拦住，这里先把线索递出去。
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(d / "t.xlsx", shared=("InvoiceDate",),
                              sheet_xml=_sheet(
                                  _row(1, [_shared("A1", 0)])
                                  + _row(2, [_inline("A2", "not-a-date")])))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                xlsx_to_csv(src, d / "out.csv", date_columns=("InvoiceDate",))
            self.assertIn("not-a-date", (d / "out.csv").read_text(encoding="utf-8"))
            self.assertIn("没能按 Excel 序列号解析", err.getvalue())

    def test_empty_sheet_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(d / "t.xlsx", sheet_xml=_sheet(""))
            with self.assertRaises(DatasetError):
                xlsx_to_csv(src, d / "out.csv")

    def test_output_is_utf8_so_the_bom_is_not_needed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = _write_xlsx(d / "t.xlsx", shared=("描述",),
                              sheet_xml=_sheet(_row(1, [_shared("A1", 0)])))
            xlsx_to_csv(src, d / "out.csv")
            self.assertEqual((d / "out.csv").read_text(encoding="utf-8").strip(), "描述")


# ── 解压与下载 ──────────────────────────────────────────────────────────

class TestExtractSingle(unittest.TestCase):
    def _zip_with(self, path: Path, names: list[str]) -> Path:
        with zipfile.ZipFile(path, "w") as zf:
            for n in names:
                zf.writestr(n, b"x")
        return path

    def test_single_file_is_extracted(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            z = self._zip_with(d / "a.zip", ["dir/Online Retail.xlsx"])
            got = extract_single(z, d / "out")
            self.assertEqual(got.name, "Online Retail.xlsx")
            self.assertTrue(got.exists())

    def test_multiple_candidates_is_an_error(self):
        # 猜哪一个是数据文件，就是在赌 —— 宁可让人看一眼压缩包内容。
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            z = self._zip_with(d / "a.zip", ["a.xlsx", "b.xlsx"])
            with self.assertRaises(DatasetError) as cm:
                extract_single(z, d / "out")
            self.assertIn("2 个", str(cm.exception))

    def test_no_candidate_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            z = self._zip_with(d / "a.zip", ["readme.txt"])
            with self.assertRaises(DatasetError):
                extract_single(z, d / "out")


class _FakeResp:
    def __init__(self, payload: bytes, headers: dict | None = None):
        self._b = io.BytesIO(payload)
        self.headers = headers if headers is not None else {
            "Content-Length": str(len(payload))}

    def read(self, n: int = -1) -> bytes:
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _zip_bytes() -> bytes:
    """一个**内容正确**的 UCI 压缩包替身：zip 里装一个真能解析的 xlsx。

    故意不是随手几个字节 —— 因为有些调用路径会真的继续读下去，
    塞假内容会让测试在错误的层失败（看起来像下载逻辑坏了，其实是夹具坏了）。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Online Retail.xlsx", _xlsx_bytes(
            shared=("InvoiceDate",),
            sheet_xml=_sheet(
                _row(1, [_shared("A1", 0)])
                + _row(2, [_num("A2", "40513.351388888892")]))))
    return buf.getvalue()


class _DroppingResp(_FakeResp):
    """模拟**连接在传输中途断掉**。

    抛的是 `http.client.IncompleteRead`（不是 OSError）—— 这正是 UCI 在
    chunked 编码下断流的真实表现，也是那个新 clone 故障的原始形态。
    默认不带 Content-Length，因为 UCI 实测就是不带。
    """

    def __init__(self, payload: bytes, headers: dict | None = None, after: int = 1):
        super().__init__(payload, headers={} if headers is None else headers)
        self._calls = 0
        self._after = after

    def read(self, n: int = -1) -> bytes:
        self._calls += 1
        if self._calls <= self._after:
            return self._b.read(n)
        raise http.client.IncompleteRead(b"partial")


class TestDownloadZip(unittest.TestCase):
    def test_existing_file_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "online+retail.zip"
            dest.write_bytes(_zip_bytes())
            with mock.patch("replayprobe.dataset.urllib.request.urlopen") as m:
                self.assertEqual(download_zip(dest), dest)
                m.assert_not_called()

    def test_successful_download_replaces_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "sub" / "online+retail.zip"
            payload = _zip_bytes()
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            return_value=_FakeResp(payload)):
                got = download_zip(dest, url="https://example.invalid/x.zip")
            self.assertEqual(got.read_bytes(), payload)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_html_error_page_is_rejected(self):
        # 下载"成功"但拿到的其实是个错误页 —— 这是最典型的静默失败：
        # 如果不验，zipfile 后面会读得莫名其妙。
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            return_value=_FakeResp(b"<html>404</html>")):
                with self.assertRaises(DatasetError):
                    download_zip(dest, url="https://example.invalid/x.zip")
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_truncated_download_is_rejected(self):
        # 声明 1 万字节却收不到 —— 判为截断。截断也是瞬时的，所以它会走重试路径，
        # 重试用尽才报错。这里注入空 sleep，否则测试要真的睡 2+4 秒。
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            payload = _zip_bytes()
            headers = {"Content-Length": str(len(payload) + 10_000)}
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            return_value=_FakeResp(payload, headers)) as m:
                with self.assertRaises(DatasetError) as cm:
                    download_zip(dest, url="https://example.invalid/x.zip",
                                 sleep=lambda _: None)
            self.assertIn("不完整", str(cm.exception))
            self.assertEqual(m.call_count, 3)      # 重试了 3 次
            self.assertFalse(dest.exists())

    def test_network_error_is_wrapped_with_actionable_hint(self):
        import urllib.error
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            side_effect=urllib.error.URLError("boom")):
                with self.assertRaises(DatasetError) as cm:
                    download_zip(dest, url="https://example.invalid/x.zip",
                                 retries=1)
            # 报错必须告诉人"还能怎么办"，而不是只把异常原样抛上去。
            self.assertIn("手动下载", str(cm.exception))

    # ── 下面这一组，全部来自一次「新 clone 实测」逼出来的真实故障 ──────────
    #
    # 现象：全新 clone 后跑 README 第一步，UCI 下载到 0.5 MB 时连接断掉，
    # 抛 http.client.IncompleteRead 直接穿透出去，磁盘上留下一个
    # 正好 1 MB（一个 chunk）的 online+retail.zip.part。

    def test_incomplete_read_is_not_an_oserror(self):
        """把根因钉住：这就是原先 except 抓不住它的原因。

        这段断言不是在测产品代码，是在守住一个**容易再次踩进去的假设** ——
        "网络异常都是 OSError"。IncompleteRead 不是，所以漏了它。
        """
        self.assertFalse(issubclass(http.client.IncompleteRead, OSError))
        self.assertTrue(issubclass(http.client.IncompleteRead,
                                   http.client.HTTPException))

    def test_incomplete_read_is_retried_then_succeeds(self):
        """断一次不算失败 —— 第二次成功就应当静默恢复，不该惊动使用者。"""
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            payload = _zip_bytes()
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            side_effect=[_DroppingResp(payload),
                                         _FakeResp(payload)]) as m:
                got = download_zip(dest, url="https://example.invalid/x.zip",
                                   sleep=lambda _: None)
            self.assertEqual(m.call_count, 2)
            self.assertEqual(got.read_bytes(), payload)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_persistent_failure_leaves_no_residue(self):
        """重试用尽后：抛可操作的错、并且**不留半截文件**。

        留残留物是这次真故障的第二个后果 —— 那个 1 MB 的 .part 一直躺在
        data/raw/ 里，下一个人看到只会怀疑"到底下载了没有"。
        """
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            side_effect=lambda *a, **k: _DroppingResp(_zip_bytes())):
                with self.assertRaises(DatasetError) as cm:
                    download_zip(dest, url="https://example.invalid/x.zip",
                                 sleep=lambda _: None)
            msg = str(cm.exception)
            self.assertIn("IncompleteRead", msg)   # 说清是什么断了
            self.assertIn("手动下载", msg)          # 说清还能怎么办
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name + ".part").exists())

    def test_incomplete_read_without_content_length_is_still_caught(self):
        """UCI 是 chunked 编码（没有 Content-Length），所以长度校验根本不参与 ——
        此时唯一的防线是抛出来的异常本身。这条把"没有 total 也得能兜住"钉住。"""
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            with mock.patch(
                    "replayprobe.dataset.urllib.request.urlopen",
                    side_effect=lambda *a, **k: _DroppingResp(_zip_bytes(),
                                                             headers={})):
                with self.assertRaises(DatasetError):
                    download_zip(dest, url="https://example.invalid/x.zip",
                                 retries=1)

    def test_stale_part_file_is_cleaned_before_retrying(self):
        """上一次崩在半路留下的 .part，这一次开头就要清掉。"""
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            stale = dest.with_name(dest.name + ".part")
            stale.write_bytes(b"garbage from a previous crash")
            payload = _zip_bytes()
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            return_value=_FakeResp(payload)):
                download_zip(dest, url="https://example.invalid/x.zip")
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(stale.exists())

    def test_retries_can_be_turned_down(self):
        """retries=1 就是"只试一次" —— 也保证测试和 CI 不会为了重试空等。"""
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            side_effect=urllib.error.URLError("boom")) as m:
                with self.assertRaises(DatasetError):
                    download_zip(dest, url="https://example.invalid/x.zip",
                                 retries=1)
            self.assertEqual(m.call_count, 1)

    def test_backoff_grows_between_attempts(self):
        """退避必须是递增的，不然"重试"就只是"更用力地撞同一堵墙"。"""
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "x.zip"
            waits: list[float] = []
            with mock.patch("replayprobe.dataset.urllib.request.urlopen",
                            side_effect=urllib.error.URLError("boom")):
                with self.assertRaises(DatasetError):
                    download_zip(dest, url="https://example.invalid/x.zip",
                                 retries=3, backoff=2.0, sleep=waits.append)
            self.assertEqual(len(waits), 2)         # 3 次尝试之间睡 2 次
            self.assertLess(waits[0], waits[1])


class TestFetchOnlineRetail(unittest.TestCase):
    def test_existing_csv_short_circuits_everything(self):
        """已有 CSV 就一步都不做 —— 尤其是**不打网络**。

        这是第二次跑能秒回的原因，也是"离线也能重建真值库"的前提。
        """
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)
            (raw / "online_retail.csv").write_text("InvoiceNo\n1\n", encoding="utf-8")
            with mock.patch("replayprobe.dataset.urllib.request.urlopen") as m:
                got = fetch_online_retail(raw)
                m.assert_not_called()
            self.assertEqual(got, raw / "online_retail.csv")

    def test_force_redownloads(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)
            (raw / "online_retail.csv").write_text("InvoiceNo\n1\n", encoding="utf-8")
            with mock.patch("replayprobe.dataset.urllib.request.urlopen") as m:
                m.return_value = _FakeResp(_zip_bytes())
                fetch_online_retail(raw, force=True)
                m.assert_called_once()


# ── 源 CSV 编码 ─────────────────────────────────────────────────────────

class TestSniffEncoding(unittest.TestCase):
    """顺序反了就会**静默出错**：ISO-8859-1 解码永不失败，所以不能先试它。"""

    def test_utf8_with_non_ascii(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.csv"
            p.write_bytes("产品,编号\n台灯,1\n".encode("utf-8"))
            self.assertEqual(sniff_encoding(p), "utf-8")

    def test_latin1_bytes_are_detected(self):
        # 0xE9 在 UTF-8 里是个非法的孤立字节，必须被识别出来回落。
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.csv"
            p.write_bytes("café,1\n".encode("ISO-8859-1"))
            self.assertEqual(sniff_encoding(p), "ISO-8859-1")

    def test_pure_ascii_is_judged_utf8_and_that_is_fine(self):
        # 全 ASCII 时两种编码解码结果完全相同，判成哪个都无所谓 ——
        # 这条测试把这个"无所谓"固定下来，免得以后有人当 bug 改。
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.csv"
            p.write_bytes(b"InvoiceNo,Quantity\n536365,6\n")
            self.assertEqual(sniff_encoding(p), "utf-8")

    def test_multibyte_split_across_chunk_boundary_is_not_misjudged(self):
        """头部试读的经典误判：一个多字节字符正好被切在两块之间。

        所以实现用的是增量解码器，而不是"读前 N 字节试一下"。
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.csv"
            # 造一个 > 1 MB 的 UTF-8 文件，让字符必然跨越 1 MB 的块边界
            line = "商品名称,数量\n"
            body = (line * (1 << 17)).encode("utf-8") + "英国,1\n".encode("utf-8")
            p.write_bytes(body)
            self.assertEqual(sniff_encoding(p), "utf-8")


if __name__ == "__main__":
    unittest.main()
