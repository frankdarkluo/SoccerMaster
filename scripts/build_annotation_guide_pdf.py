#!/usr/bin/env python3
"""把标注规范的 Markdown 源渲染成中文 PDF。

用法：
    python3 scripts/build_annotation_guide_pdf.py \
        data/annotations/足球战术人工标注注意事项_v2.md \
        data/annotations/足球战术人工标注注意事项_v2.pdf

只支持本项目实际用到的 Markdown 子集：标题、段落、无序列表、表格、分隔线，
以及行内的 **强调** 与 `代码`。

字体说明：本机唯一带中文字形的字体是 Droid Sans Fallback。它是纯回退字体，
cmap 里 **完全没有 ASCII**（只有 CJK 与全角标点），也没有粗体字面。所以：

* **强调** 用重音色渲染，不用伪粗体；
* 逐字符路由字体：CJK 走 Droid（嵌入），拉丁与常见符号走标准 14 字体 Helvetica，
  两者都没有的符号退到 Noto Mono（嵌入）。
* 构建时会检查有没有三个字体都渲染不了的字符，有就报错，避免出现空白。
"""

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

SYMBOL_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"

FONT = "CJK"
LATIN = "Helvetica"                   # 标准 14 字体，任何阅读器自带，无需嵌入
SYMBOL = "SYM"
ACCENT_HEX = "#12395c"                # 强调色，代替粗体
CODE_HEX = "#8a3a12"                  # 行内代码色
ACCENT = colors.HexColor(ACCENT_HEX)
CODE = colors.HexColor(CODE_HEX)
RULE = colors.HexColor("#c8d2da")
HEAD_BG = colors.HexColor("#eef2f6")

_cjk_cmap: dict = {}
_sym_cmap: dict = {}


def register_font() -> None:
    global _cjk_cmap, _sym_cmap
    for path in CJK_FONT_CANDIDATES:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(FONT, path))
            # 没有真粗体字面，把 bold/italic 都指回同一个字面，避免 <b> 触发缺字。
            pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)
            _cjk_cmap = pdfmetrics.getFont(FONT).face.charToGlyph
            break
    else:
        raise SystemExit("找不到可用的中文字体，请检查 CJK_FONT_CANDIDATES")

    if Path(SYMBOL_FONT_PATH).exists():
        pdfmetrics.registerFont(TTFont(SYMBOL, SYMBOL_FONT_PATH))
        _sym_cmap = pdfmetrics.getFont(SYMBOL).face.charToGlyph


def font_for(ch: str) -> str:
    """逐字符选字体。Droid 没有 ASCII，Helvetica 没有 CJK，必须路由。"""
    if ord(ch) in _cjk_cmap:
        return FONT
    try:
        ch.encode("cp1252")            # Helvetica 的 WinAnsi 覆盖范围
        return LATIN
    except UnicodeEncodeError:
        pass
    if ord(ch) in _sym_cmap:
        return SYMBOL
    return ""                          # 三个字体都渲染不了


def route(text: str, missing: set) -> str:
    """把一段纯文本按字体切成若干 <font name=...> 片段。"""
    out = []
    run: list = []
    run_font = None
    for ch in text:
        f = font_for(ch)
        if not f:
            missing.add(ch)
            continue
        if f != run_font:
            if run:
                out.append(f'<font name="{run_font}">{html.escape("".join(run))}</font>')
            run, run_font = [], f
        run.append(ch)
    if run:
        out.append(f'<font name="{run_font}">{html.escape("".join(run))}</font>')
    return "".join(out)


def styles() -> dict:
    # 项目符号 • 不在 Droid 的 cmap 里，必须用拉丁字体画。
    base = dict(fontName=FONT, bulletFontName=LATIN, alignment=TA_LEFT, wordWrap="CJK")
    return {
        "title": ParagraphStyle("title", fontSize=15, leading=20, spaceAfter=2,
                                textColor=ACCENT, **base),
        "sub": ParagraphStyle("sub", fontSize=8, leading=12, spaceAfter=8,
                              textColor=colors.HexColor("#5a6a78"), **base),
        "h1": ParagraphStyle("h1", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3,
                             textColor=ACCENT, **base),
        "h2": ParagraphStyle("h2", fontSize=9.2, leading=12.5, spaceBefore=6, spaceAfter=2.5,
                             textColor=ACCENT, **base),
        "body": ParagraphStyle("body", fontSize=8.2, leading=11.6, spaceAfter=2.6, **base),
        "bullet": ParagraphStyle("bullet", fontSize=8.2, leading=11.6, spaceAfter=1.8,
                                 leftIndent=10, bulletIndent=2, **base),
        "cell": ParagraphStyle("cell", fontSize=7.4, leading=10.0, **base),
        "cellhead": ParagraphStyle("cellhead", fontSize=7.4, leading=10.0,
                                   textColor=ACCENT, **base),
    }


MISSING: set = set()


def inline(text: str) -> str:
    """行内标记 → reportlab 的迷你 HTML（含逐字符字体路由）。

    用扫描而不是正则，因为 `代码` 经常嵌在 **强调** 里面，正则的两个分支互相吃不到。
    """
    spans: list = []
    buf: list = []
    bold = False
    i = 0

    def flush():
        if buf:
            spans.append(("".join(buf), ACCENT_HEX if bold else None))
            buf.clear()

    while i < len(text):
        if text.startswith("**", i):
            flush()
            bold = not bold
            i += 2
            continue
        if text[i] == "`":
            close = text.find("`", i + 1)
            if close > i:
                flush()
                spans.append((text[i + 1:close], CODE_HEX))
                i = close + 1
                continue
        buf.append(text[i])
        i += 1
    flush()

    out = []
    for chunk, color in spans:
        if not chunk:
            continue
        routed = route(chunk, MISSING)
        out.append(f'<font color="{color}">{routed}</font>' if color else routed)
    return "".join(out)


def split_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def build_table(rows: list, st: dict, width: float) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    if ncols == 2:
        widths = [width * 0.28, width * 0.72]
    elif ncols == 3:
        widths = [width * 0.22, width * 0.30, width * 0.48]
    else:
        widths = [width / ncols] * ncols

    data = [[Paragraph(inline(c), st["cellhead"]) for c in header]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("INNERGRID", (0, 1), (-1, -1), 0.25, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def parse(md: str, st: dict, width: float) -> list:
    story = []
    lines = md.splitlines()
    i = 0
    first_heading = True

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [split_row(b) for b in block
                    if not re.fullmatch(r"[|\s:\-]+", b.strip())]
            if rows:
                story.append(Spacer(1, 2))
                story.append(build_table(rows, st, width))
                story.append(Spacer(1, 4))
            continue

        if re.fullmatch(r"-{3,}", line.strip()):
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                    spaceBefore=0, spaceAfter=5))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h2"]))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), st["h1"]))
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), st["title"] if first_heading else st["h1"]))
            first_heading = False
            i += 1
            continue

        if line.startswith("- "):
            story.append(Paragraph(inline(line[2:]), st["bullet"], bulletText="•"))
            i += 1
            continue

        story.append(Paragraph(inline(line), st["body"]))
        i += 1

    return story


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    register_font()
    st = styles()

    margin = 1.6 * cm
    width = A4[0] - 2 * margin

    doc = SimpleDocTemplate(
        str(dst), pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="足球战术人工标注注意事项 v2", author="SoccerMaster",
    )

    def footer(canvas, _doc):
        # 页码用拉丁字体：Droid 的 cmap 里没有阿拉伯数字。
        canvas.saveState()
        canvas.setFont(LATIN, 7.5)
        canvas.setFillColor(colors.HexColor("#8a97a3"))
        canvas.drawCentredString(A4[0] / 2, 0.9 * cm, f"- {canvas.getPageNumber()} -")
        canvas.restoreState()

    story = parse(src.read_text(encoding="utf-8"), st, width)
    if MISSING:
        raise SystemExit(
            "以下字符三个字体都渲染不了，会在 PDF 里变成空白，请在 Markdown 源里替换：\n  "
            + " ".join(f"{c!r}(U+{ord(c):04X})" for c in sorted(MISSING))
        )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"已生成 {dst}")


if __name__ == "__main__":
    main()
