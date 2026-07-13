from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#18212B")
MUTED = colors.HexColor("#586474")
ACCENT = colors.HexColor("#167D73")
RULE = colors.HexColor("#D7DDE3")
TABLE_HEADER = colors.HexColor("#E7F2F0")
TABLE_ALT = colors.HexColor("#F6F8FA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the FREUID Markdown report as a submission PDF.")
    parser.add_argument("--input", default="reports/freuid_short_report_2026_07_13.md")
    parser.add_argument("--output", default="output/pdf/freuid_short_report_2026_07_13.pdf")
    return parser.parse_args()


def _inline_markup(value: str) -> str:
    escaped = html.escape(value.strip())
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return escaped


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=7 * mm,
        ),
        "h2": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=ACCENT,
            spaceBefore=5 * mm,
            spaceAfter=2.3 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Subsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=INK,
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=12.4,
            textColor=INK,
            spaceAfter=2.2 * mm,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=12,
            textColor=INK,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            spaceAfter=1.2 * mm,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.8,
            leading=10.2,
            textColor=INK,
            leftIndent=3 * mm,
            rightIndent=3 * mm,
            borderColor=RULE,
            borderWidth=0.5,
            borderPadding=4,
            backColor=colors.HexColor("#F4F6F8"),
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9,
            textColor=INK,
        ),
    }


def _table(lines: list[str], styles: dict[str, Paragraph], available_width: float) -> Table:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0], *rows[2:]]
    formatted = [
        [Paragraph(_inline_markup(cell), styles["table_header"] if index == 0 else styles["table"]) for cell in row]
        for index, row in enumerate(rows)
    ]
    columns = len(formatted[0])
    table = Table(formatted, colWidths=[available_width / columns] * columns, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(2, len(formatted), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), TABLE_ALT))
    table.setStyle(TableStyle(commands))
    return table


def _story(markdown: str, available_width: float):
    styles = _styles()
    lines = markdown.splitlines()
    story = []
    index = 0
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            story.append(Paragraph(_inline_markup(" ".join(paragraph_lines)), styles["body"]))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            index += 1
            continue
        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_inline_markup(line[2:]), styles["title"]))
            index += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            section_title = line[3:]
            if section_title == "Results":
                story.append(PageBreak())
            story.append(Paragraph(_inline_markup(section_title), styles["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_inline_markup(line[4:]), styles["h3"]))
            index += 1
            continue
        if line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(f"<b>-</b>&nbsp;&nbsp;{_inline_markup(line[2:])}", styles["bullet"]))
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1]):
            flush_paragraph()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.extend([KeepTogether([_table(table_lines, styles, available_width)]), Spacer(1, 3 * mm)])
            continue
        paragraph_lines.append(line)
        index += 1
    flush_paragraph()
    return story


def build_pdf(markdown_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    left = right = 18 * mm
    top = 18 * mm
    bottom = 17 * mm
    page_width, page_height = A4
    frame = Frame(left, bottom, page_width - left - right, page_height - top - bottom, id="content")

    def header_footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(left, page_height - 12 * mm, page_width - right, page_height - 12 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, page_height - 9.5 * mm, "FREUID Challenge 2026 - Frozen Technical Report")
        canvas.drawRightString(page_width - right, 9.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="Type-Adversarial High-Resolution Forensics for FREUID 2026",
        author="FREUID Challenge Team",
    )
    document.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=header_footer)])
    available_width = page_width - left - right
    document.build(_story(markdown_path.read_text(encoding="utf-8"), available_width))


def main() -> None:
    args = parse_args()
    build_pdf(Path(args.input), Path(args.output))
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
