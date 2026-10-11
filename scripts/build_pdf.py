"""Convert reports/project_report.md to reports/project_report.pdf using
reportlab and markdown2.

Usage:
    python scripts/build_pdf.py

The script:
1. Parses the Markdown to HTML via markdown2 (tables, images, code blocks).
2. Lays out the HTML onto A4 pages using reportlab's Platypus framework.
3. Embeds figures from reports/figures/ as inline images.
4. Applies a clean academic style (serif body font, page numbers, headers).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from typing import Any

import markdown2

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT        = Path(__file__).resolve().parent.parent
REPORT_MD   = ROOT / "reports" / "project_report.md"
REPORT_PDF  = ROOT / "reports" / "project_report.pdf"
FIGURES_DIR = ROOT / "reports" / "figures"

PAGE_W, PAGE_H = A4
MARGIN = 2.5 * cm

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def build_styles() -> dict[str, ParagraphStyle]:
    """Build a dict of named paragraph styles for the report.

    Returns:
        Dict mapping style name → ParagraphStyle.
    """
    base = getSampleStyleSheet()

    styles: dict[str, ParagraphStyle] = {
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=18, leading=22, spaceAfter=12, spaceBefore=6,
            textColor=colors.HexColor("#1A237E"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=14, leading=18, spaceAfter=8, spaceBefore=14,
            textColor=colors.HexColor("#283593"),
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"],
            fontSize=12, leading=16, spaceAfter=6, spaceBefore=10,
            textColor=colors.HexColor("#1565C0"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10.5, leading=15, spaceAfter=6,
            alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=9, leading=12, spaceAfter=8, spaceBefore=2,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            italics=True,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"],
            fontSize=8.5, leading=12, spaceAfter=6, spaceBefore=4,
            fontName="Courier",
            backColor=colors.HexColor("#F5F5F5"),
            leftIndent=12, rightIndent=12,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"],
            fontSize=10.5, leading=14, spaceAfter=3,
            leftIndent=16, bulletIndent=6,
            alignment=TA_LEFT,
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=4,
            textColor=colors.HexColor("#333333"),
            alignment=TA_CENTER,
        ),
        "th": ParagraphStyle(
            "th", parent=base["Normal"],
            fontSize=9, leading=12,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "td", parent=base["Normal"],
            fontSize=9, leading=12,
            fontName="Helvetica",
        ),
        "ref": ParagraphStyle(
            "ref", parent=base["Normal"],
            fontSize=9.5, leading=14, spaceAfter=5,
            leftIndent=20, firstLineIndent=-20,
            alignment=TA_JUSTIFY,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

def _header_footer(canvas: Any, doc: Any) -> None:
    """Draw page header (title) and footer (page number) on every page.

    Args:
        canvas: ReportLab canvas for the current page.
        doc: The document being built.
    """
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(colors.HexColor("#1A237E"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 4*mm,
                PAGE_W - MARGIN, PAGE_H - MARGIN + 4*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 5*mm,
                      "Employee Attrition Prediction — Final-Year Project Report")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 5*mm,
                           "AI & Data Science")

    # Footer
    canvas.line(MARGIN, MARGIN - 4*mm, PAGE_W - MARGIN, MARGIN - 4*mm)
    canvas.drawCentredString(PAGE_W / 2, MARGIN - 9*mm, f"— {doc.page} —")
    canvas.restoreState()


def build_doc(output_path: Path) -> BaseDocTemplate:
    """Create the BaseDocTemplate with a single page template.

    Args:
        output_path: Path where the PDF will be written.

    Returns:
        Configured BaseDocTemplate ready for build.
    """
    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 8*mm, bottomMargin=MARGIN + 8*mm,
    )
    frame = Frame(
        MARGIN, MARGIN + 8*mm,
        PAGE_W - 2*MARGIN, PAGE_H - 2*MARGIN - 16*mm,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame],
                            onPage=_header_footer)
    doc.addPageTemplates([template])
    return doc


# ---------------------------------------------------------------------------
# Markdown → Flowables parser
# ---------------------------------------------------------------------------

# Regex to detect image captions (italic text after an image line)
_IMG_RE  = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_REF_RE  = re.compile(r'^\[(\d+)\]')

def _resolve_image(src: str) -> Path | None:
    """Resolve an image path relative to the report's figures directory.

    Args:
        src: Image source path from the Markdown.

    Returns:
        Resolved absolute Path if the file exists, else None.
    """
    candidate = ROOT / "reports" / src
    if candidate.exists():
        return candidate
    candidate2 = FIGURES_DIR / Path(src).name
    if candidate2.exists():
        return candidate2
    return None


def _escape_html(text: str) -> str:
    """Escape characters that break Paragraph XML in reportlab.

    Args:
        text: Raw text string.

    Returns:
        XML-safe string.
    """
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def _clean_inline(text: str) -> str:
    """Convert common Markdown inline marks to reportlab XML tags.

    Applies substitutions in safe order: code spans first (protecting their
    content from italic/bold replacement), then bold, then italic.

    Args:
        text: Inline Markdown text (already HTML-escaped).

    Returns:
        Text with <b>, <i>, <font name="Courier"> tags, properly nested.
    """
    # Step 1: Extract code spans and replace with placeholders so their
    # content is not affected by bold/italic substitutions.
    code_spans: list[str] = []

    def save_code(m: re.Match) -> str:
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans)-1}\x00"

    text = re.sub(r'`([^`]+)`', save_code, text)

    # Step 2: Bold + italic (must come before separate bold/italic)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    # Step 3: Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Step 4: Italic via asterisk only — skip underscore to avoid false positives
    # inside identifiers like scale_pos_weight
    text = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', text)

    # Step 5: Restore code spans as <font name="Courier">...</font>
    for idx, span in enumerate(code_spans):
        text = text.replace(
            f"\x00CODE{idx}\x00",
            f'<font name="Courier">{span}</font>',
        )

    return text


def md_to_flowables(md_text: str, styles: dict) -> list:
    """Parse a Markdown string into a list of reportlab Flowable objects.

    Handles: headings (H1-H3), paragraphs, bullet lists, fenced code blocks,
    pipe tables, image + caption pairs, horizontal rules, and bold/italic/code
    inline formatting.

    Args:
        md_text: Raw Markdown source.
        styles: Dict of named ParagraphStyle objects.

    Returns:
        List of reportlab Flowables ready to pass to doc.build().
    """
    flowables: list = []
    lines = md_text.split("\n")
    i = 0
    in_code = False
    code_lines: list[str] = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            block = "\n".join(code_lines)
            # Split into lines for the code style
            for cl in block.split("\n"):
                if cl.strip():
                    flowables.append(
                        Paragraph(_escape_html(cl), styles["code"])
                    )
            flowables.append(Spacer(1, 4))
            code_lines = []

    while i < len(lines):
        line = lines[i]

        # ---- Fenced code block ----
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
            else:
                in_code = False
                flush_code()
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # ---- Horizontal rule ----
        if re.match(r'^[-*]{3,}$', line.strip()):
            flowables.append(Spacer(1, 4))
            flowables.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#BDBDBD"), spaceAfter=4,
            ))
            i += 1
            continue

        # ---- Headings ----
        h_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if h_match:
            level = len(h_match.group(1))
            text  = _clean_inline(_escape_html(h_match.group(2).strip()))
            key   = f"h{min(level, 3)}"
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(text, styles[key]))
            i += 1
            continue

        # ---- Pipe table ----
        if line.strip().startswith("|") and "|" in line:
            # Collect all consecutive pipe-table lines
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # Filter separator rows
            rows = [r for r in table_lines
                    if not re.match(r'^\|[-| :]+\|$', r.strip())]
            if rows:
                data = []
                for row_idx, row in enumerate(rows):
                    cells = [c.strip() for c in row.strip().strip("|").split("|")]
                    style_key = "th" if row_idx == 0 else "td"
                    data.append([
                        Paragraph(_clean_inline(_escape_html(c)),
                                  styles[style_key])
                        for c in cells
                    ])
                col_count = max(len(r) for r in data)
                avail_w = PAGE_W - 2 * MARGIN
                col_w = avail_w / col_count
                col_widths = [col_w] * col_count

                t = Table(data, colWidths=col_widths, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                    ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#F3F4FF")]),
                    ("GRID",       (0, 0), (-1, -1), 0.3,
                     colors.HexColor("#BDBDBD")),
                    ("VALIGN",     (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
                ]))
                flowables.append(Spacer(1, 6))
                flowables.append(t)
                flowables.append(Spacer(1, 8))
            continue

        # ---- Image ----
        img_match = _IMG_RE.match(line.strip())
        if img_match:
            alt_text = img_match.group(1)
            src      = img_match.group(2)
            img_path = _resolve_image(src)
            if img_path:
                max_w = PAGE_W - 2 * MARGIN
                max_h = 8 * cm
                flowables.append(Spacer(1, 6))
                try:
                    img = Image(str(img_path), width=max_w, height=max_h,
                                kind="proportional")
                    flowables.append(img)
                    # Caption on the next line (italic text)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith("*") and next_line.endswith("*"):
                            caption_text = next_line.strip("*").strip()
                            flowables.append(
                                Paragraph(
                                    _escape_html(caption_text),
                                    styles["caption"],
                                )
                            )
                            i += 1  # skip caption line
                except Exception:
                    flowables.append(
                        Paragraph(f"[Image: {alt_text}]", styles["caption"])
                    )
                flowables.append(Spacer(1, 6))
            else:
                flowables.append(
                    Paragraph(f"[Image not found: {src}]", styles["caption"])
                )
            i += 1
            continue

        # ---- Bullet / unordered list ----
        if re.match(r'^[-*+]\s+', line.strip()):
            text = re.sub(r'^[-*+]\s+', '', line.strip())
            text = _clean_inline(_escape_html(text))
            flowables.append(Paragraph(f"• {text}", styles["bullet"]))
            i += 1
            continue

        # ---- Numbered list ----
        num_match = re.match(r'^(\d+)\.\s+(.*)', line.strip())
        if num_match:
            num  = num_match.group(1)
            text = _clean_inline(_escape_html(num_match.group(2)))
            flowables.append(Paragraph(f"{num}. {text}", styles["bullet"]))
            i += 1
            continue

        # ---- Empty line ----
        if not line.strip():
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # ---- Meta lines (bold key: value, centred) ----
        # Detect front-matter style: "**Key:** Value"
        if line.strip().startswith("**") and ":**" in line:
            text = _clean_inline(_escape_html(line.strip()))
            flowables.append(Paragraph(text, styles["meta"]))
            i += 1
            continue

        # ---- Default: body paragraph ----
        text = _clean_inline(_escape_html(line.strip()))
        if text:
            flowables.append(Paragraph(text, styles["body"]))
        i += 1

    return flowables


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_pdf() -> None:
    """Read the Markdown report and render it to PDF."""
    print(f"Reading  : {REPORT_MD}")
    md_text = REPORT_MD.read_text(encoding="utf-8")

    print("Parsing  : Markdown → Flowables")
    styles    = build_styles()
    flowables = md_to_flowables(md_text, styles)

    print(f"Rendering: {REPORT_PDF}")
    doc = build_doc(REPORT_PDF)
    doc.build(flowables)
    size_kb = REPORT_PDF.stat().st_size / 1024
    print(f"Done     : {REPORT_PDF.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build_pdf()
