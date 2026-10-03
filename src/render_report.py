"""Render docs/report_draft.md to a one-page PDF in the style of docs/report_format.pdf.
Usage: .venv-image/bin/python src/render_report.py [out.pdf]   (needs: pip install reportlab markdown)
Fails loudly if the result is longer than one page."""
import re, sys, os, html
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "report_draft.md")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "report_draft.pdf")

text = re.sub(r"<!--.*?-->", "", open(SRC, encoding="utf-8").read(), flags=re.S).strip()
title_style = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=6)
h_style = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=11, leading=13, spaceBefore=6, spaceAfter=2)
p_style = ParagraphStyle("p", fontName="Helvetica", fontSize=9.2, leading=11.6, spaceAfter=3)


def inline(md_line: str) -> str:
    h = markdown.markdown(md_line)                   # <p>...</p> with <strong>/<code>
    h = re.sub(r"^<p>|</p>$", "", h.strip())
    h = h.replace("<strong>", "<b>").replace("</strong>", "</b>").replace("<em>", "<i>").replace("</em>", "</i>")
    h = re.sub(r"<code>(.*?)</code>", r'<font face="Courier-Bold" color="#1a7f37">\1</font>', h)
    return h


story = []
for block in [b for b in re.split(r"\n\s*\n", text) if b.strip()]:
    b = block.strip()
    if b.startswith("# "):
        story.append(Paragraph(html.escape(b[2:]), title_style))
    elif b.startswith("## "):
        story.append(Paragraph(html.escape(b[3:]), h_style))
    else:
        story.append(Paragraph(inline(" ".join(b.splitlines())), p_style))

pages = []
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=14 * mm,
                        title="Technical Proposal", author="Team")
doc.build(story, onFirstPage=lambda c, d: pages.append(1), onLaterPages=lambda c, d: pages.append(1))
n = len(pages)
print(f"wrote {OUT} ({n} page{'s' if n != 1 else ''}, {os.path.getsize(OUT)//1024} KB)")
if n != 1:
    sys.exit(f"ERROR: report is {n} pages; the portal wants one. Trim docs/report_draft.md.")
