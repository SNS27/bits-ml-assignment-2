"""
Build the submission PDF in the required order:
  1) clickable GitHub link  2) clickable Streamlit link  3) BITS Lab screenshot  4) full README.
Usage: python tools/build_pdf.py --repo URL --app URL --shot screenshot.png --readme README.md --out NAME.pdf
"""
import argparse, re, sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--app", required=True)
ap.add_argument("--shot", required=True); ap.add_argument("--readme", default="README.md")
ap.add_argument("--out", required=True)
a = ap.parse_args()

text = Path(a.readme).read_text(encoding="utf-8")
for bad in ("TODO", "<PASTE", "{{"):
    if bad in text:
        sys.exit(f"README still contains placeholder '{bad}'. Finish it first.")
if "http" not in a.app:
    sys.exit("--app must be the real Streamlit URL")

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
cell = ParagraphStyle("c", parent=body, fontSize=8.5, leading=11)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10)
h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11)
code = ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10, backColor=colors.HexColor("#F2F2F2"))


def clean(s):                       # reportlab base fonts: keep Latin-1 only
    return s.encode("latin-1", "ignore").decode("latin-1")


def inline(s):
    s = clean(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"(https?://[^\s<)]+)", r'<a href="\1" color="blue">\1</a>', s)
    return s


def md_table(rows, width):
    data = [[Paragraph(inline(c.strip()), cell) for c in r] for r in rows]
    n = len(data[0])
    if n == 2:
        widths = [width * 0.26, width * 0.74]
    else:
        widths = [width * 0.22] + [width * 0.78 / (n - 1)] * (n - 1)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E1D9")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm, title="ML Assignment 2", author="Shaleen Sharma")
W = A4[0] - 4 * cm
S = [Paragraph("ML Assignment 2: Submission", h1), Spacer(1, 10),
     Paragraph("<b>1. GitHub repository:</b>", body),
     Paragraph(f'<a href="{a.repo}" color="blue">{a.repo}</a>', body), Spacer(1, 8),
     Paragraph("<b>2. Live Streamlit app:</b>", body),
     Paragraph(f'<a href="{a.app}" color="blue">{a.app}</a>', body), Spacer(1, 8),
     Paragraph("<b>3. Screenshot of execution on BITS Virtual Lab:</b>", body), Spacer(1, 4)]
from reportlab.lib.utils import ImageReader
iw, ih = ImageReader(a.shot).getSize()
scale = min(W / iw, 15 * cm / ih)
S += [Image(a.shot, iw * scale, ih * scale), PageBreak(),
      Paragraph("4. README content", h1), Spacer(1, 6)]

lines, i = text.splitlines(), 0
while i < len(lines):
    ln = lines[i]
    if ln.strip().startswith("<!--"):
        while "-->" not in lines[i]:
            i += 1
        i += 1; continue
    if ln.startswith("```"):
        j = i + 1; blk = []
        while not lines[j].startswith("```"):
            blk.append(lines[j]); j += 1
        S += [Preformatted(clean("\n".join(blk)), code), Spacer(1, 6)]; i = j + 1; continue
    if ln.startswith("|"):
        tbl = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|[\s:|-]+\|$", lines[i].strip()):
                tbl.append(lines[i].strip().strip("|").split("|"))
            i += 1
        S += [md_table(tbl, W), Spacer(1, 8)]; continue
    if ln.startswith("### "): S.append(Paragraph(inline(ln[4:]), h3))
    elif ln.startswith("## "): S.append(Paragraph(inline(ln[3:]), h2))
    elif ln.startswith("# "): S.append(Paragraph(inline(ln[2:]), h2))
    elif ln.startswith("- "):
        items = []
        while i < len(lines) and lines[i].startswith("- "):
            items.append(ListItem(Paragraph(inline(lines[i][2:]), body))); i += 1
        S.append(ListFlowable(items, bulletType="bullet", start="-", leftIndent=14)); continue
    elif ln.strip():
        S.append(Paragraph(inline(ln), body))
    i += 1
doc.build(S)
print("written:", a.out)
