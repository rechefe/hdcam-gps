"""Render PROPOSAL.md to a print-ready PDF.

    uv run python scripts/make_proposal_pdf.py

Kept as a script rather than a one-off so the PDF can be regenerated whenever the
proposal changes. Needs the dev dependency group (weasyprint, markdown).
"""
from pathlib import Path
import markdown
from weasyprint import HTML

SOURCE = Path("PROPOSAL.md")
OUTPUT = Path("PROPOSAL.pdf")

CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 18mm 18mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "DejaVu Sans", Helvetica, sans-serif;
        font-size: 8pt; color: #888;
    }
}
body {
    font-family: "DejaVu Serif", Georgia, serif;
    font-size: 10pt; line-height: 1.45; color: #1a1a1a;
    hyphens: auto;
}
h1 {
    font-family: "DejaVu Sans", Helvetica, sans-serif;
    font-size: 19pt; line-height: 1.2; margin: 0 0 2mm 0;
    padding-bottom: 3mm; border-bottom: 1.5pt solid #1a1a1a;
}
h2 {
    font-family: "DejaVu Sans", Helvetica, sans-serif;
    font-size: 12.5pt; margin: 8mm 0 2.5mm 0; color: #000;
    break-after: avoid;
}
h3 {
    font-family: "DejaVu Sans", Helvetica, sans-serif;
    font-size: 10.5pt; margin: 5mm 0 2mm 0; break-after: avoid;
}
p { margin: 0 0 2.5mm 0; text-align: justify; }
strong { font-weight: 700; }
ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1.6mm; text-align: justify; break-inside: avoid; }
li > p { margin-bottom: 1.2mm; }
code {
    font-family: "DejaVu Sans Mono", Menlo, monospace;
    font-size: 8.5pt; background: #f2f2f0; padding: 0.3mm 0.8mm;
    border-radius: 1pt;
}
pre {
    font-family: "DejaVu Sans Mono", Menlo, monospace;
    font-size: 7.6pt; line-height: 1.32;
    background: #f7f7f5; border: 0.4pt solid #ddd; border-left: 2pt solid #999;
    padding: 2.5mm 3mm; margin: 2.5mm 0 3.5mm 0;
    white-space: pre-wrap; word-wrap: break-word;
    break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
    border-collapse: collapse; width: 100%;
    margin: 2.5mm 0 4mm 0; font-size: 8.6pt;
    break-inside: avoid;
}
th, td {
    border: 0.4pt solid #bbb; padding: 1.4mm 2mm;
    text-align: left; vertical-align: top;
}
th {
    background: #eeeeec;
    font-family: "DejaVu Sans", Helvetica, sans-serif; font-size: 8.2pt;
}
em { font-style: italic; }
"""

html_body = markdown.markdown(
    SOURCE.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code", "sane_lists"],
)
TITLE = "GPS L1 C/A acquisition as a 1-bit HD-CAM lookup"
document = (
    f"<html><head><meta charset='utf-8'><title>{TITLE}</title>"
    f"<style>{CSS}</style></head><body>{html_body}</body></html>"
)
HTML(string=document, base_url=".").write_pdf(OUTPUT)
print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size/1024:.0f} kB)")
