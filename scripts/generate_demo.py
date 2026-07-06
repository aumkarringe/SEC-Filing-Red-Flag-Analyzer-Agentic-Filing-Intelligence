from agent import run_analysis
from pathlib import Path
from datetime import datetime
from jinja2 import Template
import json

out = Path("output")
out.mkdir(exist_ok=True)

TICKER = "AAPL"
FORM = "10-K"

report, filing_diff = run_analysis(TICKER, FORM)

# Save JSON
json_path = out / f"{TICKER}_{FORM}_{report.newer_date}_redflag.json"
json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

# Render HTML (same template as app)
_HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{{ company }} — Red Flag Report</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #111; }
      header { border-bottom: 2px solid #222; margin-bottom: 16px; }
      h1 { font-size: 20pt; margin: 0; }
      .meta { color: #555; margin-top: 4px; }
      .summary { background:#f7f7f9; padding:12px; border-radius:6px; margin:12px 0; }
      .flag { border-left:4px solid #c00; padding:8px 12px; margin:12px 0; }
      .flag.low { border-color: #2a9d8f; }
      .flag.medium { border-color: #e9c46a; }
      .flag.high { border-color: #e76f51; }
      .evidence { font-family: monospace; background:#fff; padding:8px; border-radius:4px; display:block; margin-top:6px; }
    </style>
  </head>
  <body>
    <header>
      <h1>SEC Red-Flag Report — {{ company }} ({{ ticker }})</h1>
      <div class="meta">{{ form_type }} — Newer: {{ newer }}  |  Older: {{ older }}</div>
    </header>

    <section class="summary">
      <strong>Summary</strong>
      <p>{{ summary }}</p>
    </section>

    {% if flags %}
      <section>
      <h2>Flags ({{ flags|length }})</h2>
      {% for f in flags %}
        <article class="flag {{ f.severity|lower }}">
          <strong>[{{ f.severity }}] {{ f.category }} — {{ f.headline }}</strong>
          <div>{{ f.explanation }}</div>
          {% if f.evidence_newer %}
            <div class="evidence">Newer: {{ f.evidence_newer }}</div>
          {% endif %}
          {% if f.evidence_older %}
            <div class="evidence">Older: {{ f.evidence_older }}</div>
          {% endif %}
        </article>
      {% endfor %}
      </section>
    {% else %}
      <p>No red flags identified.</p>
    {% endif %}

    <footer style="margin-top:24px;color:#666;font-size:0.9em">Generated: {{ generated }}</footer>
  </body>
</html>
"""

tmpl = Template(_HTML_TEMPLATE)
html = tmpl.render(
    company=report.company_name,
    ticker=report.ticker,
    form_type=report.form_type,
    newer=report.newer_date,
    older=report.older_date,
    summary=report.analyst_summary,
    flags=[f.model_dump() for f in report.flags],
    generated=datetime.utcnow().isoformat() + " UTC",
)
html_path = out / f"{TICKER}_{FORM}_{report.newer_date}_redflag.html"
html_path.write_text(html, encoding="utf-8")

# Convert to PDF using WeasyPrint if available
try:
    from weasyprint import HTML
    pdf_bytes = HTML(string=html).write_pdf()
    pdf_path = out / f"{TICKER}_{FORM}_{report.newer_date}_redflag_htmltempl.pdf"
    pdf_path.write_bytes(pdf_bytes)
    print('Wrote:', json_path, html_path, pdf_path)
except Exception as e:
    print('Wrote JSON+HTML; PDF conversion failed or WeasyPrint not available:', e)
    print('Wrote:', json_path, html_path)
