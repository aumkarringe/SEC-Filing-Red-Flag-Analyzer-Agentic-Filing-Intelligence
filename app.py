"""
app.py — Streamlit UI for the SEC Red-Flag Agent (Stage 5)

Run with:  streamlit run app.py
"""

import streamlit as st
import io
from datetime import datetime, timezone
from jinja2 import Template

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from agent import run_analysis
from models import FilingDiff, RedFlagReport

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SEC Red-Flag Agent",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚨 SEC Red-Flag Agent")
st.caption(
    "Compare a company's two most recent SEC filings and identify material changes "
    "that may signal risk to investors."
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("analysis_form"):
    col1, col2, col3 = st.columns([3, 1, 1])

    ticker_input = col1.text_input(
        "Company Ticker",
        placeholder="e.g. AAPL, MSFT, TSLA",
        help="Enter the stock ticker symbol (must be listed on EDGAR)",
    )
    form_type_input = col2.selectbox(
        "Filing Type",
        ["10-K", "10-Q"],
        help="10-K = annual report  ·  10-Q = quarterly report",
    )
    col3.markdown("<br>", unsafe_allow_html=True)  # vertical align
    submitted = col3.form_submit_button("Analyze →", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Run analysis on submit
# ---------------------------------------------------------------------------

if submitted:
    ticker = ticker_input.strip().upper()
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    with st.spinner(f"Analyzing **{ticker} {form_type_input}** — fetching filings, running diff, and LLM reasoning (~30–60s)..."):
        try:
            report, filing_diff = run_analysis(ticker, form_type_input)
        except ValueError as e:
            st.error(f"**Filing error:** {e}")
            st.stop()
        except EnvironmentError as e:
            st.error(f"**Configuration error:** {e}")
            st.stop()
        except Exception as e:
            st.error(f"**Error:** {e}")
            st.stop()

    st.session_state["report"] = report
    st.session_state["diff"] = filing_diff

# ---------------------------------------------------------------------------
# Display results (persists across reruns via session_state)
# ---------------------------------------------------------------------------

if "report" not in st.session_state:
    st.info("Enter a ticker and click **Analyze →** to generate a red-flag report.")
    st.stop()

report: RedFlagReport = st.session_state["report"]
diff: FilingDiff = st.session_state["diff"]

st.divider()

# ── Filing metadata ────────────────────────────────────────────────────────
st.subheader(f"{report.company_name} ({report.ticker}) — {report.form_type}")
st.caption(
    f"Newer filing: **{report.newer_date}**  ·  Older filing: **{report.older_date}**"
)
# Show cache indicators if available
cached_new = diff.cached_map.get(report.newer_date) if getattr(diff, "cached_map", None) else None
cached_old = diff.cached_map.get(report.older_date) if getattr(diff, "cached_map", None) else None
if cached_new or cached_old:
    badges = []
    if cached_new:
        badges.append(f"Newer filing ({report.newer_date}) served from cache")
    if cached_old:
        badges.append(f"Older filing ({report.older_date}) served from cache")
    st.info("  ·  ".join(badges))

# Report-level cache badge (if a precomputed RedFlagReport was used)
report_cached_at = diff.cached_map.get("report_cached_at") if getattr(diff, "cached_map", None) else None
report_cached_flag = diff.cached_map.get("report") if getattr(diff, "cached_map", None) else None
if report_cached_flag and report_cached_at:
    st.warning(f"Report served from cache — results frozen at {report_cached_at}")
elif report_cached_flag:
    st.warning("Report served from cache — results frozen (timestamp unavailable)")

# ── Section-level word count metrics ──────────────────────────────────────
if diff.section_diffs:
    st.markdown("#### Section Changes")
    metric_cols = st.columns(len(diff.section_diffs))
    for col, sec_diff in zip(metric_cols, diff.section_diffs.values()):
        # Extract short label: "Item 1A — Risk Factors" → "Risk Factors"
        short = sec_diff.section_label.split(" — ", 1)[-1]
        delta = sec_diff.word_count_delta
        delta_str = f"{delta:+,} words"
        col.metric(
            label=short,
            value=f"{sec_diff.newer_word_count:,}",
            delta=delta_str,
            delta_color="off",  # word count direction is ambiguous as good/bad
        )

# ── Summary banner ────────────────────────────────────────────────────────
st.markdown("#### Analysis Summary")
n_flags = len(report.flags)
n_high = sum(1 for f in report.flags if f.severity == "HIGH")
n_med = sum(1 for f in report.flags if f.severity == "MEDIUM")
n_low = sum(1 for f in report.flags if f.severity == "LOW")

if n_flags == 0:
    st.success(f"✅ **No red flags identified.**  {report.analyst_summary}")
elif n_high > 0:
    st.error(
        f"🔴 **{n_flags} red flag(s) found — {n_high} HIGH, {n_med} MEDIUM, {n_low} LOW severity.**\n\n"
        f"{report.analyst_summary}"
    )
else:
    st.warning(
        f"🟡 **{n_flags} red flag(s) found — {n_med} MEDIUM, {n_low} LOW severity.**\n\n"
        f"{report.analyst_summary}"
    )

# ── Flag cards ────────────────────────────────────────────────────────────
if report.flags:
    st.markdown("#### Red Flags")

    sev_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

    for flag in report.flags:
        icon = sev_icon.get(flag.severity, "⚪")
        conf_icon = {"HIGH": "●", "MEDIUM": "◑", "LOW": "○"}.get(flag.confidence, "○")
        expander_title = (
            f"{icon} **[{flag.severity}]** `{flag.category}` — {flag.headline}"
            f"  ·  confidence {conf_icon} {flag.confidence}"
        )

        with st.expander(expander_title, expanded=(flag.severity == "HIGH")):
            st.markdown(f"**Section:** {flag.section}  ·  **Confidence:** {flag.confidence}")
            st.markdown(flag.explanation)

            if flag.evidence_newer or flag.evidence_older:
                st.markdown("**Evidence**")
                ev1, ev2 = st.columns(2)

                with ev1:
                    if flag.evidence_newer:
                        st.markdown(f"*Newer filing ({report.newer_date})*")
                        st.info(flag.evidence_newer)
                    else:
                        st.markdown("*(no prior text)*")

                with ev2:
                    if flag.evidence_older:
                        st.markdown(f"*Older filing ({report.older_date})*")
                        st.warning(flag.evidence_older)
                    else:
                        st.markdown("*(text not present in older filing)*")

# ── Download ──────────────────────────────────────────────────────────────
st.divider()
st.download_button(
    label="⬇ Download JSON Report",
    data=report.model_dump_json(indent=2),
    file_name=f"{report.ticker}_{report.form_type}_{report.newer_date}_redflag.json",
    mime="application/json",
)


def _render_report_pdf(report: RedFlagReport) -> bytes:
    """Render a simple styled PDF for the RedFlagReport using ReportLab."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = styles["Heading1"]
    elems = []

    elems.append(Paragraph(f"SEC Red-Flag Report — {report.company_name} ({report.ticker})", heading))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(f"Filing: {report.form_type} — Newer: {report.newer_date}  |  Older: {report.older_date}", normal))
    elems.append(Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC", normal))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("Summary:", styles["Heading2"]))
    elems.append(Paragraph(report.analyst_summary, normal))
    elems.append(Spacer(1, 12))

    if not report.flags:
        elems.append(Paragraph("No red flags identified.", normal))
    else:
        from reportlab.platypus import Table, TableStyle
        sev_map = {
            "HIGH": (colors.HexColor("#fff0f0"), colors.HexColor("#e76f51")),
            "MEDIUM": (colors.HexColor("#fffaf0"), colors.HexColor("#e9c46a")),
            "LOW": (colors.HexColor("#f0fff5"), colors.HexColor("#2a9d8f")),
        }

        for i, flag in enumerate(report.flags, 1):
            bg, border = sev_map.get(flag.severity, (colors.HexColor("#f7f7f7"), colors.HexColor("#888888")))
            headline = Paragraph(f"{i}. [{flag.severity}] {flag.category} — {flag.headline}", styles.get("Heading3", heading))
            meta = Paragraph(f"Section: {flag.section} — Confidence: {flag.confidence}", normal)
            expl = Paragraph(flag.explanation, normal)
            rows = [[headline], [meta], [expl]]
            t = Table(rows, colWidths=[doc.width])
            t.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BOX", (0, 0), (-1, -1), 1, border),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            elems.append(t)
            if flag.evidence_newer:
                elems.append(Paragraph(f"Newer evidence: {flag.evidence_newer}", normal))
            if flag.evidence_older:
                elems.append(Paragraph(f"Older evidence: {flag.evidence_older}", normal))
            elems.append(Spacer(1, 10))

    # Draw page background on each page
    def _draw_bg(canvas, document):
        w, h = document.pagesize
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#eef6fb"))
        canvas.rect(0, 0, w, h, stroke=0, fill=1)
        canvas.restoreState()

    doc.build(elems, onFirstPage=_draw_bg, onLaterPages=_draw_bg)
    buf.seek(0)
    return buf.read()


pdf_bytes = _render_report_pdf(report)
st.download_button(
    label="⬇ Download PDF Report",
    data=pdf_bytes,
    file_name=f"{report.ticker}_{report.form_type}_{report.newer_date}_redflag.pdf",
    mime="application/pdf",
)

# -----------------------
# HTML/CSS template export
# -----------------------

_HTML_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <meta charset="utf-8"/>
        <title>{{ company }} — Red Flag Report</title>
        <style>
            body { font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #0b2545; background: #eef6fb; }
            header { border-bottom: 2px solid #2b5d8a; margin-bottom: 16px; }
            h1 { font-size: 20pt; margin: 0; color: #103a63 }
            .meta { color: #345; margin-top: 6px; }
            .summary { background:#e8f3ff; padding:14px; border-radius:8px; margin:12px 0; border:1px solid #d0e6ff }
            .flag { padding:12px 14px; margin:12px 0; border-radius:8px; color:#0b2545 }
            .flag.low { background:#e8fff3; border:1px solid #bfe9d6 }
            .flag.medium { background:#fff7e6; border:1px solid #ffe6b8 }
            .flag.high { background:#ffecec; border:1px solid #ffcfcf }
            .flag .headline { font-weight:700; display:block; margin-bottom:6px }
            .evidence { font-family: monospace; background:#f6fbff; padding:8px; border-radius:6px; display:block; margin-top:8px; border:1px solid #e1efff }
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


def _render_report_html(report: RedFlagReport) -> str:
        tmpl = Template(_HTML_TEMPLATE)
        html = tmpl.render(
                company=report.company_name,
                ticker=report.ticker,
                form_type=report.form_type,
                newer=report.newer_date,
                older=report.older_date,
                summary=report.analyst_summary,
                flags=[f.model_dump() for f in report.flags],
                generated=datetime.now(timezone.utc).isoformat() + " UTC",
        )
        return html


def _render_pdf_from_html(html: str) -> bytes | None:
        """Try to convert HTML -> PDF using WeasyPrint if available; return bytes or None."""
        try:
                from weasyprint import HTML
        except Exception:
                return None

        out = io.BytesIO()
        HTML(string=html).write_pdf(out)
        return out.getvalue()


html_report = _render_report_html(report)
st.download_button(
        label="⬇ Download HTML Report (designable)",
        data=html_report,
        file_name=f"{report.ticker}_{report.form_type}_{report.newer_date}_redflag.html",
        mime="text/html",
)

pdf_from_html = _render_pdf_from_html(html_report)
if pdf_from_html:
        st.download_button(
                label="⬇ Download PDF (from HTML template)",
                data=pdf_from_html,
                file_name=f"{report.ticker}_{report.form_type}_{report.newer_date}_redflag_htmltempl.pdf",
                mime="application/pdf",
        )
else:
        st.info("To export PDF from the HTML template, install WeasyPrint and its system dependencies; falling back to the simple PDF export above.")
