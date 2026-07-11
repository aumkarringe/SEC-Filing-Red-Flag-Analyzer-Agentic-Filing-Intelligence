# SEC Red-Flag Agent

Streamlit app for comparing two recent SEC filings for a ticker and surfacing material red flags.

## About

The SEC Red-Flag Agent automates the work of comparing two consecutive SEC filings (10-K or 10-Q) for a company and highlights concrete, material changes that could signal investor risk. It fetches filings from the SEC EDGAR service, extracts named sections (e.g., Risk Factors, MD&A), computes a paragraph-level diff, and uses an LLM-based analyst agent to reason about which changes qualify as red flags.

Key features:
- Fetches and parses EDGAR filings and extracts relevant sections (Item 1A, MD&A, Market Risk, Legal Proceedings).
- Computes structured diffs at the paragraph level and summarizes section word-count deltas.
- Uses a Google ADK LlmAgent with LiteLLM (routed through OpenRouter) to produce a structured `RedFlagReport` with categorized flags, severity and confidence labels, and verbatim evidence quotes.
- Provides multiple exports: structured JSON, a designable HTML report, and PDF exports (ReportLab or HTML→PDF via WeasyPrint).
- On-disk and in-memory caching of fetched filings to speed up repeat runs and reduce EDGAR requests; cache status is surfaced in the UI.

The tool is intended for analysts who want a fast, reproducible way to surface materially new facts between filings and produce shareable reports for review.

## Quick start (macOS / zsh)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Configure secrets (local): create a `.env` file in the repo root with at least:

```env
EDGAR_USER_AGENT="AppName contact@your.email"
OPENROUTER_API_KEY="your-openrouter-api-key"
OPENROUTER_MODEL="openrouter/openai/gpt-4o-mini"
# (optional) OPENROUTER_MODEL can be changed for different quality/cost tradeoffs
```

4. Run the Streamlit app:

```bash
source .venv/bin/activate
streamlit run app.py
```

The UI will appear on the Local URL printed by Streamlit (e.g. `http://localhost:8501`).

## Demo script

You can run a non-interactive demo that generates report artifacts (JSON, HTML, PDF) for `AAPL`:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/generate_demo.py
```

Generated files are written to the `output/` directory.

## Caching

- The app uses an in-memory `lru_cache` for fast repeated requests during a single process run and an on-disk cache under `.cache/edgar` so cached filings persist across restarts.
- To bypass caching set `EDGAR_NO_CACHE=1` before running the app, or delete the cache with `rm -rf .cache/edgar`.
- You may change the cache folder with the environment variable `EDGAR_CACHE_DIR`.

## Exports (JSON / HTML / PDF)

- The app provides a JSON download of the structured `RedFlagReport`.
- There's a designable HTML report produced by a Jinja2 template; download it from the UI and edit the template in `app.py` to change styling.
- The app can convert HTML → PDF using WeasyPrint (optional). If WeasyPrint is not installed, the app falls back to a simpler ReportLab PDF export.

To enable HTML→PDF conversion, install WeasyPrint and its system dependencies (macOS example):

```bash
# macOS (using Homebrew) — may be needed for WeasyPrint
brew install cairo pango gdk-pixbuf libffi
source .venv/bin/activate
pip install weasyprint
```

## Configuration / Troubleshooting

- If you see an authentication error from OpenRouter like `Missing Authentication header` or HTTP 401, verify `OPENROUTER_API_KEY` is correct and active and that Streamlit was restarted after setting the env var.
- EDGAR requests require a valid `EDGAR_USER_AGENT` (format: `AppName contact@email`).
- If caching seems not to apply, ensure you restarted Streamlit (the on-disk cache is read on fetch) and that `EDGAR_NO_CACHE` is not set.

## Files of interest

- `app.py` — Streamlit UI and download buttons (JSON, HTML, PDF)
- `agent.py` — pipeline orchestration and ADK LLM agent
- `edgar_client.py` — EDGAR fetch + disk cache helper (`is_cached()`)
- `section_extractor.py` / `differ.py` — parsing and diff logic
- `scripts/generate_demo.py` — non-interactive demo runner
- `requirements.txt` — Python dependencies
