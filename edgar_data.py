"""
edgar_data.py — Standalone EDGAR data access module

Three public functions:
  resolve_cik(ticker)               -> (cik, company_name)
  get_recent_filings(cik, form, n)  -> list of filing dicts
  fetch_filing_text(filing)         -> plain text string

Requires in .env:
  EDGAR_USER_AGENT="YourAppName your@email.com"
  SEC enforces this header so they can contact you if your client misbehaves.
  Missing or empty → EnvironmentError with a clear message.

Rate limit: SEC allows 10 req/s. We sleep 0.12s after every request.
"""

import os
import re
import sys
import time
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

load_dotenv()

# iXBRL filings are technically XML served as HTML — suppress the parser mismatch warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_WWW   = "https://www.sec.gov"
_DATA  = "https://data.sec.gov"
_DELAY = 0.12   # seconds between requests — keeps us under SEC's 10 req/s limit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise EnvironmentError(
            "EDGAR_USER_AGENT is not set.\n"
            "Add this line to your .env file:\n"
            '  EDGAR_USER_AGENT="YourAppName your@email.com"\n'
            "SEC requires a descriptive User-Agent to identify your client."
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _get_json(url: str) -> dict:
    resp = requests.get(url, headers=_headers(), timeout=15)
    resp.raise_for_status()
    time.sleep(_DELAY)
    return resp.json()


def _get_html(url: str) -> str:
    resp = requests.get(
        url,
        headers={**_headers(), "Accept": "text/html,application/xhtml+xml"},
        timeout=60,   # large 10-K documents can be several MB
    )
    resp.raise_for_status()
    time.sleep(_DELAY)
    return resp.text


# ---------------------------------------------------------------------------
# 1. Resolve ticker → CIK
# ---------------------------------------------------------------------------

def resolve_cik(ticker: str) -> tuple[str, str]:
    """
    Look up a ticker symbol in EDGAR's full company list.

    Returns (cik, company_name).
    cik is zero-padded to 10 digits — the format EDGAR uses in all API paths.

    How it works: EDGAR publishes one JSON file (~5 MB) containing every listed
    company. We fetch it once and scan for the matching ticker. There is no
    per-ticker lookup endpoint, so this is the canonical approach.

    Raises:
        ValueError        — ticker not found
        EnvironmentError  — EDGAR_USER_AGENT not configured
    """
    data = _get_json(f"{_WWW}/files/company_tickers.json")
    needle = ticker.strip().upper()
    for entry in data.values():
        if entry["ticker"].upper() == needle:
            cik = str(entry["cik_str"]).zfill(10)
            return cik, entry["title"]
    raise ValueError(
        f"Ticker '{ticker}' not found in EDGAR. "
        "Check the symbol, or try the company's exact legal name."
    )


# ---------------------------------------------------------------------------
# 2. List filings → most recent N of a given type
# ---------------------------------------------------------------------------

def get_recent_filings(
    cik: str,
    form_type: str = "10-K",
    n: int = 2,
) -> list[dict]:
    """
    Return metadata for the n most recent filings of form_type for a given CIK.

    Each returned dict contains:
        accession_number  — e.g. "0000320193-25-000079"
        form_type         — e.g. "10-K"
        filing_date       — ISO date string, e.g. "2025-10-31"
        document_url      — direct URL to the primary HTML document

    How the URL is built:
        EDGAR Archives path = /Archives/edgar/data/{cik_no_padding}/{acc_no_dashes}/{primary_doc}
        The submissions JSON gives us accessionNumber and primaryDocument directly.

    Raises:
        ValueError — fewer than n filings of that type on record
    """
    subs   = _get_json(f"{_DATA}/submissions/CIK{cik}.json")
    recent = subs.get("filings", {}).get("recent", {})

    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs       = recent.get("primaryDocument", [])

    target  = form_type.strip().upper()
    matches = []

    for form, date, acc, doc in zip(forms, dates, accessions, docs):
        if form.strip().upper() == target and doc:
            acc_path = acc.replace("-", "")           # "0000320193-25-000079" → "000032019325000079"
            cik_int  = str(int(cik))                  # strip leading zeros for the Archives path
            url = f"{_WWW}/Archives/edgar/data/{cik_int}/{acc_path}/{doc}"
            matches.append({
                "accession_number": acc,
                "form_type":        form,
                "filing_date":      date,
                "document_url":     url,
            })
        if len(matches) == n:
            break

    if len(matches) < n:
        raise ValueError(
            f"Only {len(matches)} {form_type} filing(s) found for CIK {cik} "
            f"(need {n} to compare). The company may never have filed this form type."
        )
    return matches


# ---------------------------------------------------------------------------
# 3. Download filing document → plain text
# ---------------------------------------------------------------------------

def fetch_filing_text(filing: dict) -> str:
    """
    Download the primary HTML document for a filing and return clean plain text.

    Steps:
      1. Fetch the HTML (may be iXBRL — Inline XBRL wrapped around HTML)
      2. Remove <script>, <style>, <head> tags
      3. Unwrap iXBRL inline elements (ix:nonfraction, etc.) — keeps the text,
         drops the XBRL measurement metadata that would appear as noise
      4. Extract text with newline separators
      5. Collapse runs of blank lines
    """
    html = _get_html(filing["document_url"])

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # iXBRL elements wrap financial figures inline — unwrap keeps the number text
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)   # collapse excess blank lines
    return text.strip()


# ---------------------------------------------------------------------------
# Test — run directly to confirm live EDGAR access
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ticker    = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    form_type = sys.argv[2] if len(sys.argv) > 2 else "10-K"

    # ── Step 1: CIK lookup ────────────────────────────────────────────────
    print(f"\nStep 1 — Resolving CIK for '{ticker}'...")
    try:
        cik, company_name = resolve_cik(ticker)
    except (ValueError, EnvironmentError) as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    print(f"  Company : {company_name}")
    print(f"  CIK     : {cik}")

    # ── Step 2: Filing list ───────────────────────────────────────────────
    print(f"\nStep 2 — Fetching two most recent {form_type} filings...")
    try:
        filings = get_recent_filings(cik, form_type=form_type, n=2)
    except ValueError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    for i, f in enumerate(filings):
        label = "newer" if i == 0 else "older"
        print(f"  [{label}]  {f['filing_date']}  →  {f['document_url']}")

    # ── Step 3: Download text ─────────────────────────────────────────────
    print(f"\nStep 3 — Downloading and converting to plain text...")
    texts = []
    for f in filings:
        label = "newer" if filings.index(f) == 0 else "older"
        print(f"  [{label}] {f['filing_date']} ...", end=" ", flush=True)
        try:
            text = fetch_filing_text(f)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        texts.append(text)
        print(f"{len(text):,} characters")

    # ── Smoke-check ───────────────────────────────────────────────────────
    print(f"\n--- First 400 characters of newer filing ({filings[0]['filing_date']}) ---")
    print(texts[0][:400].strip())
    print("\n...")

    newer_words = len(texts[0].split())
    older_words = len(texts[1].split())
    print(f"\nWord counts:  newer = {newer_words:,}  |  older = {older_words:,}")
    print("\n✓  Stage 1 complete — two real filings fetched and converted to plain text.")
