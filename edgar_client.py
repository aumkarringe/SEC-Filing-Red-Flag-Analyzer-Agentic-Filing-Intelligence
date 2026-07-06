"""
edgar_client.py — SEC EDGAR API client

Responsibilities:
  - Resolve a company ticker to its EDGAR CIK
  - List a company's filings by type
  - Download raw filing HTML
  - Enforce the required User-Agent header and rate limiting

EDGAR rate limit: 10 req/s. We sleep 0.12s after every request to stay safe.
"""

import os
import sys
import time
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import requests
from dotenv import load_dotenv

from models import FilingMeta

load_dotenv()

_EDGAR_WWW  = "https://www.sec.gov"
_EDGAR_DATA = "https://data.sec.gov"
_REQUEST_DELAY_S = 0.12  # stay well under 10 req/s
_DISK_CACHE_DIR = Path(os.environ.get("EDGAR_CACHE_DIR", ".cache/edgar"))
_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _user_agent() -> str:
    user_agent = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not user_agent:
        raise EnvironmentError(
            "EDGAR_USER_AGENT is not set. "
            "Copy .env.template to .env and fill in 'AppName your@email.com'."
        )
    return user_agent


def _headers(user_agent: str) -> dict[str, str]:
    """Return request headers. EDGAR blocks requests without a valid User-Agent."""
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def _get_json_uncached(url: str, user_agent: str) -> dict | list:
    """Rate-limited GET that returns parsed JSON."""
    # Respect explicit opt-out
    if not os.environ.get("EDGAR_NO_CACHE"):
        # simple on-disk cache keyed by URL hash so Streamlit reruns/processes can share
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = _DISK_CACHE_DIR / f"{key}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                # fallback to network fetch on cache read error
                pass
    resp = requests.get(url, headers=_headers(user_agent), timeout=15)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY_S)
    data = resp.json()
    try:
        if not os.environ.get("EDGAR_NO_CACHE"):
            cache_path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data


@lru_cache(maxsize=32)
def _get_json_cached(url: str, user_agent: str) -> dict | list:
    return _get_json_uncached(url, user_agent)


def _get_json(url: str) -> dict | list:
    user_agent = _user_agent()
    if os.environ.get("EDGAR_NO_CACHE"):
        return _get_json_uncached(url, user_agent)
    return _get_json_cached(url, user_agent)


def _get_text_uncached(url: str, user_agent: str) -> str:
    # Respect explicit opt-out
    if not os.environ.get("EDGAR_NO_CACHE"):
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_path = _DISK_CACHE_DIR / f"{key}.html"
        if cache_path.exists():
            try:
                return cache_path.read_text(encoding="utf-8")
            except Exception:
                pass

    resp = requests.get(
        url,
        headers={**_headers(user_agent), "Accept": "text/html,application/xhtml+xml"},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.text
    time.sleep(_REQUEST_DELAY_S)
    try:
        if not os.environ.get("EDGAR_NO_CACHE"):
            cache_path.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text


@lru_cache(maxsize=64)
def _get_text_cached(url: str, user_agent: str) -> str:
    return _get_text_uncached(url, user_agent)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_cik(ticker: str) -> tuple[str, str]:
    """
    Look up a ticker symbol in EDGAR's full company list.

    Returns:
        (cik, company_name) where cik is zero-padded to 10 digits.

    Raises:
        ValueError  — ticker not found
        EnvironmentError — EDGAR_USER_AGENT not configured
    """
    # EDGAR publishes a single JSON file mapping every listed ticker → CIK.
    # Downloading it once per lookup is fine at interactive scale; cache in Stage 2.
    data: dict = _get_json(f"{_EDGAR_WWW}/files/company_tickers.json")

    ticker_upper = ticker.strip().upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            return cik, entry["title"]

    raise ValueError(
        f"Ticker '{ticker}' not found in EDGAR. "
        "Check the symbol or use the company's exact legal name."
    )


def get_submissions(cik: str) -> dict:
    """
    Fetch the submissions index for a CIK.
    Returns the raw JSON from data.sec.gov/submissions/CIK{cik}.json.

    The JSON contains:
      - entityType, name, tickers, exchanges
      - filings.recent: arrays of accessionNumber, form, filingDate, etc.
    """
    url = f"{_EDGAR_DATA}/submissions/CIK{cik}.json"
    return _get_json(url)


def get_recent_filings(cik: str, form_type: str, n: int = 2) -> list[FilingMeta]:
    """
    Return the n most recent filings of form_type for a given CIK.

    Raises:
        ValueError — fewer than n filings of that type found
    """
    subs = get_submissions(cik)
    company_name = subs.get("name", "")
    recent = subs.get("filings", {}).get("recent", {})

    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs       = recent.get("primaryDocument", [])

    target = form_type.strip().upper()
    matches: list[FilingMeta] = []

    for form, date, acc, doc in zip(forms, dates, accessions, docs):
        if form.strip().upper() == target and doc:  # skip entries with no primary doc
            matches.append(FilingMeta(
                cik=cik,
                accession_number=acc,
                form_type=form,
                filing_date=date,
                primary_document=doc,
                company_name=company_name,
            ))
        if len(matches) == n:
            break

    if len(matches) < n:
        raise ValueError(
            f"Found only {len(matches)} {form_type} filing(s) for CIK {cik} "
            f"(need {n} to compare). The company may not have filed this form type."
        )

    return matches


def fetch_filing_html(filing: FilingMeta) -> str:
    """
    Download and return the raw HTML of a filing's primary document.
    Results are cached in memory by document URL so repeated reruns are fast and
    Streamlit-host friendly.
    Set EDGAR_NO_CACHE=1 in the environment to bypass the cache.
    """
    user_agent = _user_agent()
    if os.environ.get("EDGAR_NO_CACHE"):
        return _get_text_uncached(filing.document_url, user_agent)
    return _get_text_cached(filing.document_url, user_agent)


def is_cached(url: str) -> bool:
    """Return True if the given URL has an on-disk cache entry and caching is enabled."""
    if os.environ.get("EDGAR_NO_CACHE"):
        return False
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = _DISK_CACHE_DIR / f"{key}.html"
    return cache_path.exists()


# ---------------------------------------------------------------------------
# CLI entry-point (Stage 1 verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python edgar_client.py <TICKER>")
        print("Example: python edgar_client.py AAPL")
        sys.exit(1)

    ticker_arg = sys.argv[1]

    try:
        cik, name = resolve_cik(ticker_arg)
    except (ValueError, EnvironmentError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Ticker : {ticker_arg.upper()}")
    print(f"Company: {name}")
    print(f"CIK    : {cik}")

    # Bonus: hit the submissions endpoint to confirm the full pipeline works
    print("\nFetching submissions index to confirm full EDGAR connectivity...")
    try:
        subs = get_submissions(cik)
        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        form_counts: dict[str, int] = {}
        for f in forms:
            form_counts[f] = form_counts.get(f, 0) + 1
        top = sorted(form_counts.items(), key=lambda x: -x[1])[:5]
        print(f"Recent filing types on record: {dict(top)}")
    except Exception as exc:
        print(f"Submissions fetch failed: {exc}")
        sys.exit(1)

    print("\nStage 1 OK — EDGAR connectivity confirmed.")
