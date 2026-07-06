"""
section_extractor.py — Parse named sections from SEC filing HTML

Approach:
  1. Download the primary document HTML via edgar_client
  2. Strip HTML/iXBRL noise → plain text
  3. Find all "Item N" boundaries → split into chunks
  4. For each target section, take the longest chunk
     (filings always have a Table of Contents that creates a short duplicate;
      the actual section body is always longer than the ToC entry)

Supported form types: 10-K, 10-Q
"""

import re
import sys
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# iXBRL filings are technically XML, but lxml's HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from edgar_client import fetch_filing_html, get_recent_filings, resolve_cik
from models import FilingMeta, FilingSection

# ---------------------------------------------------------------------------
# Target sections per form type.
# Key = item number string (lowercase); Value = (machine_key, human_label)
# ---------------------------------------------------------------------------
_TARGETS: dict[str, dict[str, tuple[str, str]]] = {
    "10-K": {
        "1a": ("risk_factors",      "Item 1A — Risk Factors"),
        "3":  ("legal_proceedings", "Item 3 — Legal Proceedings"),
        "7":  ("mda",               "Item 7 — MD&A"),
        "7a": ("market_risk",       "Item 7A — Market Risk"),
    },
    "10-Q": {
        # Part I: Item 2 = MD&A, Item 3 = Market Risk
        # Part II: Item 1A = Risk Factors  (Item 1 = Legal Proceedings, but
        #   ambiguous with Part I Item 1 = Financial Statements, so skipped)
        "1a": ("risk_factors", "Item 1A — Risk Factors"),
        "2":  ("mda",          "Item 2 — MD&A"),
        "3":  ("market_risk",  "Item 3 — Market Risk"),
    },
}


def _html_to_text(html: str) -> str:
    """Strip HTML tags (including iXBRL inline elements) → normalized plain text."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # iXBRL tags (ix:nonfraction, ix:nonnumeric, etc.) wrap inline data —
    # unwrap keeps the text content while removing the XBRL-specific markup.
    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    text = soup.get_text(separator="\n")

    # Collapse runs of blank lines to single blank line
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    return text


def _split_by_items(text: str) -> dict[str, list[str]]:
    """
    Locate all "Item N" section headers and split the text at those boundaries.

    Returns dict of item_key → [chunk, chunk, ...].
    Multiple chunks per key happen when an item appears in both the ToC and body.
    Callers should take the longest chunk to get the actual section content.
    """
    # Match "Item 1A." / "ITEM 7A " at the start of a line with optional indentation.
    # We allow up to 8 leading spaces/tabs to handle indented headings in tables.
    pattern = re.compile(
        r"^[ \t]{0,8}item[ \t]+(\d+[a-z]?)[ \t]*[.\-–—\s]",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    chunks: dict[str, list[str]] = {}
    for i, match in enumerate(matches):
        item_key = match.group(1).lower()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[content_start:content_end].strip()
        chunks.setdefault(item_key, []).append(chunk)

    return chunks


def extract_sections(filing: FilingMeta) -> dict[str, FilingSection]:
    """
    Download a filing and extract its named sections.

    Returns a dict keyed by section name (e.g. "risk_factors").
    Sections absent from the filing are returned with present=False and empty text.
    """
    form_key = filing.form_type.upper()
    targets = _TARGETS.get(form_key)
    if targets is None:
        raise ValueError(
            f"Form type '{filing.form_type}' is not supported. "
            f"Supported: {list(_TARGETS.keys())}"
        )

    html = fetch_filing_html(filing)
    text = _html_to_text(html)
    chunks = _split_by_items(text)

    result: dict[str, FilingSection] = {}
    for item_key, (section_name, label) in targets.items():
        candidates = chunks.get(item_key, [])

        if not candidates:
            result[section_name] = FilingSection(
                name=section_name, label=label,
                text="", filing_date=filing.filing_date,
            )
            continue

        # The longest chunk is the actual section body (not the short ToC entry)
        best = max(candidates, key=len)
        result[section_name] = FilingSection(
            name=section_name, label=label,
            text=best, filing_date=filing.filing_date,
        )

    return result


# ---------------------------------------------------------------------------
# CLI entry-point (Stage 2 verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python section_extractor.py <TICKER> <FORM_TYPE>")
        print("Example: python section_extractor.py AAPL 10-K")
        sys.exit(1)

    ticker, form_type = sys.argv[1], sys.argv[2].upper()

    try:
        cik, company_name = resolve_cik(ticker)
    except (ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n{company_name} ({ticker.upper()}) — {form_type}")
    print("=" * 60)

    try:
        filings = get_recent_filings(cik, form_type, n=2)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Filing 1 (newer): {filings[0].filing_date}  {filings[0].document_url}")
    print(f"Filing 2 (older): {filings[1].filing_date}  {filings[1].document_url}")

    all_sections: list[dict[str, FilingSection]] = []
    for i, filing in enumerate(filings):
        label = "newer" if i == 0 else "older"
        print(f"\nDownloading and parsing {label} filing ({filing.filing_date})...")
        sections = extract_sections(filing)
        all_sections.append(sections)

    newer, older = all_sections[0], all_sections[1]
    targets = _TARGETS.get(form_type, {})

    print(f"\n{'Section':<34} {'Newer':>12}   {'Older':>12}")
    print("-" * 62)

    for item_key, (section_name, label) in targets.items():
        n_sec = newer.get(section_name)
        o_sec = older.get(section_name)
        n_str = f"{n_sec.word_count:,} words" if n_sec and n_sec.present else "MISSING"
        o_str = f"{o_sec.word_count:,} words" if o_sec and o_sec.present else "MISSING"
        print(f"  {label:<32} {n_str:>12}   {o_str:>12}")

    # Spot-check: print the first 300 chars of Risk Factors from the newer filing
    rf = newer.get("risk_factors")
    if rf and rf.present:
        print(f"\n--- Risk Factors snippet (newer filing) ---")
        print(rf.text[:300].strip())
        print("...")

    print("\nStage 2 OK — sections extracted from both filings.")
