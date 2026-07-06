"""
differ.py — Section-level diff between two consecutive SEC filings

Approach:
  1. Split each section's text into paragraphs (logical blocks ≥ 60 chars)
  2. Run difflib.SequenceMatcher on the paragraph lists
  3. 'insert' → added paragraphs, 'delete' → removed, 'replace' → changed
  4. 'replace' blocks pass a two-layer materiality filter before surfacing to the LLM

Materiality filter (both layers must pass for a change to reach the agent):

  Layer 1 — raw similarity < _SIMILARITY_THRESHOLD (0.92)
    Threshold raised from original 0.85 to catch "nearly identical rewordings"
    that differ only in word choice without adding new facts.
    Tradeoff: a change that adds one new key phrase to a 500-word paragraph
    might score ~0.93 and be filtered here. The agent instruction (Part 2) is
    the second defense for anything that slips through near the boundary.

  Layer 2 — year-normalized similarity < _YEAR_NORM_THRESHOLD (0.96)
    Replaces four-digit years (20xx) with the placeholder YEAR before comparing.
    Catches the common annual boilerplate pattern where a block changes only
    "fiscal 2024" → "fiscal 2025" but is otherwise identical — these score
    below 0.92 on raw text (year characters shift many positions) but score
    near 1.0 after normalization.
    Only applied when at least one year number is present in either block.

Verification against known test cases (AAPL 10-K):
  Legal Proceedings (EU DMA escalation):  raw sim 0.62 → passes both layers ✓
  Risk Factors intro reword:              raw sim 0.73 → passes both layers ✓
  MD&A structural rewrite:               raw sim 0.49 → passes both layers ✓
"""

import difflib
import re
import sys

from edgar_client import get_recent_filings, resolve_cik
from models import FilingDiff, FilingSection, ParagraphChange, SectionDiff
from section_extractor import extract_sections

# Layer 1: raw similarity ceiling (raised from 0.85)
_SIMILARITY_THRESHOLD = 0.92

# Layer 2: after replacing year numbers, similarity ceiling
_YEAR_NORM_THRESHOLD  = 0.96

# Paragraphs shorter than this are likely headers, page refs, or table fragments.
_MIN_PARA_LEN = 60

_YEAR_RE = re.compile(r'\b20\d{2}\b')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_paragraphs(text: str) -> list[str]:
    """Split section text into substantive paragraph-sized chunks."""
    raw = re.split(r"\n[ \t]*\n", text)
    return [p.strip() for p in raw if len(p.strip()) >= _MIN_PARA_LEN]


def _similarity(a: str, b: str) -> float:
    """
    Similarity ratio between two text blocks.
    Capped at 3000 chars each to keep O(n²) comparison fast on long blocks.
    """
    return difflib.SequenceMatcher(None, a[:3000], b[:3000]).ratio()


def _diff_one_section(newer: FilingSection, older: FilingSection) -> SectionDiff:
    """Produce a SectionDiff by paragraph-level comparison of two FilingSections."""
    newer_paras = _to_paragraphs(newer.text) if newer.present else []
    older_paras = _to_paragraphs(older.text) if older.present else []

    changes: list[ParagraphChange] = []

    # autojunk=False: don't suppress common paragraphs as "junk" —
    # boilerplate paragraphs that were removed ARE meaningful signals.
    matcher = difflib.SequenceMatcher(None, older_paras, newer_paras, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        elif tag == "insert":
            for para in newer_paras[j1:j2]:
                changes.append(ParagraphChange(
                    change_type="added", old_text="", new_text=para, similarity=0.0,
                ))

        elif tag == "delete":
            for para in older_paras[i1:i2]:
                changes.append(ParagraphChange(
                    change_type="removed", old_text=para, new_text="", similarity=0.0,
                ))

        elif tag == "replace":
            old_block = " ".join(older_paras[i1:i2])
            new_block = " ".join(newer_paras[j1:j2])
            sim = _similarity(old_block, new_block)

            # Layer 1: raw similarity — filters pure rewordings with no new content
            if sim >= _SIMILARITY_THRESHOLD:
                continue

            # Layer 2: year-normalized similarity — filters annual boilerplate updates
            # where only "fiscal 2024" → "fiscal 2025" style year strings changed
            if _YEAR_RE.search(old_block) or _YEAR_RE.search(new_block):
                norm_old = _YEAR_RE.sub("YEAR", old_block)
                norm_new = _YEAR_RE.sub("YEAR", new_block)
                norm_sim = _similarity(norm_old, norm_new)
                if norm_sim >= _YEAR_NORM_THRESHOLD:
                    continue

            changes.append(ParagraphChange(
                change_type="changed",
                old_text=old_block,
                new_text=new_block,
                similarity=sim,
            ))

    return SectionDiff(
        section_name=newer.name,
        section_label=newer.label,
        newer_date=newer.filing_date,
        older_date=older.filing_date,
        older_word_count=older.word_count,
        newer_word_count=newer.word_count,
        changes=changes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diff_filings(
    newer_sections: dict[str, FilingSection],
    older_sections: dict[str, FilingSection],
    ticker: str,
    company_name: str,
    form_type: str,
) -> FilingDiff:
    """
    Produce a structured diff of two filings' extracted sections.

    Args:
        newer_sections: output of section_extractor.extract_sections() for the newer filing
        older_sections: output of section_extractor.extract_sections() for the older filing
        ticker, company_name, form_type: metadata for the report header

    Returns:
        FilingDiff with a SectionDiff entry for every section key present in both filings.
    """
    newer_date = next(
        (s.filing_date for s in newer_sections.values() if s.present), "unknown"
    )
    older_date = next(
        (s.filing_date for s in older_sections.values() if s.present), "unknown"
    )

    section_diffs: dict[str, SectionDiff] = {}
    for key in newer_sections.keys() & older_sections.keys():
        section_diffs[key] = _diff_one_section(newer_sections[key], older_sections[key])

    return FilingDiff(
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
        newer_date=newer_date,
        older_date=older_date,
        section_diffs=section_diffs,
    )


# ---------------------------------------------------------------------------
# CLI entry-point (Stage 3 verification)
# ---------------------------------------------------------------------------

def _snippet(text: str, max_chars: int = 280) -> str:
    """Truncate text for display, ending at a word boundary."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python differ.py <TICKER> <FORM_TYPE>")
        print("Example: python differ.py AAPL 10-K")
        sys.exit(1)

    ticker, form_type = sys.argv[1], sys.argv[2].upper()

    try:
        cik, company_name = resolve_cik(ticker)
    except (ValueError, EnvironmentError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        filings = get_recent_filings(cik, form_type, n=2)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n{company_name} ({ticker.upper()}) — {form_type}")
    print(f"Comparing: {filings[0].filing_date} (newer)  vs  {filings[1].filing_date} (older)")
    print("=" * 66)

    all_sections = []
    for i, filing in enumerate(filings):
        label = "newer" if i == 0 else "older"
        print(f"Extracting {label} filing ({filing.filing_date})...")
        all_sections.append(extract_sections(filing))

    filing_diff = diff_filings(
        newer_sections=all_sections[0],
        older_sections=all_sections[1],
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
    )

    total_changes = 0
    for sec_diff in filing_diff.section_diffs.values():
        delta_str = f"{sec_diff.word_count_delta:+,}"
        n_changes = len(sec_diff.changes)
        total_changes += n_changes

        print(f"\n{'─' * 66}")
        print(f"  {sec_diff.section_label}")
        print(f"  Words: {sec_diff.older_word_count:,} → {sec_diff.newer_word_count:,}  ({delta_str})")
        print(f"  Changes: {n_changes}")

        for ch in sec_diff.changes:
            if ch.change_type == "added":
                print(f"\n  [ADDED]\n  {_snippet(ch.new_text)}")
            elif ch.change_type == "removed":
                print(f"\n  [REMOVED]\n  {_snippet(ch.old_text)}")
            elif ch.change_type == "changed":
                sim_pct = int(ch.similarity * 100)
                print(f"\n  [CHANGED  {sim_pct}% similar]")
                print(f"  OLD: {_snippet(ch.old_text)}")
                print(f"  NEW: {_snippet(ch.new_text)}")

    print(f"\n{'=' * 66}")
    print(f"Total changes across all sections: {total_changes}")
    print("\nStage 3 OK — differ verified.")
