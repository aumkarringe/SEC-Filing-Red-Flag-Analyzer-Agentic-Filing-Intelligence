"""
models.py — shared data structures for the SEC Red-Flag Agent

Imported by edgar_client, section_extractor, differ, and agent.
No project-level imports here to avoid circular dependencies.

Dataclasses are used for internal pipeline objects (FilingMeta, FilingSection, *Diff).
Pydantic BaseModels are used for LLM-facing schemas (RedFlag, RedFlagReport)
because ADK's output_schema parameter requires Pydantic.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from typing import Dict

from pydantic import BaseModel


@dataclass
class FilingMeta:
    """Metadata for a single SEC filing — enough to fetch its documents."""
    cik: str               # zero-padded 10-digit CIK
    accession_number: str  # e.g. "0000320193-24-000001"
    form_type: str         # e.g. "10-K"
    filing_date: str       # ISO date, e.g. "2024-10-31"
    primary_document: str  # filename, e.g. "aapl-20240928.htm"
    company_name: str = ""

    @property
    def document_url(self) -> str:
        """Construct the EDGAR Archives URL for the primary document."""
        acc_path = self.accession_number.replace("-", "")
        cik_int = str(int(self.cik))  # Archives path uses un-padded CIK
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{acc_path}/{self.primary_document}"
        )


@dataclass
class FilingSection:
    """Plain-text content of one named section from a filing."""
    name: str         # machine key, e.g. "risk_factors"
    label: str        # human label, e.g. "Item 1A — Risk Factors"
    text: str         # extracted plain text; empty string if section not found
    filing_date: str  # ISO date of the parent filing

    @property
    def present(self) -> bool:
        return bool(self.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0


# ---------------------------------------------------------------------------
# Diff structures (Stage 3+)
# ---------------------------------------------------------------------------

@dataclass
class ParagraphChange:
    """A single paragraph-level change between two filings."""
    change_type: str   # "added" | "removed" | "changed"
    old_text: str      # text from older filing; empty string if change_type == "added"
    new_text: str      # text from newer filing; empty string if change_type == "removed"
    # 0.0–1.0: how similar old_text and new_text are.
    # 0.0 for pure adds/removes; closer to 1.0 = only minor rewording.
    similarity: float = 0.0


@dataclass
class SectionDiff:
    """All paragraph-level changes for one section across two filings."""
    section_name: str
    section_label: str
    newer_date: str
    older_date: str
    older_word_count: int
    newer_word_count: int
    changes: list[ParagraphChange] = field(default_factory=list)

    @property
    def word_count_delta(self) -> int:
        return self.newer_word_count - self.older_word_count

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)


@dataclass
class FilingDiff:
    """Complete structured diff between two consecutive filings of the same type."""
    ticker: str
    company_name: str
    form_type: str
    newer_date: str
    older_date: str
    section_diffs: dict[str, SectionDiff] = field(default_factory=dict)
    # Map of filing_date -> bool indicating whether the raw document came from disk cache
    cached_map: Dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM output schemas (Stage 4+) — Pydantic required for ADK output_schema
# ---------------------------------------------------------------------------

class RedFlag(BaseModel):
    """A single potential red flag identified by the agent."""
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]  # how certain is this a real new risk
    category: Literal[
        "NEW_RISK", "EXPANDED_RISK", "LITIGATION", "GOING_CONCERN",
        "LIQUIDITY_CHANGE", "DEBT_CHANGE", "AUDITOR_CHANGE",
        "MANAGEMENT_CHANGE", "OTHER"
    ]
    section: str        # e.g. "Item 1A — Risk Factors"
    headline: str       # one-line summary, ≤ 15 words
    explanation: str    # 2-4 sentences: what changed and why it's concerning
    evidence_newer: str  # verbatim quoted snippet from the newer filing (≤ 300 chars)
    evidence_older: str  # verbatim quoted snippet from the older filing; "" if entirely new


class RedFlagReport(BaseModel):
    """The agent's complete analysis output for one filing comparison."""
    ticker: str
    company_name: str
    form_type: str
    newer_date: str
    older_date: str
    flags: list[RedFlag]
    analyst_summary: str  # 2-3 sentence overall assessment of the filing changes
