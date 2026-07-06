"""
agent.py — SEC Red-Flag Agent (Stage 4)

Architecture:
  - Python pipeline (Stages 1-3) runs deterministically → produces a FilingDiff
  - A Google ADK LlmAgent with LiteLlm backend receives the formatted diff
    and reasons about which changes constitute red flags
  - LiteLlm routes the call to OpenRouter (supports GPT-4o, Claude, Gemini, etc.)
  - Structured output is enforced via output_schema (Pydantic) and a JSON schema
    in the instruction as a belt-and-suspenders fallback

Requires:
  OPENROUTER_API_KEY in .env  — get one at https://openrouter.ai/keys
  OPENROUTER_MODEL (optional) — defaults to openrouter/openai/gpt-4o-mini
                                 use openrouter/openai/gpt-4o for higher quality
"""

import asyncio
import json
import os
import re
import sys

from dotenv import load_dotenv
from datetime import datetime, timezone
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from differ import diff_filings
from edgar_client import get_recent_filings, resolve_cik
from edgar_client import is_cached
from models import FilingDiff, RedFlagReport
from section_extractor import extract_sections

load_dotenv()

_DEFAULT_MODEL = "openrouter/openai/gpt-4o-mini"
_MODEL = os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)
_REPORT_CACHE_DIR = os.environ.get("EDGAR_CACHE_DIR", ".cache") + "/reports"

# ---------------------------------------------------------------------------
# Agent instruction — defines the red-flag taxonomy and output rules
# ---------------------------------------------------------------------------

_BASE_INSTRUCTION = """
You are a financial analyst reviewing SEC filing comparison summaries.
Your job is to identify genuine material changes that increase investor risk.
You will be shown text blocks marked ADDED, REMOVED, or CHANGED between two consecutive
filings of the same type (e.g. two annual 10-Ks).

WHAT QUALIFIES AS A RED FLAG:
A change qualifies ONLY if it introduces a concrete new fact that did not exist in the prior
filing. Ask yourself: "Is there a specific new fact here — a new dollar amount, a new named
proceeding, a new covenant breach, a new regulator, a new going-concern phrase — that an
investor reading both filings would consider materially worse?"

If yes, flag it. If no, do not flag it.

RED FLAG CATEGORIES:
- NEW_RISK:         A risk factor explicitly present in the newer filing that did not exist before
- EXPANDED_RISK:    An existing risk that now names a new enforcement action, quantified loss,
                    new regulator, or explicitly broader scope — NOT just reworded description
- LITIGATION:       New named legal proceedings, new regulatory investigations, new fines
- GOING_CONCERN:    Any language about ability to continue as a going concern
- LIQUIDITY_CHANGE: Specific deterioration in cash, working capital, or credit facility access
- DEBT_CHANGE:      New debt facilities, covenant changes, credit rating downgrades
- AUDITOR_CHANGE:   Change in auditor, new qualification, new material weakness
- MANAGEMENT_CHANGE: Named executive departure or unexpected board change
- OTHER:            Genuinely concerning material change not fitting above categories

SEVERITY:
- HIGH:   Immediate, direct threat to business viability or investor value
- MEDIUM: Significant new concrete risk requiring close attention
- LOW:    Specific new fact that may develop into a larger issue

CONFIDENCE (separate from severity):
- HIGH:   The new fact is unambiguous and clearly present in the provided text
- MEDIUM: The change is present but context is partial — interpretation requires some inference
- LOW:    The change is subtle; a reasonable analyst might disagree it's a red flag

HARD RULES — READ THESE BEFORE WRITING A SINGLE FLAG:

RULE 1 — EVIDENCE PAIR MUST BE THE SAME TOPIC.
evidence_older and evidence_newer must both discuss the same underlying risk or subject.
Do NOT pull the closest-sounding old sentence from a different part of the section and
pair it with a new sentence that is about something else entirely. If the CHANGED block
has new text on Topic A and the old text was on Topic B, the correct category is NEW_RISK
with evidence_older left empty (""), not EXPANDED_RISK with a mismatched pair.
Test before writing: read both quotes aloud. Would a reader agree they are the same risk?
If not, do not create the pair.

RULE 2 — EVERY CLAIM IN YOUR OUTPUT MUST TRACE TO WORDS IN YOUR EVIDENCE QUOTES.
The headline, explanation, and category must be supported exclusively by words that appear
in evidence_newer or evidence_older. You may NOT draw on your knowledge of the company,
its products, its industry, or prior SEC filings to add specificity. If you want to name
a product, technology, regulatory body, or business unit — it must appear verbatim in the
quoted text. If the evidence quotes do not contain the specific word or phrase, do not
assert it. A claim with no supporting text in the evidence is a fabrication.
Examples of forbidden behavior:
  - evidence quotes mention "products and services" → explanation says "autonomous vehicles"
  - evidence quotes mention "AI systems" → explanation says "Azure cloud platform"
  - evidence quotes have no dollar amount → headline says "significant fine"

RULE 3 — CONFIDENCE IS DETERMINED BY THE CONTENT OF EVIDENCE, NOT BY YOUR CERTAINTY.
Confidence is not about how confident YOU are. It is a mechanical label determined by
what is literally present in evidence_newer. Apply it as follows:

  STEP 1: Look at evidence_newer. Does it contain any of:
    (a) A specific monetary amount (e.g. "€500 million", "$2 billion", "30%")
    (b) A specific calendar date tied directly to a regulatory or legal outcome
        (e.g. "On April 23, 2025, the Commission fined...")
    (c) An explicit statement of a final enforcement order or judgment
  → If YES to any of (a), (b), or (c): confidence = HIGH
  → If NO to all of (a), (b), (c): confidence CANNOT be HIGH. Go to Step 2.

  STEP 2: Is the new language unambiguous, clearly directional, and was it clearly
  absent from the older text?
  → If YES: confidence = MEDIUM
  → If the change is subtle, a wording adjustment, or involves a single added word
    or phrase: confidence = LOW

Do not override this mechanical rule. Even if you are personally very confident that
the flag is real, if evidence_newer lacks (a), (b), and (c), confidence is MEDIUM or LOW.
"HIGH confidence" means the evidence is self-evidently conclusive — a fine amount, a date
tied to an outcome, a final order. It does not mean you found good evidence.

RULE 4 — CATEGORY DEFINITIONS ARE STRICT. DO NOT STRETCH THEM.

  GOING_CONCERN requires the text to contain language explicitly about the company's
  ability to continue as a going concern or to fund its ongoing operations. Language
  about macroeconomic uncertainty, tariffs, trade policy, supply chains, or cost
  pressures does NOT qualify as GOING_CONCERN — use OTHER or NEW_RISK instead.

  LIQUIDITY_CHANGE requires the evidence to discuss actual deterioration in cash,
  access to credit facilities, covenant compliance, or working capital. Accounting
  policy disclosures (how the company measures goodwill, equity investments, etc.)
  are NOT LIQUIDITY_CHANGE — they are accounting methodology, which is not a
  liquidity signal.

  NOT A RED FLAG AT ALL — do not flag any of the following regardless of category:
    • Removal or addition of a cross-reference sentence pointing to another section
      (e.g. "Refer to Note 1..." or "See Part II, Item 8...")
    • Section reorganization where the same information moved to a different location
    • Standard boilerplate about critical accounting estimates or management judgment
      that every company includes in MD&A
    • Year-number updates in boilerplate language (e.g. "fiscal 2025" replacing "fiscal 2024")

RULE 5 — SAME MEANING IN DIFFERENT WORDS IS NOT A RED FLAG.
If the old and new text convey the same risk using different phrasing, do not flag it.

RULE 6 — VERBATIM EVIDENCE ONLY.
Copy evidence_newer and evidence_older character-for-character from the provided text.
Do not paraphrase, summarize, or shorten by substituting synonyms.

RULE 7 — EMPTY FLAGS IS A VALID CORRECT ANSWER.
If no change clears all rules above, return flags: []. Do not manufacture flags.
""".strip()


def _build_instruction() -> str:
    """Embed the Pydantic JSON schema in the instruction for models that ignore output_schema."""
    schema = json.dumps(RedFlagReport.model_json_schema(), indent=2)
    return (
        _BASE_INSTRUCTION
        + "\n\n"
        + "OUTPUT FORMAT — respond with ONLY valid JSON, no prose, no code fences:\n"
        + schema
    )


# ---------------------------------------------------------------------------
# Prompt formatter — converts FilingDiff → readable context for the LLM
# ---------------------------------------------------------------------------

def _format_diff_prompt(diff: FilingDiff) -> str:
    """Convert a FilingDiff into a structured prompt for the agent."""
    lines = [
        f"COMPANY: {diff.company_name} ({diff.ticker})",
        f"FILING TYPE: {diff.form_type}",
        f"NEWER FILING: {diff.newer_date}",
        f"OLDER FILING:  {diff.older_date}",
        "",
        "CHANGES DETECTED BETWEEN FILINGS",
        "=" * 50,
    ]

    for sec_diff in diff.section_diffs.values():
        delta_str = f"{sec_diff.word_count_delta:+,}"
        lines += [
            "",
            f"[SECTION: {sec_diff.section_label}]",
            f"Word count: {sec_diff.older_word_count:,} → {sec_diff.newer_word_count:,}  ({delta_str} words)",
        ]

        if not sec_diff.has_changes:
            lines.append("No substantive changes detected in this section.")
            continue

        for ch in sec_diff.changes:
            if ch.change_type == "added":
                lines += [
                    f"\n[ADDED — present in {diff.newer_date}, absent from {diff.older_date}]",
                    ch.new_text[:2000],
                ]
            elif ch.change_type == "removed":
                lines += [
                    f"\n[REMOVED — present in {diff.older_date}, absent from {diff.newer_date}]",
                    ch.old_text[:2000],
                ]
            elif ch.change_type == "changed":
                sim_pct = int(ch.similarity * 100)
                lines += [
                    f"\n[CHANGED — {sim_pct}% text similarity between filings]",
                    f"OLDER ({diff.older_date}):",
                    ch.old_text[:1500],
                    f"NEWER ({diff.newer_date}):",
                    ch.new_text[:1500],
                ]

    lines += [
        "",
        "=" * 50,
        "Produce a RedFlagReport JSON object for the changes above.",
        "If nothing warrants a red flag, return flags: [] with an appropriate analyst_summary.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction — handles models that wrap output in code fences
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """
    Pull a JSON object out of text that may be wrapped in markdown code blocks.
    Returns the raw JSON string for Pydantic to parse.
    """
    # Strip ```json ... ``` or ``` ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Find the outermost { ... } if no fences
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text  # let Pydantic raise a clear error


# ---------------------------------------------------------------------------
# ADK agent builder
# ---------------------------------------------------------------------------

def _build_agent() -> LlmAgent:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file. Get a key at https://openrouter.ai/keys"
        )
    # LiteLlm reads OPENROUTER_API_KEY from environment automatically
    os.environ["OPENROUTER_API_KEY"] = api_key

    return LlmAgent(
        name="redflag_analyzer",
        model=LiteLlm(model=_MODEL),
        instruction=_build_instruction(),
        output_schema=RedFlagReport,  # enforces schema for models that support it
        output_key="report",          # also stores result in session state
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_analysis_async(
    ticker: str, form_type: str
) -> tuple[RedFlagReport, FilingDiff]:
    """
    Full pipeline: CIK resolution → filing fetch → section extraction
    → diff → LLM red-flag analysis.

    Returns (RedFlagReport, FilingDiff) so callers can display both the
    agent's flags and the raw section-level statistics.
    """
    print(f"[1/4] Resolving CIK for {ticker.upper()}...")
    cik, company_name = resolve_cik(ticker)
    print(f"      {company_name}  (CIK {cik})")

    print(f"[2/4] Fetching two most recent {form_type} filings...")
    filings = get_recent_filings(cik, form_type, n=2)
    print(f"      Newer: {filings[0].filing_date}   Older: {filings[1].filing_date}")

    print(f"[3/4] Extracting and diffing sections...")
    newer_sections = extract_sections(filings[0])
    older_sections = extract_sections(filings[1])
    filing_diff = diff_filings(
        newer_sections=newer_sections,
        older_sections=older_sections,
        ticker=ticker.upper(),
        company_name=company_name,
        form_type=form_type.upper(),
    )
    # Record whether each filing was served from the on-disk cache (helpful for UI)
    filing_diff.cached_map = {
        filings[0].filing_date: is_cached(filings[0].document_url),
        filings[1].filing_date: is_cached(filings[1].document_url),
    }
    total_changes = sum(len(sd.changes) for sd in filing_diff.section_diffs.values())
    print(f"      {total_changes} substantive change(s) across {len(filing_diff.section_diffs)} section(s)")

    # Ensure report cache directory exists
    try:
        os.makedirs(_REPORT_CACHE_DIR, exist_ok=True)
    except Exception:
        # best-effort; continue if we cannot create cache dir
        pass

    # If a cached full report exists for this ticker/form/newer_date, load and return it
    report_cache_path = os.path.join(
        _REPORT_CACHE_DIR, f"{ticker.upper()}_{form_type.upper()}_{filings[0].filing_date}.json"
    )
    if not os.environ.get("EDGAR_NO_CACHE") and os.path.exists(report_cache_path):
        print(f"[4/4] Loading cached RedFlagReport from {report_cache_path}...")
        try:
            with open(report_cache_path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            report = RedFlagReport.model_validate_json(raw)
            # Mark that a precomputed report was used and record its timestamp
            filing_diff.cached_map["report"] = True
            try:
                mtime = os.path.getmtime(report_cache_path)
                filing_diff.cached_map["report_cached_at"] = datetime.fromtimestamp(mtime, timezone.utc).isoformat()
            except Exception:
                filing_diff.cached_map["report_cached_at"] = None
            return report, filing_diff
        except Exception as e:
            print(f"Warning: failed to load cached report: {e}")

    print(f"[4/4] Running LLM analysis ({_MODEL})...")
    agent = _build_agent()
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="sec_redflag", session_service=session_service)
    session = await session_service.create_session(app_name="sec_redflag", user_id="analyst")

    prompt_text = _format_diff_prompt(filing_diff)

    report: RedFlagReport | None = None
    # Consume all events rather than breaking early — avoids ADK tracing cleanup noise.
    async for event in runner.run_async(
        user_id="analyst",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt_text)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            if report is None:  # take the first final response, ignore subsequent
                raw = event.content.parts[0].text
                report = RedFlagReport.model_validate_json(_extract_json(raw))

    if report is None:
        # Fallback: ADK may have stored result in session state via output_key
        saved = await session_service.get_session(
            app_name="sec_redflag", user_id="analyst", session_id=session.id
        )
        report_data = saved.state.get("report") if saved else None
        if report_data is None:
            raise RuntimeError(
                "Agent produced no output. "
                "Check OPENROUTER_API_KEY and that the model is accessible."
            )
        report = RedFlagReport.model_validate(report_data)

    # Persist the produced report for future fast responses (best-effort)
    try:
        if not os.environ.get("EDGAR_NO_CACHE"):
            with open(report_cache_path, "w", encoding="utf-8") as fh:
                fh.write(report.model_dump_json(indent=2))
            # indicate we have a saved report (this run used a freshly generated report)
            filing_diff.cached_map["report"] = False
            filing_diff.cached_map["report_cached_at"] = datetime.now(timezone.utc).isoformat()
    except Exception:
        pass

    return report, filing_diff


def run_analysis(ticker: str, form_type: str) -> tuple[RedFlagReport, FilingDiff]:
    """Synchronous wrapper — use this from non-async code (Streamlit, CLI)."""
    return asyncio.run(run_analysis_async(ticker, form_type))


# ---------------------------------------------------------------------------
# CLI entry-point (Stage 4 verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent.py <TICKER> <FORM_TYPE>")
        print("Example: python agent.py AAPL 10-K")
        sys.exit(1)

    ticker_arg, form_arg = sys.argv[1], sys.argv[2]

    try:
        report, _ = run_analysis(ticker_arg, form_arg)
    except (ValueError, EnvironmentError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 66)
    print(f"RED FLAG REPORT: {report.company_name} ({report.ticker}) — {report.form_type}")
    print(f"Comparing {report.newer_date} (newer) vs {report.older_date} (older)")
    print("=" * 66)

    if not report.flags:
        print("\nNo red flags identified.")
    else:
        sev_icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        for i, flag in enumerate(report.flags, 1):
            icon = sev_icons.get(flag.severity, "")
            print(f"\nFlag {i}/{len(report.flags)}  [{icon} {flag.severity}]  confidence={flag.confidence}  {flag.category}")
            print(f"  {flag.headline}")
            print(f"  Section: {flag.section}")
            print(f"  {flag.explanation}")
            if flag.evidence_newer:
                print(f"  NEWER: \"{flag.evidence_newer[:200]}\"")
            if flag.evidence_older:
                print(f"  OLDER: \"{flag.evidence_older[:200]}\"")

    print(f"\nSUMMARY: {report.analyst_summary}")
    print("\n--- JSON ---")
    print(report.model_dump_json(indent=2))
    print("\nStage 4 OK.")
