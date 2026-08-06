"""
====================================================================================
mf_overrides.py — HAND-CURATED UNIVERSE OVERRIDES (git-tracked, validated)
====================================================================================
WHY THIS EXISTS
---------------
Some facts the pipeline needs are published by NO free source and can only come from
a human reading an AMC factsheet or SID. The binding case today is SECTOR: a
Sectoral/Thematic fund's cohort is keyed on `("sector", ...)`, the trained manifest's
52 sector values are hand-typed, and AMFI's NAVAll carries no sector field at all.
Without a sector, 257 of the 572 canonical scoreable schemes have no honest cohort
and `mf_live_score` correctly refuses them with SECTOR_UNRESOLVED.

This module is where that human knowledge lives, and the guard rail around it.

WHY IT IS GIT-TRACKED AND NOT IN mf_cache/
------------------------------------------
`mf_cache/` is gitignored and documented as regenerable from `bootstrap.py`. Curated
labels are the exact opposite: unbackfillable once lost, because no fetch reproduces
them. The prediction ledger was moved out of `mf_cache/` for precisely this reason
(see `ledger/predictions.jsonl`), and this file follows it. Note the nightly CI
restores `mf_cache/` from `actions/cache`, which GitHub evicts on idle or size
pressure — anything curated in there is one eviction from gone.

WHAT AN OVERRIDE MAY AND MAY NOT DO (the honesty boundary)
----------------------------------------------------------
An override supplies INPUTS ONLY — `category` and `sector`. It can never set a
label, a probability, or a verdict. If it could, the file would become a way to
launder a desired answer past the screen, which is the one thing the whole project
is built to prevent. `validate()` enforces this structurally: any unknown column is
an error, not a warning.

Further checks, each guarding a real failure mode rather than tidiness:
  * `category` must already exist in the TRAINED manifest. Hand-writing a fund into
    a cohort the model never trained on would fabricate exactly the transfer of
    skill the OUT_OF_TRAINING_UNIVERSE gate exists to refuse.
  * `sector` should match a sector the manifest already uses. A typo silently
    creates a one-member cohort, which then fails downstream as THIN_COHORT — a
    misleading reason that sends you looking in the wrong place.
  * `source_url` is required, mirroring `managers.csv`'s "every row traceable to a
    real AMC factsheet/SID — never inferred" rule.
  * Duplicate `amfi_code` rows are an error: last-write-wins would make the file's
    effect depend on row order.
====================================================================================
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

OVERRIDES_PATH = Path("overrides/universe_overrides.csv")

# INPUTS only. `amfi_code` identifies the row; the rest are documentation.
REQUIRED_COLUMNS = ("amfi_code",)
VALUE_COLUMNS = ("category", "sector")
META_COLUMNS = ("source_url", "note")
ALLOWED_COLUMNS = REQUIRED_COLUMNS + VALUE_COLUMNS + META_COLUMNS

# Columns that would let the file dictate an OUTPUT rather than supply an input.
# Named explicitly so the error message can say why, instead of "unknown column".
_FORBIDDEN_COLUMNS = {
    "label", "y_cohort_q1", "y", "target", "probability", "prob", "score",
    "verdict", "verdict_color", "cohort_percentile", "rank",
}

SEVERITY_ERROR = "ERROR"      # override is DROPPED — never applied
SEVERITY_WARN = "WARN"        # applied, but surfaced


@dataclass(frozen=True)
class Issue:
    amfi_code: Optional[str]
    severity: str
    message: str

    def __str__(self) -> str:
        who = f"[{self.amfi_code}] " if self.amfi_code else ""
        return f"{self.severity}: {who}{self.message}"


@dataclass(frozen=True)
class Override:
    amfi_code: str
    category: Optional[str] = None
    sector: Optional[str] = None
    source_url: Optional[str] = None
    note: Optional[str] = None


def _clean(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def validate(frame: pd.DataFrame, trained_categories: Set[str],
             known_sectors: Set[str]) -> tuple[Dict[str, Override], List[Issue]]:
    """Pure validation. Returns (accepted overrides, issues).

    Rows with an ERROR are dropped, never partially applied — a half-accepted row
    is how a typo'd sector reaches a cohort assignment.
    """
    issues: List[Issue] = []
    accepted: Dict[str, Override] = {}

    if frame.empty:
        return accepted, issues

    cols = {str(c).strip().lower() for c in frame.columns}
    forbidden = cols & _FORBIDDEN_COLUMNS
    if forbidden:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"columns {sorted(forbidden)} would let this file set an OUTPUT "
                            "(a label/score/verdict). Overrides supply inputs only — "
                            "category and sector. Whole file rejected."))
        return {}, issues
    unknown = cols - set(ALLOWED_COLUMNS)
    if unknown:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"unknown columns {sorted(unknown)}; allowed: {list(ALLOWED_COLUMNS)}. "
                            "Whole file rejected rather than silently ignoring them."))
        return {}, issues
    if "amfi_code" not in cols:
        issues.append(Issue(None, SEVERITY_ERROR, "missing required column 'amfi_code'"))
        return {}, issues

    seen: Set[str] = set()
    for _, raw in frame.iterrows():
        code = _clean(raw.get("amfi_code"))
        if not code:
            issues.append(Issue(None, SEVERITY_ERROR, "row with blank amfi_code — dropped"))
            continue
        if code in seen:
            # last-write-wins would make the file's meaning depend on row order
            issues.append(Issue(code, SEVERITY_ERROR, "duplicate amfi_code — both rows dropped"))
            accepted.pop(code, None)
            continue
        seen.add(code)

        category = _clean(raw.get("category"))
        sector = _clean(raw.get("sector"))
        source_url = _clean(raw.get("source_url"))

        if category is None and sector is None:
            issues.append(Issue(code, SEVERITY_ERROR, "row sets neither category nor sector — dropped"))
            continue
        if category is not None and category not in trained_categories:
            issues.append(Issue(code, SEVERITY_ERROR,
                                f"category {category!r} is not in the trained universe "
                                f"({len(trained_categories)} categories). Hand-writing a fund into "
                                "an untrained cohort would fabricate model coverage — dropped."))
            continue
        if sector is not None and known_sectors and sector not in known_sectors:
            # WARN not ERROR: a genuinely new sector is legitimate, but a typo here
            # silently produces a one-member cohort that later fails as THIN_COHORT.
            issues.append(Issue(code, SEVERITY_WARN,
                                f"sector {sector!r} matches no sector already in the manifest. "
                                "If this is a typo it will create a phantom one-member cohort; "
                                f"known: {sorted(known_sectors)[:6]}..."))
        if not source_url:
            issues.append(Issue(code, SEVERITY_WARN,
                                "no source_url — every curated row should be traceable to a real "
                                "AMC factsheet/SID (same rule as managers.csv)"))

        accepted[code] = Override(amfi_code=code, category=category, sector=sector,
                                  source_url=source_url, note=_clean(raw.get("note")))
    return accepted, issues


def load(path: Path = OVERRIDES_PATH, *, trained_categories: Optional[Set[str]] = None,
         known_sectors: Optional[Set[str]] = None,
         manifest: Optional[pd.DataFrame] = None) -> tuple[Dict[str, Override], List[Issue]]:
    """Read + validate the overrides file. Absent file is normal, not an error."""
    if manifest is not None:
        trained_categories = trained_categories or {str(c) for c in manifest["category"].dropna().unique()}
        known_sectors = known_sectors if known_sectors is not None else {
            str(s) for s in manifest["sector"].dropna().unique()}
    if not Path(path).exists():
        return {}, []
    try:
        frame = pd.read_csv(path, dtype=str)
    except Exception as exc:  # noqa: BLE001 — a malformed hand-edited CSV must not crash a run
        return {}, [Issue(None, SEVERITY_ERROR, f"could not read {path}: {exc}")]
    return validate(frame, set(trained_categories or ()), set(known_sectors or ()))


def _selftest() -> None:
    trained = {"Flexi Cap", "Large Cap", "Sectoral/Thematic"}
    sectors = {"Banking/Financials", "Technology", "Pharma"}

    def frame(rows): return pd.DataFrame(rows)

    # 1. A good row is accepted.
    ok, issues = validate(frame([dict(amfi_code="100001", sector="Technology",
                                      source_url="https://amc.example/sid.pdf")]), trained, sectors)
    assert list(ok) == ["100001"] and ok["100001"].sector == "Technology"
    assert not [i for i in issues if i.severity == SEVERITY_ERROR], issues
    print("[selftest] a sourced sector override is accepted — PASS")

    # 2. THE HONESTY BOUNDARY: a file that tries to set an OUTPUT is rejected whole.
    for bad in ("label", "y_cohort_q1", "probability", "verdict", "score"):
        ok, issues = validate(frame([{"amfi_code": "100001", bad: "1"}]), trained, sectors)
        assert ok == {} and any(i.severity == SEVERITY_ERROR for i in issues), bad
        assert "inputs only" in " ".join(i.message for i in issues), bad
    print("[selftest] overrides that set a label/score/verdict are rejected wholesale — PASS")

    # 3. Cannot hand-write a fund into an UNTRAINED cohort.
    ok, issues = validate(frame([dict(amfi_code="100002", category="Dividend Yield",
                                      source_url="u")]), trained, sectors)
    assert ok == {} and any("not in the trained universe" in i.message for i in issues)
    print("[selftest] category outside the trained set is refused — PASS")

    # 4. A typo'd sector WARNs (would otherwise become a silent one-member cohort).
    ok, issues = validate(frame([dict(amfi_code="100003", sector="Tecnology", source_url="u")]),
                          trained, sectors)
    assert "100003" in ok, "a novel sector is legitimate — must not be dropped"
    assert any(i.severity == SEVERITY_WARN and "phantom" in i.message for i in issues)
    print("[selftest] unrecognised sector warns about phantom cohorts, still applies — PASS")

    # 5. Unsourced row warns; duplicate codes drop BOTH (order-independence).
    ok, issues = validate(frame([dict(amfi_code="100004", sector="Pharma")]), trained, sectors)
    assert "100004" in ok and any("source_url" in i.message for i in issues)
    ok, issues = validate(frame([dict(amfi_code="1", sector="Pharma", source_url="u"),
                                 dict(amfi_code="1", sector="Technology", source_url="u")]),
                          trained, sectors)
    assert ok == {} and any("duplicate" in i.message for i in issues), ok
    print("[selftest] unsourced row warns; duplicate amfi_code drops both — PASS")

    # 6. Unknown columns are rejected rather than silently ignored, and a row that
    #    sets nothing is dropped.
    ok, issues = validate(frame([dict(amfi_code="1", sectr="Pharma")]), trained, sectors)
    assert ok == {} and any("unknown columns" in i.message for i in issues)
    ok, issues = validate(frame([dict(amfi_code="1", source_url="u")]), trained, sectors)
    assert ok == {} and any("neither category nor sector" in i.message for i in issues)
    # A missing file is normal, never an error.
    assert load(Path("overrides/__definitely_absent__.csv"), trained_categories=trained,
                known_sectors=sectors) == ({}, [])
    print("[selftest] unknown column rejected, empty row dropped, absent file is not an error — PASS")

    # ---- sector proposer -----------------------------------------------------------
    amcs = {"quant", "Bajaj Finserv", "Quantum", "Nippon India", "Aditya Birla Sun Life",
            "Tata", "ICICI Prudential", "Kotak", "SBI", "Baroda BNP Paribas"}

    # THE TRAP THIS EXISTS FOR: `quant` is an AMC, not a sector claim. Every one of
    # these would be mislabeled `Quant` by a plain keyword scan.
    assert propose_sector("quant Healthcare Fund", amcs)[0] == "Pharma/Healthcare"
    assert propose_sector("quant BFSI Fund", amcs)[0] == "Banking/Financial Services"
    assert propose_sector("quant Infrastructure Fund", amcs)[0] == "Infrastructure"
    assert propose_sector("quant PSU Fund", amcs)[0] == "PSU"
    # ...while a real quant fund from another AMC still resolves to the Quant sector.
    assert propose_sector("SBI Quant Fund", amcs)[0] == "Quant"
    assert propose_sector("Tata Quant Fund", amcs)[0] == "Quant"
    # The same shape one AMC over: 'Finserv' is the house, not the mandate.
    assert propose_sector("Bajaj Finserv Consumption Fund", amcs)[0] == "FMCG/Consumption"
    assert propose_sector("Bajaj Finserv Healthcare Fund", amcs)[0] == "Pharma/Healthcare"
    # 'Quantum' must not be read as 'quant'.
    assert propose_sector("Quantum Ethical Fund", amcs)[0] is None

    # A strategy name is NOT a sector, even when it contains sector-ish words.
    for nm in ("ICICI Prudential Innovation Fund", "Kotak Active Momentum Fund",
               "SBI Quality Fund", "Baroda BNP Paribas Multi-Factor Fund",
               "Nippon India Japan Equity Fund", "Tata Ethical Fund"):
        sec, _, why = propose_sector(nm, amcs)
        assert sec is None and why, f"{nm} should be left for a human"
    # ...specifically: an ethical/Shariah screen must not be filed under the ESG sector.
    # (Assert on the SECTOR, not the reason text — the reason says the word "ESG"
    # precisely because it is explaining that this is not it.)
    assert propose_sector("Tata Ethical Fund", amcs)[0] != "ESG"
    assert propose_sector("Taurus Ethical Fund", amcs)[0] is None
    # ...but a genuine ESG-mandate fund still resolves.
    assert propose_sector("Kotak ESG Exclusionary Strategy Fund", amcs)[0] == "ESG"
    # ...and a sector-ROTATION fund must never be pinned to one sector.
    assert propose_sector("Shriram Multi Sector Rotation Fund", amcs)[0] is None

    # Ordinary sector reads still work, and report the token that fired.
    sec, matched, _ = propose_sector("Nippon India Pharma Fund", amcs)
    assert (sec, matched) == ("Pharma/Healthcare", "pharma")
    assert propose_sector("ICICI Prudential Transportation and Logistics Fund", amcs)[0] == "Auto/Transportation"
    assert propose_sector("Aditya Birla Sun Life Business Cycle Fund", amcs)[0] == "Business Cycle"

    # Longest-prefix stripping: 'Bajaj Finserv' must beat a hypothetical 'Bajaj'.
    assert strip_amc_prefix("Bajaj Finserv Consumption Fund", {"Bajaj", "Bajaj Finserv"}) == "Consumption Fund"

    # THE 24-FUND REGRESSION, locked. A bare `services` marker swallowed every
    # "Banking and Financial Services" fund — the largest clean group in the worklist —
    # and the first real run proposed exactly ONE financials fund out of 25.
    for nm in ("HDFC Banking & Financial Services Fund - Growth Option",
               "Canara Robeco Banking and Financials Services Fund - Direct",   # AMFI's own spelling
               "Motilal Oswal Financial Services Fund- Direct Growth",
               "Edelweiss Financial Services Fund - Direct Plan - Growth",
               "quant BFSI Fund - Growth Option - Direct Plan"):
        assert propose_sector(nm, amcs)[0] == "Banking/Financial Services", nm
    # ...while a STANDALONE services mandate is still correctly left for the human.
    for nm in ("Kotak Services Fund", "Axis Services Opportunities Fund - Direct Plan"):
        assert propose_sector(nm, amcs)[0] is None, nm

    # Plan/option tails must not decide anything, and must not truncate a fund whose
    # NAME contains an option word — 'Nippon India Growth Mid Cap Fund' is a real
    # manifest scheme, and cutting at the first 'Growth' would leave 'Nippon India'.
    assert normalize_scheme_name("Nippon India Growth Mid Cap Fund") == "Nippon India Growth Mid Cap Fund"
    assert normalize_scheme_name("Kotak Energy Opportunities Fund-Direct-Growth") == "Kotak Energy Opportunities Fund"
    assert normalize_scheme_name("SBI CONSUMPTION OPPORTUNITIES FUND - DIRECT PLAN - GROWTH") == "SBI CONSUMPTION OPPORTUNITIES FUND"
    # An 'Opportunities' name that DOES name a sector keeps its sector; only the
    # contentless 'Special Opportunities' is withheld.
    assert propose_sector("SBI CONSUMPTION OPPORTUNITIES FUND - DIRECT PLAN - GROWTH", amcs)[0] == "FMCG/Consumption"
    assert propose_sector("Tata Housing Opportunities Fund - Direct Plan", amcs)[0] == "Housing/Real Estate"
    assert propose_sector("Kotak Special Opportunities Fund - Direct Plan", amcs)[0] is None

    # A non-domestic mandate outranks a sector word ON PURPOSE. This fund names a
    # sector, and the proposer still withholds it: a global agri sleeve ranked against
    # Indian commodity funds is the wrong peer group, which is worse than a blank.
    sec, _, why = propose_sector("Aditya Birla Sun Life Commodity Equities Fund - Global Agri Plan", amcs)
    assert sec is None and "non-domestic" in why
    # 'Quantamental' and 'Quantum' both continue past \bquant\b, so neither is the
    # Quant sector — the boundary does the work of two lookaheads.
    assert propose_sector("quant Quantamental Fund - Growth Option", amcs)[0] is None
    assert propose_sector("Quantum India ESG Equity Fund", amcs)[0] == "ESG"
    print("[selftest] sector proposer: AMC prefixes stripped before matching (quant/"
          "Finserv/Quantum traps held), strategy names refused a sector — PASS")

    print("[selftest] PASS — mf_overrides accepts curated inputs and refuses curated ANSWERS")


# ====================================================================================
# SECTOR PROPOSER (--propose)
# ====================================================================================
# 205 blocked funds is too many to type from scratch and too few to guess at. This
# narrows the human's job WITHOUT taking the decision away: it emits SUGGESTIONS with
# the evidence that produced them, into a separate `_sector_proposals.csv`. Nothing
# here writes to OVERRIDES_PATH, and validate() still demands a source_url, so a
# proposal cannot reach a score without a human moving it across.
#
# WHY READING THE SCHEME NAME IS NOT "INVENTING A FACT"
# -----------------------------------------------------
# The honesty invariant forbids fabricating data. A scheme's NAME is not fabricated:
# it is the AMC's own published label, and under SEBI's categorization circular a
# fund named for a sector must hold >=80% of assets in that sector. "Nippon India
# Pharma Fund" being a pharma fund is a regulated claim by its issuer, not an
# inference of ours. What we must NOT do is stretch that to names that describe a
# STRATEGY rather than a sector — Innovation, Momentum, Special Opportunities,
# Multi-Factor, Quality — where no sector is named at all and any assignment would be
# our opinion wearing the AMC's clothes. Those return None and land in the human's
# pile, which is the entire point of separating confidence from coverage.
#
# THE AMC-PREFIX TRAP, which is why this is not a one-line keyword scan
# ---------------------------------------------------------------------
# `quant Mutual Fund` is a real AMC. A naive scan for "quant" tags `quant Healthcare
# Fund`, `quant BFSI Fund`, `quant Infrastructure Fund` and 7 more as the *Quant*
# sector — mislabeling 10 funds into a cohort they have nothing to do with, which is
# exactly the silent wrong-peer-group failure this file's docstring warns about.
# Same shape: `Bajaj Finserv Consumption Fund` is not a financials fund. So the AMC
# prefix is stripped FIRST, using the manifest's own `amc` column as the vocabulary.

# Ordered: the first pattern that matches wins, so put the specific before the general
# ("transportation and logistics" before "logistics"). Every sector on the right-hand
# side MUST already exist in the manifest — proposing a brand-new sector is a product
# decision (it creates a cohort), not something a regex gets to make.
SECTOR_RULES: tuple[tuple[str, str], ...] = (
    (r"banking|bfsi|financial services|financials services|financial svcs", "Banking/Financial Services"),
    (r"pharma|healthcare|health care|health and wellness", "Pharma/Healthcare"),
    (r"transportation|logistics|automotive|\bauto\b", "Auto/Transportation"),
    (r"technology|\bteck\b|\btech\b|digital", "IT/Technology"),
    (r"consumption|consumer|fmcg", "FMCG/Consumption"),
    (r"infrastructure|\binfra\b|t\.i\.g\.e\.r|economic reform|build india", "Infrastructure"),
    (r"manufacturing|manufacture", "Manufacturing"),
    (r"\bpsu\b|public sector", "PSU"),
    (r"\bmnc\b|multinational", "MNC"),
    (r"energy|\bpower\b", "Energy/Power"),
    (r"commodit|\bcomma\b|natural resource|agri", "Commodities/Natural Resources"),
    (r"housing|real estate|realty", "Housing/Real Estate"),
    (r"business cycle", "Business Cycle"),
    (r"\besg\b|sustainab|responsible investing", "ESG"),
    (r"defence|defense", "Defence"),
    # \b on both sides, not `quant(?!um)`: the boundary already excludes 'Quantum' AND
    # 'Quantamental' (both continue with a word char), so one idiom covers both instead
    # of a lookahead per exception. Runs after the AMC strip regardless.
    (r"\bquant\b", "Quant"),
    (r"export", "Exports"),
)

# Names that describe a STRATEGY, not a sector. Matched only to explain WHY a fund was
# left blank, so the human sees "no sector named" rather than an unexplained gap.
#
# Checked BEFORE the sector rules, deliberately: the conservative direction is to leave
# a fund for the human, not to reach for a sector. That ordering is also what makes the
# `services` entry below dangerous, and it cost 24 funds on the first real run.
_STRATEGY_MARKERS: tuple[tuple[str, str], ...] = (
    (r"innovation|innovative|pioneer", "innovation theme — names a strategy, not a sector"),
    (r"momentum", "momentum factor — names a factor, not a sector"),
    # `opportunities fund$` used to live here and was DEAD: every AMFI name carries a
    # trailing plan/option suffix, so nothing ever reached the anchor. Normalizing the
    # name (below) would have woken it up and made it actively wrong — it would have
    # blanked 'SBI Consumption Opportunities' and 'Tata Housing Opportunities', which
    # DO name a sector. Only the genuinely contentless 'Special Opportunities' remains;
    # a bare 'Opportunities Fund' now falls through to "no sector keyword", which is
    # the same outcome with an honest reason.
    (r"special opportunit", "opportunistic mandate — no sector named"),
    (r"multi.?factor|quantamental|minimum variance|\bquality\b", "factor strategy — no sector named"),
    (r"sector rotation", "rotates ACROSS sectors — a single sector would be wrong by construction"),
    (r"conglomerate", "conglomerates — cross-sector holding companies"),
    (r"ethical|shariah", "ethical/Shariah screen — a religious-law filter, NOT the ESG sector"),
    # Deliberately beats a sector match rather than losing to one. 'ABSL Commodity
    # Equities Fund - Global Agri Plan' DOES name a sector, and filing it under the
    # domestic Commodities cohort would rank a global agri sleeve against Indian
    # commodity funds — the wrong-peer-group failure this module exists to prevent,
    # arrived at politely. Whether a global mandate may join a domestic cohort is a
    # human's call, so it goes in the human's pile.
    (r"international|global|asian|japan|taiwan|\bus\b|china|europe",
     "non-domestic mandate — a global sleeve does not belong in a domestic sector "
     "cohort even when it names one"),
    (r"rural", "rural theme — spans several sectors"),
    # THE 24-FUND BUG. A bare `services` also matches "Banking and Financial Services",
    # so every one of the 24 financials funds — the single largest clean group in the
    # worklist — was blanked as a "broad services theme". The lookbehinds exclude the
    # financial-services compound while leaving a standalone services mandate (Kotak
    # Services, Sundaram Services, Axis Services Opportunities) correctly unresolved.
    # Two separate fixed-width lookbehinds because Python's re rejects a variable-width
    # one; 'financials services' is a real AMFI spelling, not a typo guard.
    (r"(?<!financial )(?<!financials )\bservices\b", "services theme — broad, spans finance/IT/telecom"),
    (r"\bipo\b|recently listed", "listing-age theme, not a sector"),
)


@dataclass(frozen=True)
class Proposal:
    amfi_code: str
    scheme_name: str
    sector: Optional[str]          # None = deliberately left for the human
    matched_on: Optional[str]      # the token that fired, so the call is auditable
    reason: Optional[str]          # why it was left blank, when sector is None


def normalize_scheme_name(scheme_name: str) -> str:
    """Drop the plan/option tail ('- Direct Plan - Growth', '- Growth Option', ...).

    Only ever cuts AFTER the word 'fund'. That guard is load-bearing: 'Nippon India
    Growth Mid Cap Fund' is a real manifest scheme whose NAME contains 'Growth', and a
    naive cut at the first option keyword would truncate it to 'Nippon India'. Matching
    on the tail-free name is what lets a rule anchor on the end of a name at all."""
    import re as _re
    s = " ".join(str(scheme_name).split())
    m = _re.search(r"\bfund\b", s, _re.IGNORECASE)
    if not m:
        return s
    head, tail = s[: m.end()], s[m.end():]
    cut = _re.search(r"(?i)\b(direct|regular|growth|idcw|dividend|cumulative|payout|reinvest)",
                     tail)
    if cut:
        tail = tail[: cut.start()]
    return (head + tail).strip(" -–—,")


def strip_amc_prefix(scheme_name: str, amc_prefixes: Set[str]) -> str:
    """Remove a leading AMC name so its words cannot be read as sector evidence.

    Longest prefix first: 'Bajaj Finserv' must win over 'Bajaj' so that
    'Bajaj Finserv Consumption Fund' does not keep a stray 'Finserv' to match on."""
    s = " ".join(str(scheme_name).split())
    low = s.lower()
    for pref in sorted(amc_prefixes, key=len, reverse=True):
        p = pref.lower().strip()
        if p and low.startswith(p):
            return s[len(pref):].lstrip(" -–—")
    return s


def propose_sector(scheme_name: str, amc_prefixes: Set[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(sector, matched_on, reason). Pure — no I/O, so the selftest can pin the traps."""
    import re as _re
    body = strip_amc_prefix(normalize_scheme_name(scheme_name), amc_prefixes).lower()
    for pattern, why in _STRATEGY_MARKERS:          # strategy check FIRST: a strategy
        if _re.search(pattern, body):               # name that happens to contain a
            return None, None, why                  # sector word is still a strategy
    for pattern, sector in SECTOR_RULES:
        m = _re.search(pattern, body)
        if m:
            return sector, m.group(0), None
    return None, None, "no sector keyword in the name"


def _amc_prefixes(manifest: "pd.DataFrame") -> Set[str]:
    """AMC vocabulary from the manifest's own `amc` column, minus the ' Mutual Fund'
    suffix that never appears in a scheme name. Derived, never hardcoded, so it widens
    with the manifest exactly like trained_categories() does."""
    out: Set[str] = set()
    for a in manifest.get("amc", pd.Series(dtype=str)).dropna().astype(str):
        base = a.strip()
        for suffix in (" Mutual Fund", " Asset Management", " AMC"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if base:
            out.add(base.strip())
    # Scheme names use trading names the `amc` column spells differently.
    out.update({"Kotak", "Franklin India", "Franklin", "LIC MF", "Jio BlackRock",
                "JioBlackRock", "Invesco India", "Mirae Asset", "Baroda BNP Paribas",
                "Canara Robeco", "Bank of India", "The Wealth Company", "360 ONE",
                "Aditya Birla Sun Life", "Motilal Oswal", "Nippon India", "quant",
                "WhiteOak Capital", "Mahindra Manulife", "Bajaj Finserv", "PGIM India"})
    return out


def _propose(worklist: Path, out_path: Optional[Path]) -> None:
    """Suggest a sector for every blocked fund, with the evidence, for human review.

    Reads the git-tracked worklist rather than re-deriving from mf_cache/amfi_master
    (as --gaps does) so it runs on a cold checkout with no fetched data."""
    import mf_labels
    manifest = mf_labels.load_manifest()
    known = {str(s).strip() for s in manifest["sector"].dropna() if str(s).strip()}
    amcs = _amc_prefixes(manifest)

    if not worklist.exists():
        raise SystemExit(f"{worklist} not found — run --gaps --out {worklist} first")
    with worklist.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    proposals: List[Proposal] = []
    for r in rows:
        name = _clean(r.get("note")) or ""
        sector, matched, reason = propose_sector(name, amcs)
        # A regex must never mint a NEW cohort. If a rule ever drifts away from the
        # manifest's vocabulary, drop the suggestion rather than quietly create a
        # sector nobody decided on.
        if sector and sector not in known:
            sector, matched, reason = None, None, f"proposed '{sector}' is not an existing sector"
        proposals.append(Proposal(str(r.get("amfi_code", "")).strip(), name, sector, matched, reason))

    filled = [p for p in proposals if p.sector]
    blank = [p for p in proposals if not p.sector]
    by_sector: Dict[str, int] = {}
    for p in filled:
        by_sector[p.sector] = by_sector.get(p.sector, 0) + 1
    by_reason: Dict[str, int] = {}
    for p in blank:
        by_reason[p.reason or "?"] = by_reason.get(p.reason or "?", 0) + 1

    print(f"worklist rows                : {len(proposals)}")
    print(f"  proposed from the name     : {len(filled)}")
    print(f"  left for a human           : {len(blank)}")
    print("\nproposed sectors (all already exist in the manifest):")
    for s, n in sorted(by_sector.items(), key=lambda kv: -kv[1]):
        cur = int((manifest["sector"] == s).sum())
        print(f"  {n:4d}  {s}   (cohort {cur} -> {cur + n})")
    print("\nleft blank, by reason:")
    for s, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {s}")

    if out_path is not None:
        if out_path.resolve() == OVERRIDES_PATH.resolve():
            raise SystemExit(f"refusing to write proposals over the curated overrides "
                             f"file ({OVERRIDES_PATH}) — these are SUGGESTIONS, not curation")
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["amfi_code", "category", "proposed_sector", "matched_on",
                        "needs_human", "reason", "note"])
            for p in proposals:
                w.writerow([p.amfi_code, "Sectoral/Thematic", p.sector or "",
                            p.matched_on or "", "" if p.sector else "YES",
                            p.reason or "", p.scheme_name])
        print(f"\nwrote {len(proposals)} proposal(s) to {out_path}")
        print("REVIEW REQUIRED. These are suggestions read off the scheme name, not curation.")
        print(f"Accepted rows go to {OVERRIDES_PATH} with a real source_url, then --validate.")


def _gaps(out_path: Optional[Path] = None) -> None:
    """Worklist: which real funds are blocked, and what filling them in would unlock.

    With --out, writes EVERY blocked fund as a ready-to-edit CSV skeleton instead of
    the 10-row console preview. The preview is fine for "how bad is it"; it is useless
    as a work surface, and this is the single biggest hand-curation task in the project
    (204 funds), so it needs to leave the terminal."""
    import mf_labels, mf_universe as U
    manifest = mf_labels.load_manifest()
    trained = U.trained_categories(manifest)
    master = pd.read_parquet("mf_cache/amfi_master.parquet")
    have, issues = load(manifest=manifest)
    for i in issues:
        print(f"  {i}")

    in_manifest = set(manifest["amfi_code"].astype(str))
    rows = []
    for code, name, raw in zip(master["amfi_code"], master["scheme_name"], master["category_raw"]):
        v = U.classify(raw, trained)
        if not v.scoreable or str(code) in in_manifest:
            continue
        plan, option = U.plan_option(name)
        if (plan, option) != ("DIRECT", "GROWTH"):
            continue
        rows.append((str(code), name, v.category))

    blocked = [r for r in rows if r[2] in ("Sectoral/Thematic",)]
    ready = [r for r in rows if r[2] not in ("Sectoral/Thematic",)]
    resolved = [r for r in blocked if r[0] in have and have[r[0]].sector]

    print(f"\ncanonical scoreable funds OUTSIDE the trained manifest: {len(rows)}")
    print(f"  scoreable now (category-keyed cohort)      : {len(ready)}")
    print(f"  BLOCKED on sector (Sectoral/Thematic)      : {len(blocked)}")
    print(f"  of those, resolved by overrides            : {len(resolved)}")
    print(f"  still needing a curated sector             : {len(blocked) - len(resolved)}")
    print(f"\noverrides file: {OVERRIDES_PATH} ({len(have)} valid row(s))")
    todo = [b for b in blocked if b[0] not in have]
    if out_path is not None:
        # Never write over the curated file itself — that is hand-typed, unbackfillable
        # work and a skeleton dump would silently erase it (same reasoning that keeps
        # this whole directory out of mf_cache/).
        if out_path.resolve() == OVERRIDES_PATH.resolve():
            raise SystemExit(f"refusing to overwrite the curated overrides file "
                             f"({OVERRIDES_PATH}) with a blank skeleton — pick another --out path")
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["amfi_code", "category", "sector", "source_url", "note"])
            for code, name, _ in todo:
                # sector/source_url deliberately BLANK, not guessed: a wrong sector
                # silently builds the wrong peer group, and the cohort label is defined
                # against exactly that peer group.
                w.writerow([code, "Sectoral/Thematic", "", "", name])
        print(f"\nwrote {len(todo)} row(s) to {out_path}")
        print("fill the `sector` column (reuse an existing sector where one fits — a new "
              "one-member sector is below COHORT_MIN_SIZE and fails later as THIN_COHORT),")
        print(f"then append the filled rows to {OVERRIDES_PATH} and run --validate")
        known = sorted({s for s in manifest["sector"].dropna().astype(str) if s.strip()})
        print(f"\nsectors already in use ({len(known)}): {', '.join(known)}")
    elif todo:
        print("\nnext rows to curate (amfi_code,category,sector,source_url,note):")
        for code, name, _ in todo[:10]:
            print(f"  {code},Sectoral/Thematic,<sector>,<source_url>,{name[:52]}")
        print(f"  ... and {max(0, len(todo) - 10)} more — use --gaps --out <file.csv> for all of them")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gaps", action="store_true", help="show what is blocked and what to curate")
    ap.add_argument("--out", type=Path, default=None,
                    help="with --gaps: write EVERY blocked fund to this CSV as a "
                         "ready-to-edit skeleton, instead of the 10-row console preview")
    ap.add_argument("--validate", action="store_true", help="validate the real overrides file")
    ap.add_argument("--propose", action="store_true",
                    help="SUGGEST a sector for each blocked fund by reading its scheme "
                         "name, with the matched token as evidence. Writes suggestions "
                         "to --out for human review; never touches the curated file.")
    ap.add_argument("--worklist", type=Path, default=Path("overrides/_sector_worklist.csv"),
                    help="with --propose: the blocked-fund worklist to read")
    a = ap.parse_args()
    if a.propose:
        _propose(a.worklist, a.out)
    elif a.gaps:
        _gaps(a.out)
    elif a.validate:
        import mf_labels
        _have, _issues = load(manifest=mf_labels.load_manifest())
        for i in _issues:
            print(f"  {i}")
        print(f"{len(_have)} valid override(s); "
              f"{sum(1 for i in _issues if i.severity == SEVERITY_ERROR)} error(s), "
              f"{sum(1 for i in _issues if i.severity == SEVERITY_WARN)} warning(s)")
    else:
        _selftest()
