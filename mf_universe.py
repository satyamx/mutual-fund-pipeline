"""
====================================================================================
mf_universe.py — AMFI NAVAll -> CANONICAL CATEGORY + TRAINED-UNIVERSE GATE
====================================================================================
WHY THIS EXISTS
---------------
`mf_cache/amfi_master.parquet` (AMFI's NAVAll.txt) carries ~14k rows covering the
whole Indian mutual-fund market. The trained `cohort_q1` model saw 136 funds across
10 EQUITY categories and nothing else. Serving a probability for the other ~14k rows
would claim the holdout AUC (~0.578) transfers to debt/index/FoF/ETF populations the
model never saw — the exact "fabricated accuracy" the honesty invariant forbids.

This module is the boundary. It answers two questions, and only these two:
  1. What canonical SEBI category is this AMFI row, if any? (`normalize_category`)
  2. Is that category inside the TRAINED universe? (`classify`, given the live
     manifest's own category set)

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
- **No scoring, no probability, no model.** This module only decides eligibility.
- **No hardcoded list of trained categories.** The trained set is read from the live
  manifest by the caller and passed in. A frozen literal here would silently rot the
  moment the model is retrained wider — refusing funds the model now covers, or worse
  admitting ones it doesn't. The manifest is the single source of truth.
- **No guessing.** A row whose `category_raw` doesn't parse is UNMAPPED (a coverage
  status), never bucketed into a "closest" category. Absence of evidence is a
  coverage flag, never a pass.

THE AMFI STRING IS DIRTIER THAN IT LOOKS (all verified against the real cached file)
-----------------------------------------------------------------------------------
- AMFI writes BOTH "Equity Scheme - X" and "Equity Schemes - X" (singular and plural)
  for the same category. Both spellings occur in the live data.
- One trained cohort has three spellings: "Sectoral/ Thematic", "Thematic Fund",
  "Sectoral Fund" (note the space after the slash in the first).
- "ELSS" also appears as "ELSS- Tax Saver Fund".
- 2,544 rows carry an AMC NAME in the category column ("IL&FS Mutual Fund (IDF)")
  rather than any category at all. These are UNMAPPED, never inferred from the AMC.
- "Dividend Yield Fund" is a real SEBI EQUITY category that is NOT in the trained
  universe. It parses fine and is then correctly REFUSED by the trained-set check —
  the two steps are separate on purpose, so "we understood it" and "we may score it"
  never get conflated.
====================================================================================
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd

# --- Statuses -----------------------------------------------------------------
# TRAINED           — canonical category is present in the trained manifest.
# OUT_OF_TRAINING_UNIVERSE — parsed cleanly, but the model never saw this category.
# UNMAPPED_CATEGORY — AMFI's category string does not parse (dirty/absent).
STATUS_TRAINED = "TRAINED"
STATUS_OUT_OF_UNIVERSE = "OUT_OF_TRAINING_UNIVERSE"
STATUS_UNMAPPED = "UNMAPPED_CATEGORY"

# AMFI's sub-category label -> the canonical category string the MANIFEST uses.
# Keys are lowercased and whitespace-collapsed before lookup. Values must match the
# manifest's own `category` spelling exactly, because PeerProxyResolver._group_key
# keys cohorts off that string — a near-miss here silently creates a phantom cohort.
_SUBCATEGORY_MAP: Dict[str, str] = {
    "elss": "ELSS",
    "elss- tax saver fund": "ELSS",
    "elss - tax saver fund": "ELSS",
    "flexi cap fund": "Flexi Cap",
    "focused fund": "Focused",
    "large & mid cap fund": "Large & Mid Cap",
    "large cap fund": "Large Cap",
    "mid cap fund": "Mid Cap",
    "multi cap fund": "Multi Cap",
    "small cap fund": "Small Cap",
    # one trained cohort, three AMFI spellings
    "sectoral/ thematic": "Sectoral/Thematic",
    "sectoral/thematic": "Sectoral/Thematic",
    "sectoral fund": "Sectoral/Thematic",
    "thematic fund": "Sectoral/Thematic",
    # the manifest pools Value and Contra into ONE category; AMFI splits them
    "value fund": "Value/Contra",
    "contra fund": "Value/Contra",
    # parses cleanly but is NOT trained — kept here so it resolves to a real category
    # and is refused by the trained-set check, rather than looking like a parse failure
    "dividend yield fund": "Dividend Yield",
}

# "Open Ended Schemes(Equity Scheme - Large Cap Fund)" ->
#   family="Open Ended", asset="Equity Scheme", sub="Large Cap Fund"
_CATEGORY_RE = re.compile(
    r"^(?P<family>.*?)\s*Schemes?\s*\(\s*(?P<asset>[^-)]+?)\s*-\s*(?P<sub>.+?)\s*\)\s*$",
    re.IGNORECASE)

# Asset-class prefix that marks an EQUITY row. AMFI uses both singular and plural.
_EQUITY_ASSET_RE = re.compile(r"^equity\s+schemes?$", re.IGNORECASE)

# AMFI's LEGACY pre-2018-recategorization form has no "Asset - Sub" split at all,
# just a single bare label: "Close Ended Schemes(Income)", "Open Ended Schemes(Gilt)",
# "Close Ended Schemes(ELSS)". ~2.4k live rows still use it. These are perfectly
# READABLE — they are simply pre-SEBI-categorization labels with no modern
# sub-category, so they parse and are then refused as out-of-universe. Treating them
# as unreadable would repeat the very conflation `parsed` exists to prevent.
# Note this cannot swallow "IL&FS Mutual Fund (IDF)": that string has no
# Scheme/Schemes token before the parenthesis, so it still fails to parse.
_LEGACY_RE = re.compile(
    r"^(?P<family>.*?)\s*Schemes?\s*\(\s*(?P<sub>[^-()]+?)\s*\)\s*$", re.IGNORECASE)


def _norm(s: str) -> str:
    """Lowercase + collapse internal whitespace, so 'Large  Cap Fund' == 'large cap fund'."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


@dataclass(frozen=True)
class CategoryParse:
    """Result of parsing ONE AMFI `category_raw` string.

    `parsed` separates "AMFI's string was unreadable" from "we read it perfectly and
    it simply isn't a trained equity category". Both are unscoreable, but they are
    NOT the same coverage statement — a liquid fund is fully understood, not unmapped,
    and reporting it as a data gap would misdescribe ~9k funds to the app.
    """
    category: Optional[str]      # canonical manifest category, or None
    asset_class: Optional[str]   # e.g. "Equity Scheme" / "Debt Scheme", as AMFI wrote it
    parsed: bool                 # AMFI's string was structurally readable
    is_equity: bool
    is_open_ended: bool
    reason: str                  # human-readable, always populated


def normalize_category(category_raw: Optional[str]) -> CategoryParse:
    """AMFI `category_raw` -> canonical manifest category (or None + a reason).

    Parsing succeeds ONLY for open-ended equity rows whose sub-category is a known
    SEBI label. Everything else returns category=None with an explicit reason — this
    function never falls back to a nearest match.
    """
    if category_raw is None or (isinstance(category_raw, float) and pd.isna(category_raw)):
        return CategoryParse(None, None, False, False, False, "category_raw is null")

    raw = str(category_raw).strip()
    if not raw:
        return CategoryParse(None, None, False, False, False, "category_raw is empty")

    m = _CATEGORY_RE.match(raw)
    if not m:
        legacy = _LEGACY_RE.match(raw)
        if legacy:
            # Readable, but a pre-2018 label ("Income"/"Growth"/"Gilt"/"ELSS") with no
            # modern SEBI sub-category. Mapping the old broad "Growth" onto one of the
            # 10 trained categories would be a guess, so it deliberately yields none.
            label = legacy.group("sub").strip()
            return CategoryParse(None, None, True, False, _norm(legacy.group("family")) == "open ended",
                                 f"legacy pre-2018 AMFI label {label!r} — no SEBI sub-category to map")
        # e.g. "IL&FS Mutual Fund (IDF)" — an AMC name where a category belongs.
        # This is the ONLY genuine parse failure: we cannot say what this fund is.
        return CategoryParse(None, None, False, False, False,
                             f"does not match AMFI's 'Family Schemes(Asset - Sub)' shape: {raw!r}")

    asset = m.group("asset").strip()
    sub = _norm(m.group("sub"))
    is_open = _norm(m.group("family")) == "open ended"
    is_equity = bool(_EQUITY_ASSET_RE.match(asset.strip()))

    if not is_equity:
        # Fully understood — a debt/index/FoF/ETF scheme. Not a data gap.
        return CategoryParse(None, asset, True, False, is_open,
                             f"asset class {asset!r} is not equity")
    if not is_open:
        # Close-ended equity has a fixed maturity and no ongoing subscription; the
        # trained universe is entirely open-ended, so it is out of scope by construction.
        return CategoryParse(None, asset, True, True, False,
                             "close-ended/interval scheme — trained universe is open-ended only")

    canonical = _SUBCATEGORY_MAP.get(sub)
    if canonical is None:
        return CategoryParse(None, asset, True, True, True,
                             f"equity sub-category {m.group('sub').strip()!r} has no canonical mapping")
    return CategoryParse(canonical, asset, True, True, True, "mapped")


@dataclass(frozen=True)
class UniverseVerdict:
    """Whether ONE fund may receive a cohort_q1 probability, and why."""
    status: str                  # STATUS_* above
    category: Optional[str]      # canonical category when parsed, else None
    reason: str

    @property
    def scoreable(self) -> bool:
        """True ONLY for TRAINED. The gate is a refusal, not a warning label."""
        return self.status == STATUS_TRAINED


def classify(category_raw: Optional[str], trained_categories: Iterable[str]) -> UniverseVerdict:
    """Parse `category_raw`, then check it against the TRAINED manifest's own categories.

    `trained_categories` must come from the live manifest (see `trained_categories()`),
    never a literal — so retraining the model wider automatically widens this gate,
    and a narrower retrain automatically tightens it.
    """
    trained: Set[str] = {str(c) for c in trained_categories}
    parse = normalize_category(category_raw)

    if not parse.parsed:
        # We genuinely cannot say what this fund is — a real coverage gap.
        return UniverseVerdict(STATUS_UNMAPPED, None, parse.reason)
    if parse.category is None:
        # Read perfectly; it is simply not a trained equity category (debt, index,
        # FoF, ETF, close-ended, or an equity sub-category outside the trained set).
        # Reporting this as UNMAPPED would misdescribe a fully-understood fund.
        return UniverseVerdict(STATUS_OUT_OF_UNIVERSE, None, parse.reason)
    if parse.category not in trained:
        return UniverseVerdict(
            STATUS_OUT_OF_UNIVERSE, parse.category,
            f"category {parse.category!r} is not in the trained universe "
            f"({len(trained)} categories) — the cohort_q1 model never saw it")
    return UniverseVerdict(STATUS_TRAINED, parse.category, "in trained universe")


def trained_categories(manifest: pd.DataFrame) -> Set[str]:
    """The set of categories the model was actually trained on, read from the manifest."""
    return {str(c) for c in manifest["category"].dropna().unique()}


# ==============================================================================
# PLAN / OPTION — NAVAll lists every plan+option variant of the same scheme
# ==============================================================================
# 14,202 NAVAll rows include 6,585 "Direct", 4,134 "Regular" and 7,114 IDCW/dividend
# variants. These are the SAME underlying portfolio with different fee/payout
# mechanics, so treating them as distinct funds would inflate every cohort and
# corrupt the within-cohort percentile the label is defined on.
_DIRECT_RE = re.compile(r"\bdirect\b", re.IGNORECASE)
_REGULAR_RE = re.compile(r"\bregular\b", re.IGNORECASE)
_IDCW_RE = re.compile(r"\bidcw\b|\bdividend\b|\bpayout\b|\breinvest", re.IGNORECASE)
_GROWTH_RE = re.compile(r"\bgrowth\b", re.IGNORECASE)


def plan_option(scheme_name: str) -> Tuple[str, str]:
    """(plan, option) for one NAVAll scheme name.

    plan   ∈ DIRECT | REGULAR | UNKNOWN
    option ∈ GROWTH | IDCW | UNKNOWN

    UNKNOWN is returned rather than a default: many older scheme names omit the
    plan/option entirely, and silently calling those "Regular Growth" would merge
    genuinely different series.
    """
    name = str(scheme_name)
    plan = ("DIRECT" if _DIRECT_RE.search(name)
            else "REGULAR" if _REGULAR_RE.search(name) else "UNKNOWN")
    # Check IDCW first: "IDCW Growth Option" style names exist, and the payout
    # mechanic is what distinguishes the series, so it wins the tie.
    option = ("IDCW" if _IDCW_RE.search(name)
              else "GROWTH" if _GROWTH_RE.search(name) else "UNKNOWN")
    return plan, option


def _selftest() -> None:
    # ---- 1. The two AMFI spellings of the same category must agree ------------
    for raw in ("Open Ended Schemes(Equity Scheme - Large Cap Fund)",
                "Open Ended Schemes(Equity Schemes - Large Cap Fund)"):
        p = normalize_category(raw)
        assert p.category == "Large Cap", (raw, p)
        assert p.is_equity and p.is_open_ended
    print("[selftest] AMFI 'Equity Scheme' and 'Equity Schemes' both parse — PASS")

    # ---- 2. Three spellings collapse to ONE trained cohort -------------------
    for raw in ("Open Ended Schemes(Equity Scheme - Sectoral/ Thematic)",
                "Open Ended Schemes(Equity Schemes - Thematic Fund)",
                "Open Ended Schemes(Equity Schemes - Sectoral Fund)"):
        assert normalize_category(raw).category == "Sectoral/Thematic", raw
    # ...and Value/Contra pools two AMFI sub-categories into the manifest's one.
    assert normalize_category("Open Ended Schemes(Equity Scheme - Value Fund)").category == "Value/Contra"
    assert normalize_category("Open Ended Schemes(Equity Scheme - Contra Fund)").category == "Value/Contra"
    assert normalize_category("Open Ended Schemes(Equity Schemes - ELSS- Tax Saver Fund)").category == "ELSS"
    print("[selftest] sectoral/thematic + value/contra + ELSS spelling variants collapse — PASS")

    # ---- 3. Non-equity and malformed rows never produce a category -----------
    trained_10 = {"ELSS", "Flexi Cap", "Focused", "Large & Mid Cap", "Large Cap",
                  "Mid Cap", "Multi Cap", "Sectoral/Thematic", "Small Cap", "Value/Contra"}
    # A liquid fund is READ PERFECTLY — it is out of universe, NOT a data gap.
    # Conflating the two would misreport ~9k fully-understood funds as unmapped.
    debt = classify("Open Ended Schemes(Debt Scheme - Liquid Fund)", trained_10)
    assert debt.status == STATUS_OUT_OF_UNIVERSE and debt.category is None and not debt.scoreable, debt
    assert "not equity" in debt.reason, debt
    idx = classify("Open Ended Schemes(Other Scheme - Index Funds)", trained_10)
    assert idx.status == STATUS_OUT_OF_UNIVERSE and not idx.scoreable, idx

    # Only a genuinely unreadable string is UNMAPPED. The AMC name sitting in the
    # category column (2,544 real rows) must NOT be inferred from.
    ilfs = classify("IL&FS Mutual Fund (IDF)", trained_10)
    assert ilfs.status == STATUS_UNMAPPED and ilfs.category is None, ilfs
    for bad in (None, "", "   ", float("nan")):
        assert classify(bad, trained_10).status == STATUS_UNMAPPED, bad

    # Legacy pre-2018 labels are READABLE (no ' - ' split) -> out-of-universe, not a gap.
    for legacy in ("Close Ended Schemes(Income)", "Open Ended Schemes(Gilt)",
                   "Close Ended Schemes(ELSS)", "Interval Fund Schemes(Income)"):
        v = classify(legacy, trained_10)
        assert v.status == STATUS_OUT_OF_UNIVERSE and "legacy" in v.reason, (legacy, v)
    # ...and the legacy pattern must NOT swallow the AMC-name row above.
    assert classify("IL&FS Mutual Fund (IDF)", trained_10).status == STATUS_UNMAPPED
    print("[selftest] unreadable -> UNMAPPED; understood-but-untrained (debt/index) -> "
          "OUT_OF_TRAINING_UNIVERSE, kept distinct — PASS")

    # ---- 4. THE LOAD-BEARING CASE ------------------------------------------
    # Dividend Yield is a REAL equity category that PARSES cleanly but was never
    # trained. It must be refused — proving "we understood it" and "we may score
    # it" are genuinely separate decisions, not the same one.
    p = normalize_category("Open Ended Schemes(Equity Scheme - Dividend Yield Fund)")
    assert p.category == "Dividend Yield" and p.is_equity, p
    v = classify("Open Ended Schemes(Equity Scheme - Dividend Yield Fund)", trained_10)
    assert v.status == STATUS_OUT_OF_UNIVERSE and not v.scoreable, v
    assert v.category == "Dividend Yield", "the parsed category is still reported, just not scoreable"
    print("[selftest] Dividend Yield parses cleanly yet is REFUSED (untrained) — PASS")

    # ---- 5. Close-ended equity is out of scope by construction ---------------
    ce = normalize_category("Close Ended Schemes(Equity Scheme - ELSS)")
    assert ce.category is None and ce.parsed and ce.is_equity and not ce.is_open_ended, ce
    assert classify("Close Ended Schemes(Equity Scheme - ELSS)", trained_10).status == STATUS_OUT_OF_UNIVERSE
    print("[selftest] close-ended equity refused (trained universe is open-ended only) — PASS")

    # ---- 6. The gate follows the MANIFEST, not a frozen literal --------------
    # Retraining wider must widen the gate with no code edit here.
    wider = trained_10 | {"Dividend Yield"}
    assert classify("Open Ended Schemes(Equity Scheme - Dividend Yield Fund)", wider).scoreable, \
        "FAIL: gate must widen when the trained manifest widens"
    narrower = trained_10 - {"Small Cap"}
    assert not classify("Open Ended Schemes(Equity Scheme - Small Cap Fund)", narrower).scoreable, \
        "FAIL: gate must tighten when the trained manifest narrows"
    print("[selftest] gate tracks the manifest's category set in BOTH directions — PASS")

    # ---- 7. Plan/option split ------------------------------------------------
    assert plan_option("HDFC Large Cap Fund - Direct Plan - Growth Option") == ("DIRECT", "GROWTH")
    assert plan_option("HDFC Large Cap Fund - Regular Plan - IDCW") == ("REGULAR", "IDCW")
    assert plan_option("HDFC Large Cap Fund - Direct - Dividend Reinvestment") == ("DIRECT", "IDCW")
    # An older name with neither marker must stay UNKNOWN, not be defaulted.
    assert plan_option("Some Legacy Equity Fund") == ("UNKNOWN", "UNKNOWN")
    print("[selftest] plan/option parsed, unmarked legacy names stay UNKNOWN — PASS")

    print("[selftest] PASS — mf_universe parses AMFI honestly and refuses everything untrained")


def _report() -> None:
    """Run the gate over the REAL cached NAVAll + manifest and print the split."""
    import mf_labels
    df = pd.read_parquet("mf_cache/amfi_master.parquet")
    trained = trained_categories(mf_labels.load_manifest())
    print(f"trained categories ({len(trained)}): {sorted(trained)}\n")

    verdicts = [classify(c, trained) for c in df["category_raw"]]
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    total = len(verdicts)
    print(f"NAVAll rows: {total}")
    for status in (STATUS_TRAINED, STATUS_OUT_OF_UNIVERSE, STATUS_UNMAPPED):
        n = counts.get(status, 0)
        print(f"  {status:26s} {n:6d}  ({n / total:.1%})")

    scoreable = df[[v.scoreable for v in verdicts]].copy()
    scoreable["_cat"] = [v.category for v in verdicts if v.scoreable]
    plans = [plan_option(n) for n in scoreable["scheme_name"]]
    scoreable["_plan"] = [p for p, _ in plans]
    scoreable["_option"] = [o for _, o in plans]
    canon = scoreable[(scoreable["_plan"] == "DIRECT") & (scoreable["_option"] == "GROWTH")]
    print(f"\nscoreable rows (all plans/options): {len(scoreable)}")
    print(f"  of which Direct+Growth (canonical): {len(canon)}")
    print(f"  distinct ISINs among canonical    : {canon['isin'].nunique()}")
    print("\nby trained category (canonical Direct+Growth only):")
    print(canon["_cat"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="run the gate over the real cached NAVAll and print the split")
    a = ap.parse_args()
    if a.report:
        _report()
    else:
        _selftest()
