"""
====================================================================================
mf_holdings.py — PORTFOLIO-STRUCTURE FACTS + CONCENTRATION SCREEN
====================================================================================
WHY THIS EXISTS
---------------
The rest of the pipeline reads a fund's NAV series — a price chart. It cannot see
what the fund actually OWNS. When a monthly portfolio disclosure IS on file
(mf_cache/disclosures/<code>_<YYYY-MM>.csv, loaded by RealNAVStore into the
`current_holdings` frame), it carries facts NAV never can: how concentrated the
book is, whether a single issuer breaches SEBI's 10% ceiling, how diversified the
equity sleeve really is. This module turns that disclosure into DETERMINISTIC,
VERIFIABLE facts + ONE transparent, coverage-gated concentration metric.

THE HONESTY INVARIANT, AS IT APPLIES HERE (read CLAUDE.md before changing)
-------------------------------------------------------------------------
- **Never an ML feature.** Monthly holdings cannot be reconstructed back to the
  2014-2023 training anchors, so they can never enter the out-of-sample-validated
  `cohort_q1` predictor without manufacturing an unbacktestable, in-sample number.
  Holdings drive the recommendation ONLY as (a) deterministic compliance RULES and
  (b) a visible, explainable screen metric — never a laundered accuracy claim.
- **No fabricated active share.** Free data has no index constituent WEIGHTS, so a
  real active-share figure is not computable. We refuse to emit one rather than
  invent it.
- **Absence of evidence is a coverage flag, never a pass.** With no disclosure on
  file, `HoldingsFacts.available` is False, every derived number is None/NaN, and
  the downstream concentration metric greys out. It can never colour green from
  missing data.

WHAT THE NUMBERS MEAN
---------------------
- `top10_weight` — fraction of NAV in the ten largest equity positions. The standard
  diversification read: a broadly diversified active book sits well under half its
  NAV in its top ten; a concentrated single-thesis book runs much higher (higher
  single-name risk — SURFACED, not auto-condemned; a high-conviction fund is a
  legitimate style, so this colours the metric, it does not by itself force a SELL).
- `effective_n` — 1 / HHI over the (renormalised) equity weights: the "effective
  number of stocks" the book really behaves like. 40 equal names → 40; one 90%
  position dressed up with scraps → ~1.2.
- `single_issuer_breaches` — equity issuers above SEBI's 10%-of-NAV ceiling. This is
  a real regulatory line (index/ETF schemes excepted), so a breach on an active
  mandate is routed through the compliance layer as a hard finding.
====================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Tunable thresholds (documented constants, revisit per category if needed) -------
# SEBI (MF) single-issuer ceiling: a scheme may hold at most 10% of NAV in the equity
# of one issuer; index/ETF schemes replicating an index are exempt. A month-end
# snapshot can breach transiently — we surface it as a fact and let the compliance
# layer set severity by mandate.
SEBI_SINGLE_ISSUER_CAP = 0.10

# Concentration colouring for the transparent screen metric (top-10 weight, LOWER is
# more diversified). Green at/below DIVERSIFIED, red at/above CONCENTRATED, amber
# between. These are screen bands, NOT compliance limits.
TOP10_DIVERSIFIED = 0.45
TOP10_CONCENTRATED = 0.65

_EQUITY = "equity"


@dataclass
class HoldingsFacts:
    """Deterministic portfolio-structure facts from one disclosure snapshot. Every
    numeric field is None when `available` is False — never a fabricated zero."""

    available: bool
    n_equity: int = 0
    equity_weight: Optional[float] = None
    cash_weight: Optional[float] = None
    debt_weight: Optional[float] = None
    top1_name: Optional[str] = None
    top1_weight: Optional[float] = None
    top5_weight: Optional[float] = None
    top10_weight: Optional[float] = None
    effective_n: Optional[float] = None
    top_sector: Optional[str] = None
    top_sector_weight: Optional[float] = None
    # equity issuers over SEBI_SINGLE_ISSUER_CAP: (name, weight), largest first
    single_issuer_breaches: List[Tuple[str, float]] = field(default_factory=list)
    # transparent 0..1 diversification sub-score (higher = more diversified), the
    # monotone screen transform of top10_weight; None when unavailable
    concentration_score: Optional[float] = None
    note: str = ""


def _sum_weight(holdings: pd.DataFrame, asset_type: str) -> float:
    m = holdings["asset_type"] == asset_type
    return float(holdings.loc[m, "weight"].sum()) if m.any() else 0.0


def analyze(holdings: Optional[pd.DataFrame], available: bool = True) -> HoldingsFacts:
    """Reduce a `current_holdings`-shaped frame (columns: name, weight, asset_type,
    sector, cap_band) to structure facts. `available=False` (no disclosure on file)
    short-circuits to an all-None coverage result — the ONLY honest output on no
    evidence."""
    if not available or holdings is None or holdings.empty:
        return HoldingsFacts(
            available=False,
            note="NOT AVAILABLE — no portfolio holdings disclosure on file "
            "(mf_cache/disclosures/<code>_<YYYY-MM>.csv). Concentration screen "
            "and single-issuer check NOT EVALUATED (absence of evidence, not a pass).",
        )

    eq = holdings[holdings["asset_type"] == _EQUITY].copy()
    eq = eq[pd.to_numeric(eq["weight"], errors="coerce").notna()]
    if eq.empty:
        return HoldingsFacts(
            available=True,
            n_equity=0,
            equity_weight=0.0,
            cash_weight=_sum_weight(holdings, "cash"),
            debt_weight=_sum_weight(holdings, "debt"),
            note="Disclosure on file but no equity sleeve — concentration screen N/A for a non-equity book.",
        )

    w = eq["weight"].astype(float)
    ordered = w.sort_values(ascending=False)
    equity_weight = float(w.sum())

    # effective_n over the renormalised equity book (independent of the cash drag)
    p = w / equity_weight
    hhi = float((p**2).sum())
    effective_n = float(1.0 / hhi) if hhi > 0 else None

    top10_weight = float(ordered.head(10).sum())
    concentration_score = float(
        np.clip((TOP10_CONCENTRATED - top10_weight) / (TOP10_CONCENTRATED - TOP10_DIVERSIFIED), 0.0, 1.0)
    )

    # sector concentration (sector may be absent in a sparse CSV — degrade to None)
    top_sector = top_sector_weight = None
    if "sector" in eq.columns and eq["sector"].notna().any():
        sec = eq.groupby(eq["sector"])["weight"].sum().sort_values(ascending=False)
        if len(sec):
            top_sector = str(sec.index[0])
            top_sector_weight = float(sec.iloc[0])

    breaches = [(str(n), float(v)) for n, v in ordered.items() if v > SEBI_SINGLE_ISSUER_CAP + 1e-9]

    return HoldingsFacts(
        available=True,
        n_equity=int(len(eq)),
        equity_weight=equity_weight,
        cash_weight=_sum_weight(holdings, "cash"),
        debt_weight=_sum_weight(holdings, "debt"),
        top1_name=str(ordered.index[0]),
        top1_weight=float(ordered.iloc[0]),
        top5_weight=float(ordered.head(5).sum()),
        top10_weight=top10_weight,
        effective_n=effective_n,
        top_sector=top_sector,
        top_sector_weight=top_sector_weight,
        single_issuer_breaches=breaches,
        concentration_score=concentration_score,
        note="",
    )


def single_issuer_findings(facts: HoldingsFacts) -> List[Tuple[str, bool, str]]:
    """Compliance-ready summary of the SEBI 10% single-issuer check as
    (rule_id, passed, detail) tuples. Severity is the CALLER's call (it depends on
    the mandate — index/ETF schemes are exempt), keeping this module free of any
    orchestrator import. Empty when holdings are unavailable (the compliance layer
    already emits its own NOT-EVALUATED coverage finding in that case)."""
    if not facts.available or facts.n_equity == 0:
        return []
    if not facts.single_issuer_breaches:
        return [
            (
                "single_issuer<=10%",
                True,
                f"max single equity issuer {facts.top1_weight:.1%} ({facts.top1_name}) within SEBI 10% ceiling",
            )
        ]
    listed = ", ".join(f"{n} {v:.1%}" for n, v in facts.single_issuer_breaches[:5])
    return [
        (
            "single_issuer<=10%",
            False,
            f"{len(facts.single_issuer_breaches)} equity issuer(s) over SEBI's 10% ceiling: {listed}",
        )
    ]


# ==============================================================================
# SELFTEST — pure, data-free, deterministic (no NAV/cache/sklearn needed)
# ==============================================================================
def _frame(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.index = df["name"].astype(str).str.upper()
    return df


def _selftest() -> None:
    # 1) DIVERSIFIED book: 20 equal-ish equity names + cash -> green screen, no breach
    div = _frame(
        [
            dict(
                name=f"S{i:02d}",
                weight=0.048,
                asset_type="equity",
                sector=["Financials", "IT", "Pharma", "Auto"][i % 4],
                cap_band="large",
            )
            for i in range(20)
        ]
        + [dict(name="CASH", weight=0.04, asset_type="cash", sector="Cash", cap_band="cash")]
    )
    f = analyze(div)
    assert f.available and f.n_equity == 20
    assert abs(f.top10_weight - 0.48) < 1e-6, f.top10_weight
    assert not f.single_issuer_breaches
    assert 18 < f.effective_n <= 20 + 1e-6, f.effective_n  # ~equal-weight -> ~N
    # top10 0.48 sits just inside the amber band (0.45..0.65)
    assert 0 < f.concentration_score < 1, f.concentration_score
    assert single_issuer_findings(f)[0][1] is True

    # 2) CONCENTRATED book: one 22% position -> red screen + SEBI breach
    conc = _frame(
        [dict(name="MEGA", weight=0.22, asset_type="equity", sector="IT", cap_band="large")]
        + [dict(name=f"T{i:02d}", weight=0.06, asset_type="equity", sector="IT", cap_band="large") for i in range(12)]
        + [dict(name="CASH", weight=0.06, asset_type="cash", sector="Cash", cap_band="cash")]
    )
    f = analyze(conc)
    assert f.top1_name == "MEGA" and abs(f.top1_weight - 0.22) < 1e-9
    assert f.single_issuer_breaches and f.single_issuer_breaches[0][0] == "MEGA"
    assert f.concentration_score == 0.0, f.concentration_score  # top10 well over 0.65
    assert f.top_sector == "IT"
    rid, passed, detail = single_issuer_findings(f)[0]
    assert rid == "single_issuer<=10%" and passed is False and "MEGA" in detail

    # 3) UNAVAILABLE: no disclosure -> coverage result, every number None (never a 0 pass)
    for empty in (None, pd.DataFrame(columns=["name", "weight", "asset_type", "sector", "cap_band"])):
        f = analyze(empty, available=False)
        assert f.available is False
        assert f.top10_weight is None and f.concentration_score is None and f.effective_n is None
        assert single_issuer_findings(f) == []
        assert "NOT AVAILABLE" in f.note

    # 4) available=True but genuinely empty frame -> still no fabricated numbers
    f = analyze(pd.DataFrame(columns=["name", "weight", "asset_type", "sector", "cap_band"]))
    assert f.available is False and f.top10_weight is None

    # 5) debt-only book: equity sleeve empty -> N/A screen, not a fake green
    debt = _frame(
        [
            dict(name="GSEC", weight=0.9, asset_type="debt", sector="Sovereign", cap_band=None),
            dict(name="CASH", weight=0.1, asset_type="cash", sector="Cash", cap_band="cash"),
        ]
    )
    f = analyze(debt)
    assert f.available and f.n_equity == 0 and f.concentration_score is None
    assert abs(f.debt_weight - 0.9) < 1e-9

    print("mf_holdings selftest: OK (diversified/concentrated/unavailable/empty/debt-only all honest)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Portfolio-structure facts + concentration screen")
    ap.add_argument("--selftest", action="store_true", help="run the data-free selftest")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
