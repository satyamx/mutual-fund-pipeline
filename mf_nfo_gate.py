"""
====================================================================================
mf_nfo_gate.py — ELIGIBILITY GATE + NFO ASSESSMENT
====================================================================================
WHY THIS EXISTS
---------------
Every metric the pipeline produces is a function of NAV history and disclosed
holdings. An NFO has NEITHER. Feed one in and the engine will still emit numbers
— a Sharpe from 3 data points, a beta from noise, a "Manager Alpha Consistency
Score" computed over a portfolio that does not exist. Those numbers would be
syntactically valid and completely meaningless, and they would carry the same
authoritative formatting as a real result. That is the single most dangerous
failure mode in this system.

So the gate's job is to REFUSE. A pipeline that says "I cannot score this" is
strictly more useful than one that says "62.4/100" on no evidence.

WHAT AN NFO *CAN* HONESTLY BE ASSESSED ON
-----------------------------------------
The fund has no track record. But the PEOPLE do. So instead of scoring the fund,
we score what actually carries information at launch:
  1. Manager Track Record Proxy — run the Manager Alpha Consistency Score against
     the fund manager's OTHER, EXISTING funds. Skill is attached to the manager,
     not the scheme code.
  2. Structural terms — TER vs the category cap, exit load, lock-in.
  3. Mandate vs. category base rates — what does the median fund in this SEBI
     category actually deliver? An NFO's honest prior is its category, not its
     brochure.
  4. Launch-timing context — NFOs cluster near sector/market tops because that is
     when they sell. Worth flagging, not worth pretending to quantify.
This is explicitly a QUALITATIVE dossier. It never produces a 1-100 composite,
because a composite would imply a comparability with track-record funds that
does not exist.
====================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


class Eligibility(str, Enum):
    PRE_LAUNCH = "PRE_LAUNCH"      # NFO open / not yet allotted -> no NAV exists at all
    NEWBORN = "NEWBORN"            # allotted but < 6 months -> no meaningful statistics
    YOUNG = "YOUNG"                # 6m - 3y -> partial: short-window metrics only
    EVALUABLE = "EVALUABLE"        # >= 3y -> full pipeline


# Minimum history each analytic genuinely requires. These are not style choices:
# a 3-year rolling-return series needs 3 years of NAV, full stop.
REQUIREMENTS: Dict[str, int] = {
    "nav_basic_stats":      60,                          # ~3 months
    "beta_alpha":           60,
    "sharpe_sortino":      126,                          # ~6 months
    "max_drawdown":        126,
    "rolling_3y":          3 * TRADING_DAYS_PER_YEAR,
    "rolling_5y":          5 * TRADING_DAYS_PER_YEAR,
    "manager_alpha_score": 3 * TRADING_DAYS_PER_YEAR,    # needs t-3y disclosure + prices
    "sebi_true_to_label":  1,                            # needs ONE portfolio disclosure
}


@dataclass
class EligibilityVerdict:
    scheme_name: str
    status: Eligibility
    nav_observations: int
    age_days: Optional[int]
    allotment_date: Optional[date]
    available: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def can_score(self) -> bool:
        return self.status in (Eligibility.YOUNG, Eligibility.EVALUABLE)


class EligibilityGate:
    """Run BEFORE any agent touches the fund. Cheap, and it prevents nonsense."""

    NEWBORN_DAYS = 182
    YOUNG_DAYS = 3 * 365

    def assess(self, scheme_name: str, nav: Optional[pd.Series],
               allotment_date: Optional[date] = None,
               has_disclosure: bool = False,
               today: Optional[date] = None) -> EligibilityVerdict:
        today = today or date.today()
        n = 0 if nav is None else int(nav.dropna().shape[0])

        if allotment_date is None and n == 0:
            return EligibilityVerdict(
                scheme_name, Eligibility.PRE_LAUNCH, 0, None, None,
                blocked=list(REQUIREMENTS),
                reason="No NAV series and no allotment date: the scheme does not yet exist "
                       "as a priced instrument. Nothing in this pipeline is computable.")

        if allotment_date is not None and allotment_date > today:
            days = (allotment_date - today).days
            return EligibilityVerdict(
                scheme_name, Eligibility.PRE_LAUNCH, 0, None, allotment_date,
                blocked=list(REQUIREMENTS),
                reason=f"NFO not yet allotted (allotment in {days} days, on "
                       f"{allotment_date}). First NAV is the ₹10 face value, which "
                       f"carries zero information. REFUSING to score.")

        age = (today - allotment_date).days if allotment_date else None

        available, blocked = [], []
        for metric, need in REQUIREMENTS.items():
            if metric == "sebi_true_to_label":
                (available if has_disclosure else blocked).append(metric)
            elif n >= need:
                available.append(metric)
            else:
                blocked.append(metric)

        if age is not None and age < self.NEWBORN_DAYS:
            status = Eligibility.NEWBORN
            reason = (f"Allotted {allotment_date} — {age} days old ({n} NAV observations). "
                      f"Statistics on this sample are noise, not signal. Volatility, beta "
                      f"and Sharpe are not estimable; no portfolio has been disclosed yet "
                      f"(first disclosure lands ~10 days after the first month-end). "
                      f"REFUSING to score.")
        elif age is not None and age < self.YOUNG_DAYS:
            status = Eligibility.YOUNG
            reason = (f"{age} days old. Short-window metrics available; 3y/5y rolling "
                      f"returns and the Manager Alpha Consistency Score are NOT. "
                      f"Any composite is provisional.")
        else:
            status = Eligibility.EVALUABLE
            reason = f"{n} NAV observations — full pipeline available."

        return EligibilityVerdict(scheme_name, status, n, age, allotment_date,
                                  available, blocked, reason)


# ==============================================================================
# NFO ASSESSMENT — what you CAN legitimately say about a fund with no history
# ==============================================================================
@dataclass
class NFODossier:
    scheme_name: str
    amc: str
    managers: List[str]
    category: str
    benchmark: str
    verdict: EligibilityVerdict
    manager_prior_funds: Dict[str, Any] = field(default_factory=dict)
    structural: Dict[str, Any] = field(default_factory=dict)
    considerations: List[str] = field(default_factory=list)

    def render(self) -> str:
        L = []
        L.append("=" * 84)
        L.append(f"  {self.scheme_name}")
        L.append(f"  {self.amc} | {self.category} | benchmark: {self.benchmark}")
        L.append("=" * 84)
        L.append(f"  >>> NO RECOMMENDATION ISSUED — status: {self.verdict.status.value}")
        L.append(f"      {self.verdict.reason}")
        L.append("")
        L.append(f"  Blocked analytics ({len(self.verdict.blocked)}): "
                 f"{', '.join(self.verdict.blocked)}")
        if self.verdict.available:
            L.append(f"  Available: {', '.join(self.verdict.available)}")
        L.append("")
        L.append("  WHAT CAN BE ASSESSED INSTEAD")
        L.append("  " + "-" * 60)
        L.append(f"  Fund manager(s): {', '.join(self.managers)}")
        if self.manager_prior_funds:
            L.append("  Manager's EXISTING funds (skill attaches to the manager, not the")
            L.append("  scheme code — this is the only real evidence available at launch):")
            for f, note in self.manager_prior_funds.items():
                L.append(f"    - {f}: {note}")
        else:
            L.append("    (run the pipeline on this manager's other funds to build the proxy)")
        L.append("")
        for k, v in self.structural.items():
            L.append(f"  {k}: {v}")
        if self.considerations:
            L.append("")
            L.append("  CONSIDERATIONS")
            for c in self.considerations:
                L.append(f"    * {c}")
        return "\n".join(L)


class NFOAssessor:
    """
    Produces a qualitative dossier. Deliberately emits NO composite score — a
    number here would imply comparability with track-record funds that does not
    exist, and that false equivalence is precisely what we are guarding against.
    """

    def assess(self, scheme_name: str, amc: str, managers: List[str], category: str,
               benchmark: str, verdict: EligibilityVerdict,
               manager_prior_funds: Optional[Dict[str, Any]] = None,
               structural: Optional[Dict[str, Any]] = None,
               considerations: Optional[List[str]] = None) -> NFODossier:
        base = [
            "An NFO's honest prior is its SEBI category's base rate, not its brochure. "
            "Ask what the median fund in this category actually delivered.",
            "NFO units are offered at ₹10. This is a unit-accounting convention and "
            "carries no valuation information — a ₹10 NAV is not 'cheap'.",
            "There is no cost or tax penalty to waiting. A fund that is worth owning in "
            "year 4 will still be available in year 4, by which point it will have the "
            "track record this pipeline actually needs.",
        ]
        return NFODossier(scheme_name, amc, managers, category, benchmark, verdict,
                          manager_prior_funds or {}, structural or {},
                          (considerations or []) + base)
