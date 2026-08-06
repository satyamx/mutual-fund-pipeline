"""
====================================================================================
mf_sentinel.py — SYSTEM B "SENTINEL": TYPED ALERT ENGINE
====================================================================================
Sentinel emits typed, evidence-carrying DESCRIPTIVE alerts — never a fund number,
never a prediction. Each alert states a verified fact (a metric value crossing a
threshold, or a SEBI compliance line being crossed) with the measured evidence and
an explicit BASIS tag for how much confidence that fact deserves. This is how a
factor that measured at chance (the old equity FACTOR_MAP, within-cohort rank AUC
~0.46 — see mf_factor_backtest.py) survives honestly: as a fact a holder should
see, capped BELOW the severity of a verified regulatory breach, never disguised as
a prediction.

SEVERITY POLICY (the load-bearing rule — see mf-architecture-decisions memory):
  HIGH  — reserved for a VERIFIED breach of an external regulatory line (a SEBI
          cap or category mandate), computed from real data. Nothing else may be
          HIGH — enforced structurally below and asserted in --selftest.
  WATCH — a verified adverse fact crossing a defensible descriptive threshold
          (risk asymmetry, benchmark lag, cost drag, manager-history negative,
          keyword-level news).
  INFO  — context, mild deviations, coverage statements ("this could not be
          checked").

DATA-GAP POLICY (the honesty invariant): an alert NEVER fires on missing data.
Every rule below is naturally gated on np.isfinite(...) of its inputs; funds/rules
that cannot be evaluated on today's free data land in `checks_dormant`, not in a
fabricated pass or a spurious fire. Debt/passive-cost rules need holdings, TER or
AUM that RealNAVStore honestly reports as NOT AVAILABLE (mf_realstore.py) — they
are listed as dormant metadata, not executed against data that doesn't exist.
====================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from mf_pipeline import QuantEngine, ValidationOrchestrator
from mf_nfo_gate import EligibilityGate, NFOAssessor
import mf_holdings
import mf_managers

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # type-only: avoids a module-load cycle with mf_agent_orchestrator
    from mf_agent_orchestrator import FundDossier, ComplianceFinding


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WATCH = "WATCH"
    HIGH = "HIGH"


class AlertBasis(str, Enum):
    REGULATORY = "REGULATORY"              # verified breach of an external SEBI line
    CAUSAL_COST = "CAUSAL_COST"            # arithmetic cost drag — causal, not statistical
    DESCRIPTIVE_RISK = "DESCRIPTIVE_RISK"  # true historical NAV fact; no predictive claim
    GUARDRAIL_ONLY = "GUARDRAIL_ONLY"      # factor that only cleared the no-regression guardrail (AUC~0.46)
    MANAGER_HISTORY = "MANAGER_HISTORY"    # real disclosed-holdings audit (MACS) or tenure
    NEWS_KEYWORD = "NEWS_KEYWORD"          # word-boundary keyword match on headlines — weak evidence
    COVERAGE = "COVERAGE"                  # statement about what could NOT be checked


# Bases that can never carry HIGH — factors/history/news with no demonstrated skill
# or no verification are structurally barred from the tier that drives held-fund
# push notifications. Asserted in --selftest; do not bypass in a rule definition.
_NEVER_HIGH = {AlertBasis.GUARDRAIL_ONLY, AlertBasis.DESCRIPTIVE_RISK,
               AlertBasis.MANAGER_HISTORY, AlertBasis.NEWS_KEYWORD, AlertBasis.COVERAGE}


@dataclass
class Alert:
    code: str
    severity: str           # AlertSeverity value
    source: str             # "compliance" | "factor" | "holdings" | "manager" | "nfo" | "news"
    basis: str              # AlertBasis value
    evidence: Dict[str, Any]
    detail: str

    def __post_init__(self) -> None:
        if self.basis in {b.value for b in _NEVER_HIGH} and self.severity == AlertSeverity.HIGH.value:
            raise ValueError(f"{self.code}: basis {self.basis} may never carry HIGH severity")


@dataclass
class SentinelReport:
    alerts: List[Alert] = field(default_factory=list)
    nfo_dossier: Optional[Dict[str, Any]] = None
    checks_run: int = 0
    checks_dormant: List[str] = field(default_factory=list)
    # mf_nfo_gate.Eligibility value ("PRE_LAUNCH"|"NEWBORN"|"YOUNG"|"EVALUABLE") —
    # the batch artifact's per-fund `eligibility` field reads this directly
    # rather than re-deriving it from the alert codes.
    eligibility: str = "EVALUABLE"


# ==============================================================================
# THRESHOLDS — tunable defaults, surfaced for review (see mf-architecture-decisions
# memory for the reasoning behind each number).
# ==============================================================================
EQ_DOWNCAP_WATCH, EQ_DOWNCAP_INFO = 1.15, 1.05
EQ_CAPTURE_ASYMMETRY_GAP = 0.10
EQ_UPCAP_LOW = 0.80
EQ_BENCHMARK_LAG = -0.02
EQ_BETA_DRIFT = 0.35
EQ_DD_VS_BENCH = -0.10
EQ_DD_DEEP_WATCH, EQ_DD_DEEP_INFO = -0.40, -0.30
EQ_ROLL3Y_NEG_WATCH, EQ_ROLL3Y_NEG_INFO = 0.20, 0.10
EQ_EXPENSE_WATCH, EQ_EXPENSE_INFO = 0.020, 0.015

PASSIVE_TE_BREACH = 0.02        # SEBI's explicit 1y-rolling TE cap for equity ETF/index funds
PASSIVE_TE_WATCH, PASSIVE_TE_INFO = 0.01, 0.005
PASSIVE_TD_POOR = -0.015
PASSIVE_EXPENSE_INFO, PASSIVE_EXPENSE_WATCH, PASSIVE_EXPENSE_BREACH = 0.005, 0.0075, 0.010

HY_BENCHMARK_LAG = -0.02
HY_DOWNCAP_WATCH = 1.15
HY_DD_DEEP = -0.30
HY_EXPENSE_WATCH, HY_EXPENSE_INFO = 0.020, 0.015

MACS_WATCH_BELOW = 40
MACS_INFO_BELOW = 60

CAP_MANDATED_CATEGORIES = {"Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Large & Mid Cap"}

# Debt/passive-cost rule codes that need holdings/TER/AUM RealNAVStore does not
# provide on free data (mf_realstore.py:194-198). Real code exists in the spec;
# nothing to execute here until that data is hand-supplied, so they are honestly
# reported as dormant metadata rather than run against data that doesn't exist.
_STRUCTURALLY_DORMANT_DEBT = (
    "DEBT_ISSUER_LIMIT_BREACH", "DEBT_GROUP_LIMIT_BREACH", "DEBT_MANDATE_CREDIT_BREACH",
    "DEBT_CREDIT_QUALITY_LOW", "DEBT_DURATION_STRESS", "DEBT_YIELD_REACHING",
)
_STRUCTURALLY_DORMANT_HY_ARB = ("HY_ARB_SPREAD_LAG", "HY_ARB_SPREAD_COMPRESSING")


def _fmt(x: float, nd: int = 4) -> Optional[float]:
    return round(float(x), nd) if isinstance(x, (int, float)) and np.isfinite(x) else None


# ==============================================================================
# COMPLIANCE MAPPING (existing info/warning/critical -> Sentinel's INFO/WATCH/HIGH)
# ==============================================================================
_COVERAGE_ONLY_CODES = {"SEBI_CHECKS_NOT_EVALUATED", "SEBI_CATEGORY_UNKNOWN"}


def _compliance_code(rule_id: str) -> str:
    if rule_id.startswith("cap_fidelity_min_equity_and_overlap"):
        return "SEBI_CHECKS_NOT_EVALUATED"
    if rule_id.startswith("cap_fidelity_"):
        return "SEBI_CAP_FIDELITY_BREACH"
    if rule_id.startswith("sector_theme_fidelity") or rule_id.startswith("sectoral_debt"):
        return "SEBI_SECTOR_FIDELITY_BREACH"
    if rule_id.startswith("min_equity"):
        return "SEBI_MIN_EQUITY_BREACH"
    if rule_id.startswith("overlap_vs"):
        return "SEBI_OVERLAP_BREACH"
    if rule_id == "solution_oriented_discontinued":
        return "SEBI_CATEGORY_DISCONTINUED"
    if rule_id == "category_known":
        return "SEBI_CATEGORY_UNKNOWN"
    if rule_id == "scheme_name_matches_category":
        return "NAME_CATEGORY_MISMATCH"
    if rule_id == "no_return_focused_words_in_name":
        return "RETURN_FOCUSED_NAME"
    if rule_id.startswith("single_issuer"):
        return "SEBI_SINGLE_ISSUER_BREACH"
    return f"SEBI_OTHER[{rule_id}]"


def _compliance_alerts(compliance: List["ComplianceFinding"]) -> Tuple[List[Alert], int]:
    alerts: List[Alert] = []
    checks_run = 0
    for f in compliance:
        checks_run += 1
        if f.passed:
            continue
        code = _compliance_code(f.rule_id)
        if f.severity == "critical":
            sev, basis = AlertSeverity.HIGH, AlertBasis.REGULATORY
        elif code in _COVERAGE_ONLY_CODES:
            sev, basis = AlertSeverity.INFO, AlertBasis.COVERAGE
        else:
            sev, basis = AlertSeverity.WATCH, AlertBasis.REGULATORY
        alerts.append(Alert(code, sev.value, "compliance", basis.value,
                            evidence={"rule_id": f.rule_id}, detail=f.detail))
    return alerts, checks_run


# ==============================================================================
# FACTOR ALERTS — computed live from NAV + benchmark via QuantEngine
# ==============================================================================
def _fund_metrics(quant: QuantEngine, nav: pd.Series, benchmark: Optional[pd.Series]) -> Dict[str, float]:
    nav = nav.dropna()
    rets = quant.daily_returns(nav)
    out: Dict[str, float] = {
        "cagr": quant.cagr(nav),
        "sortino": quant.sortino(rets) if len(rets) >= 126 else float("nan"),
    }
    nav3 = nav.tail(3 * 252)
    out["max_dd_3y"] = quant.max_drawdown(nav3) if len(nav3) >= 60 else float("nan")
    rr3 = quant.rolling_returns(nav, 3)
    out["rr3y_negative_share"] = float(rr3.lt(0).mean()) if len(rr3) else float("nan")

    have_bench = benchmark is not None and len(benchmark.dropna()) > 0
    bench_keys = ("excess_cagr_3y", "beta", "tracking_error", "info_ratio",
                 "upside_capture", "downside_capture", "tracking_difference_tri",
                 "benchmark_max_dd_3y")
    if not have_bench:
        out.update({k: float("nan") for k in bench_keys})
        return out

    common = nav.index.intersection(benchmark.dropna().index)
    if len(common) < 60:
        out.update({k: float("nan") for k in bench_keys})
        return out
    fnav, bnav = nav.loc[common], benchmark.loc[common]
    frets, brets = quant.daily_returns(fnav), quant.daily_returns(bnav)
    out["excess_cagr_3y"] = (float(quant.cagr(fnav) - quant.cagr(bnav))
                             if len(common) > 252 else float("nan"))
    beta, _alpha = quant.beta_alpha(frets, brets)
    out["beta"] = beta
    out["tracking_error"] = quant.tracking_error(frets, brets)
    out["info_ratio"] = quant.information_ratio(frets, brets)
    out["upside_capture"], out["downside_capture"] = quant.capture_ratios(frets, brets)
    out["tracking_difference_tri"] = (quant.tracking_difference_tri(fnav, bnav)
                                      if len(common) >= 252 else float("nan"))
    bnav3 = bnav.tail(3 * 252)
    out["benchmark_max_dd_3y"] = quant.max_drawdown(bnav3) if len(bnav3) >= 60 else float("nan")
    return out


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and np.isfinite(x) for x in xs)


def _equity_alerts(m: Dict[str, float], category: str) -> List[Alert]:
    a: List[Alert] = []
    dc = m["downside_capture"]
    if _finite(dc, m["upside_capture"]):
        if dc - m["upside_capture"] >= EQ_CAPTURE_ASYMMETRY_GAP and dc > 1.0:
            a.append(Alert("EQ_CAPTURE_ASYMMETRY", AlertSeverity.WATCH.value, "factor",
                           AlertBasis.GUARDRAIL_ONLY.value,
                           {"downside_capture": _fmt(dc), "upside_capture": _fmt(m["upside_capture"])},
                           f"Keeps more of the downside ({dc:.0%} of benchmark losses) than the "
                           f"upside ({m['upside_capture']:.0%} of benchmark gains) over 3y — "
                           f"descriptive of past NAV behaviour; this factor showed no within-cohort "
                           f"selection skill (rank AUC ~0.46)."))
        elif dc > EQ_DOWNCAP_WATCH:
            a.append(Alert("EQ_DOWNSIDE_CAPTURE_HIGH", AlertSeverity.WATCH.value, "factor",
                           AlertBasis.GUARDRAIL_ONLY.value, {"downside_capture": _fmt(dc)},
                           f"Downside capture {dc:.0%} — loses more than the benchmark in down "
                           f"markets over 3y; descriptive only, no demonstrated selection skill "
                           f"(rank AUC ~0.46)."))
    if _finite(m["upside_capture"]) and m["upside_capture"] < EQ_UPCAP_LOW:
        a.append(Alert("EQ_UPSIDE_CAPTURE_LOW", AlertSeverity.INFO.value, "factor",
                       AlertBasis.GUARDRAIL_ONLY.value, {"upside_capture": _fmt(m["upside_capture"])},
                       f"Upside capture {m['upside_capture']:.0%} — captures less than the "
                       f"benchmark's up-moves; may be a deliberate defensive mandate."))
    if _finite(m["excess_cagr_3y"]) and m["excess_cagr_3y"] <= EQ_BENCHMARK_LAG:
        a.append(Alert("EQ_BENCHMARK_LAG", AlertSeverity.WATCH.value, "factor",
                       AlertBasis.DESCRIPTIVE_RISK.value,
                       {"excess_cagr_3y": _fmt(m["excess_cagr_3y"]), "info_ratio": _fmt(m["info_ratio"])},
                       f"Trails its real category benchmark by {-m['excess_cagr_3y']:.1%}/yr over 3y."))
    if _finite(m["beta"]) and abs(m["beta"] - 1.0) > EQ_BETA_DRIFT:
        sev = AlertSeverity.WATCH if category in CAP_MANDATED_CATEGORIES else AlertSeverity.INFO
        a.append(Alert("EQ_BETA_DRIFT", sev.value, "factor", AlertBasis.DESCRIPTIVE_RISK.value,
                       {"beta_3y": _fmt(m["beta"])},
                       f"Beta {m['beta']:.2f} far from category norm — possible category drift."))
    if _finite(m["max_dd_3y"], m["benchmark_max_dd_3y"]):
        gap = m["max_dd_3y"] - m["benchmark_max_dd_3y"]
        if gap <= EQ_DD_VS_BENCH:
            a.append(Alert("EQ_DRAWDOWN_VS_BENCH", AlertSeverity.WATCH.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value,
                           {"max_dd_3y": _fmt(m["max_dd_3y"]), "benchmark_max_dd_3y": _fmt(m["benchmark_max_dd_3y"])},
                           f"Drew down {-gap:.1%} deeper than its own benchmark over 3y."))
    if _finite(m["max_dd_3y"]):
        if m["max_dd_3y"] <= EQ_DD_DEEP_WATCH:
            a.append(Alert("EQ_DRAWDOWN_DEEP", AlertSeverity.WATCH.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value, {"max_dd_3y": _fmt(m["max_dd_3y"])},
                           f"Max drawdown {m['max_dd_3y']:.1%} over 3y — beyond the Nifty's COVID drawdown (~-38%)."))
        elif m["max_dd_3y"] <= EQ_DD_DEEP_INFO:
            a.append(Alert("EQ_DRAWDOWN_DEEP", AlertSeverity.INFO.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value, {"max_dd_3y": _fmt(m["max_dd_3y"])},
                           f"Max drawdown {m['max_dd_3y']:.1%} over 3y."))
    if _finite(m["rr3y_negative_share"]):
        if m["rr3y_negative_share"] > EQ_ROLL3Y_NEG_WATCH:
            a.append(Alert("EQ_ROLL3Y_NEGATIVE_SHARE", AlertSeverity.WATCH.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value, {"rr3y_negative_share": _fmt(m["rr3y_negative_share"])},
                           f"{m['rr3y_negative_share']:.0%} of all 3-year holding windows lost money."))
        elif m["rr3y_negative_share"] > EQ_ROLL3Y_NEG_INFO:
            a.append(Alert("EQ_ROLL3Y_NEGATIVE_SHARE", AlertSeverity.INFO.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value, {"rr3y_negative_share": _fmt(m["rr3y_negative_share"])},
                           f"{m['rr3y_negative_share']:.0%} of all 3-year holding windows lost money."))
    if _finite(m["sortino"]) and m["sortino"] < 0:
        a.append(Alert("EQ_SORTINO_NEGATIVE", AlertSeverity.INFO.value, "factor",
                       AlertBasis.DESCRIPTIVE_RISK.value, {"sortino": _fmt(m["sortino"])},
                       f"Sortino {m['sortino']:.2f} — below the risk-free MAR with downside vol."))
    return a


def _passive_alerts(m: Dict[str, float]) -> List[Alert]:
    a: List[Alert] = []
    te = m["tracking_error"]
    if _finite(te):
        if te > PASSIVE_TE_BREACH:
            a.append(Alert("PASSIVE_TRACKING_ERROR_BREACH", AlertSeverity.HIGH.value, "factor",
                           AlertBasis.REGULATORY.value, {"tracking_error": _fmt(te)},
                           f"1y-rolling tracking error {te:.2%} exceeds SEBI's 2% cap for equity "
                           f"ETFs/index funds."))
        elif te > PASSIVE_TE_WATCH:
            a.append(Alert("PASSIVE_TRACKING_ERROR_HIGH", AlertSeverity.WATCH.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value, {"tracking_error": _fmt(te)},
                           f"Tracking error {te:.2%} — sloppy replication vs well-run peers (typically <50bp)."))
        elif te > PASSIVE_TE_INFO:
            a.append(Alert("PASSIVE_TRACKING_ERROR_HIGH", AlertSeverity.INFO.value, "factor",
                           AlertBasis.DESCRIPTIVE_RISK.value, {"tracking_error": _fmt(te)},
                           f"Tracking error {te:.2%}."))
    td = m["tracking_difference_tri"]
    if _finite(td) and td <= PASSIVE_TD_POOR:
        a.append(Alert("PASSIVE_TRACKING_DIFF_POOR", AlertSeverity.WATCH.value, "factor",
                       AlertBasis.DESCRIPTIVE_RISK.value,
                       {"tracking_difference_tri": _fmt(td), "benchmark_basis": "adj_PRI(+1.3%)"},
                       f"Lags its TRI benchmark by {-td:.2%}/yr — beyond plausible TER drag alone. "
                       f"NOTE: live benchmark is a PRI+1.3%-dividend-addback approximation (±0.5pp)."))
    return a


def _hybrid_alerts(m: Dict[str, float]) -> List[Alert]:
    a: List[Alert] = []
    if _finite(m["excess_cagr_3y"]) and m["excess_cagr_3y"] <= HY_BENCHMARK_LAG:
        a.append(Alert("HY_BENCHMARK_LAG", AlertSeverity.WATCH.value, "factor",
                       AlertBasis.DESCRIPTIVE_RISK.value, {"excess_cagr_3y": _fmt(m["excess_cagr_3y"])},
                       f"Trails its category hybrid benchmark by {-m['excess_cagr_3y']:.1%}/yr over 3y."))
    if _finite(m["downside_capture"]) and m["downside_capture"] > HY_DOWNCAP_WATCH:
        a.append(Alert("HY_DOWNSIDE_CAPTURE_HIGH", AlertSeverity.WATCH.value, "factor",
                       AlertBasis.GUARDRAIL_ONLY.value, {"downside_capture": _fmt(m["downside_capture"])},
                       f"Downside capture {m['downside_capture']:.0%} of the hybrid benchmark — "
                       f"a hybrid mandate's reason to exist is cushioning drawdowns."))
    if _finite(m["max_dd_3y"]) and m["max_dd_3y"] <= HY_DD_DEEP:
        a.append(Alert("HY_DRAWDOWN_DEEP", AlertSeverity.WATCH.value, "factor",
                       AlertBasis.DESCRIPTIVE_RISK.value, {"max_dd_3y": _fmt(m["max_dd_3y"])},
                       f"Max drawdown {m['max_dd_3y']:.1%} over 3y — beyond a typical aggressive-hybrid edge."))
    return a


def _expense_alert(prefix: str, expense_ratio: float, watch: float, info: float,
                   breach: Optional[float] = None) -> Tuple[Optional[Alert], bool]:
    """Returns (alert_or_None, was_dormant). Dormant when TER isn't on file (always
    true today per mf_realstore.py — no free TER source)."""
    if not np.isfinite(expense_ratio):
        return None, True
    if breach is not None and expense_ratio > breach:
        return Alert(f"{prefix}_EXPENSE_ABOVE_REG_CAP", AlertSeverity.HIGH.value, "factor",
                     AlertBasis.REGULATORY.value, {"expense_ratio": _fmt(expense_ratio)},
                     f"TER {expense_ratio:.2%} exceeds the SEBI slab ceiling for this scheme type."), False
    if expense_ratio >= watch:
        return Alert(f"{prefix}_EXPENSE_HIGH", AlertSeverity.WATCH.value, "factor",
                     AlertBasis.CAUSAL_COST.value, {"expense_ratio": _fmt(expense_ratio)},
                     f"TER {expense_ratio:.2%} — a 1% gap compounds to ~7.5% over a 7.5y horizon."), False
    if expense_ratio >= info:
        return Alert(f"{prefix}_EXPENSE_HIGH", AlertSeverity.INFO.value, "factor",
                     AlertBasis.CAUSAL_COST.value, {"expense_ratio": _fmt(expense_ratio)},
                     f"TER {expense_ratio:.2%}."), False
    return None, False


def _factor_alerts(quant: QuantEngine, dossier: "FundDossier",
                   benchmark: Optional[pd.Series]) -> Tuple[List[Alert], List[str], int]:
    bucket = dossier.bucket.value
    alerts: List[Alert] = []
    dormant: List[str] = []
    checks_run = 0

    if bucket == "Equity":
        m = _fund_metrics(quant, dossier.nav, benchmark)
        checks_run += 1
        alerts += _equity_alerts(m, dossier.category)
        alert, was_dormant = _expense_alert("EQ", dossier.expense_ratio, EQ_EXPENSE_WATCH, EQ_EXPENSE_INFO)
        if alert:
            alerts.append(alert)
        if was_dormant:
            dormant.append("EQ_EXPENSE_HIGH")
    elif bucket == "Other (Index/ETF/FoF)":
        m = _fund_metrics(quant, dossier.nav, benchmark)
        checks_run += 1
        alerts += _passive_alerts(m)
        alert, was_dormant = _expense_alert("PASSIVE", dossier.expense_ratio,
                                            PASSIVE_EXPENSE_WATCH, PASSIVE_EXPENSE_INFO,
                                            breach=PASSIVE_EXPENSE_BREACH)
        if alert:
            alerts.append(alert)
        if was_dormant:
            dormant.append("PASSIVE_EXPENSE_HIGH")
    elif bucket in ("Hybrid", "Life Cycle"):
        m = _fund_metrics(quant, dossier.nav, benchmark)
        checks_run += 1
        alerts += _hybrid_alerts(m)
        alert, was_dormant = _expense_alert("HY", dossier.expense_ratio, HY_EXPENSE_WATCH, HY_EXPENSE_INFO)
        if alert:
            alerts.append(alert)
        if was_dormant:
            dormant.append("HY_EXPENSE_HIGH")
        if dossier.category == "Arbitrage":
            dormant.extend(_STRUCTURALLY_DORMANT_HY_ARB)
    elif bucket == "Debt":
        m = _fund_metrics(quant, dossier.nav, benchmark)
        checks_run += 1
        if _finite(m["rr3y_negative_share"]) and m["rr3y_negative_share"] > 0:
            alerts.append(Alert("DEBT_NEGATIVE_3Y_WINDOW", AlertSeverity.WATCH.value, "factor",
                                AlertBasis.DESCRIPTIVE_RISK.value,
                                {"rr3y_negative_share": _fmt(m["rr3y_negative_share"])},
                                "At least one 3-year holding window lost money — a credit-event "
                                "signature for a debt fund, not rate noise."))
        alert, was_dormant = _expense_alert("DEBT", dossier.expense_ratio, 0.010, 0.0075)
        if alert:
            alerts.append(alert)
        if was_dormant:
            dormant.append("DEBT_EXPENSE_HIGH")
        dormant.extend(_STRUCTURALLY_DORMANT_DEBT)
    elif bucket == "Solution Oriented (DISCONTINUED)":
        # No live factor rules defined for a discontinued mandate; the category
        # breach itself is already raised via compliance (SEBI_CATEGORY_DISCONTINUED).
        dormant.append("FACTOR_RULES_SKIPPED_DISCONTINUED_CATEGORY")

    return alerts, dormant, checks_run


# ==============================================================================
# HOLDINGS ALERTS — portfolio structure from a disclosure snapshot (mf_holdings)
# ==============================================================================
# System A already colours a `concentration` metric and routes a SEBI single-issuer
# breach through compliance. Sentinel's job here is the typed-alert view of the SAME
# facts, so it deliberately reuses mf_holdings' OWN screen bands rather than defining
# parallel thresholds — if these drifted apart the app could render a green
# concentration metric beside a concentration WATCH alert on one fund.
#
# Basis is DESCRIPTIVE_RISK, never REGULATORY: a concentrated book is a legitimate
# high-conviction style, not a breach. `_NEVER_HIGH` structurally caps it below HIGH,
# which is the intended honesty guard — only the genuine SEBI single-issuer breach
# (raised via the compliance channel as `SEBI_SINGLE_ISSUER_BREACH`) may reach HIGH.
_HOLDINGS_CONCENTRATION_CODE = "HOLDINGS_CONCENTRATION_HIGH"


def _holdings_alerts(dossier: "FundDossier") -> Tuple[List[Alert], List[str], int]:
    """Concentration alerts from the current disclosure. Returns (alerts, dormant,
    checks_run).

    Scope is the EQUITY/HYBRID buckets only, mirroring the single-issuer exemption in
    `TrueToLabelVerifier` (mf_agent_orchestrator.py): an index/ETF scheme's top-10
    weight is a property of the INDEX it replicates, not a manager decision, so
    alerting on it would flag the benchmark itself as a risk. Those buckets are
    mandate-inapplicable, not unevaluable, so they are not reported dormant either.
    """
    bucket = dossier.bucket.value
    if bucket not in ("Equity", "Hybrid"):
        return [], [], 0

    # Deliberately NOT gated on the NFO eligibility status: concentration needs a
    # disclosure, not NAV history, so it is often the ONLY evidence available on a
    # newborn fund whose NAV-based factor rules are all dormant.
    facts = mf_holdings.analyze(dossier.current_holdings,
                                available=dossier.holdings_available)
    if not facts.available or facts.n_equity == 0:
        # Absence of evidence is dormancy, never a quiet pass.
        return [], [_HOLDINGS_CONCENTRATION_CODE], 0

    top10 = facts.top10_weight
    if top10 is None or not np.isfinite(top10):
        return [], [_HOLDINGS_CONCENTRATION_CODE], 0

    evidence = {
        "top10_weight": _fmt(top10),
        "effective_n": _fmt(facts.effective_n),
        "n_equity": facts.n_equity,
        "top1": facts.top1_name,
        "top1_weight": _fmt(facts.top1_weight),
        "top_sector": facts.top_sector,
        "top_sector_weight": _fmt(facts.top_sector_weight),
        "band_diversified": mf_holdings.TOP10_DIVERSIFIED,
        "band_concentrated": mf_holdings.TOP10_CONCENTRATED,
    }
    if top10 >= mf_holdings.TOP10_CONCENTRATED:
        sev = AlertSeverity.WATCH
    elif top10 >= mf_holdings.TOP10_DIVERSIFIED:
        sev = AlertSeverity.INFO
    else:
        return [], [], 1

    eff = f"{facts.effective_n:.1f}" if facts.effective_n is not None else "n/a"
    return ([Alert(_HOLDINGS_CONCENTRATION_CODE, sev.value, "holdings",
                   AlertBasis.DESCRIPTIVE_RISK.value, evidence,
                   f"Top-10 equity weight {top10:.1%} across {facts.n_equity} names "
                   f"(effective N {eff}) — concentrated book, higher single-name risk. "
                   f"A style observation, not a breach.")], [], 1)


# ==============================================================================
# MANAGER ALERTS — per-scheme MACS (real) + tenure (needs managers.csv)
# ==============================================================================
def _manager_alerts(backtest: Dict[str, Any]) -> Tuple[List[Alert], List[str]]:
    macs = backtest.get("manager_alpha_consistency_score")
    if macs is None:
        return [], []
    if macs < MACS_WATCH_BELOW:
        return [Alert("MANAGER_PICKS_UNDERPERFORMED", AlertSeverity.WATCH.value, "manager",
                      AlertBasis.MANAGER_HISTORY.value, {"macs": macs},
                      f"Manager Alpha Consistency Score {macs}/100 — revealed top-10 picks "
                      f"underperformed passive sector clones.")], []
    if macs < MACS_INFO_BELOW:
        return [Alert("MANAGER_ALPHA_NOT_EVIDENCED", AlertSeverity.INFO.value, "manager",
                      AlertBasis.MANAGER_HISTORY.value, {"macs": macs},
                      f"Manager Alpha Consistency Score {macs}/100 — stock selection roughly "
                      f"matches a passive clone; alpha not evidenced.")], []
    return [], []  # Sentinel is a risk/coverage channel — strong MACS emits nothing (user decision)


def _load_managers_csv(path: str = "mf_cache/managers.csv") -> Optional[pd.DataFrame]:
    """Hand-sourced manager<->scheme mapping (no free API publishes fund-manager
    identity — mf_realstore.py:194). Absent by default; cross-fund/tenure rules stay
    dormant until the user populates it.

    Delegates to mf_managers, which validates the file rather than trusting it. The
    three lines this replaced swallowed EVERY exception into `return None`, so a
    malformed hand-edited CSV was indistinguishable from an absent one and presented
    as "dormant" forever. Validation issues are logged here; a broken file yields an
    EMPTY frame (rules produce nothing) rather than None (rules never ran), and the
    log line is the only thing that tells the two apart."""
    rows, issues = mf_managers.load(Path(path))
    for i in issues:
        (LOGGER.error if i.severity == mf_managers.SEVERITY_ERROR else LOGGER.warning)(
            "managers.csv %s", i)
    return rows


def _tenure_alert(managers_df: Optional[pd.DataFrame], dossier: "FundDossier") -> Optional[Alert]:
    # Resolved by latest start_date, NOT by `rows.iloc[-1]` as this once was — that
    # made the current manager depend on the order a human typed the file, so history
    # entered oldest-last silently reported a departed manager as the incumbent. Same
    # defect class as the resolver's `hit.iloc[0]` collision.
    row = mf_managers.current_manager(managers_df, dossier.scheme_name)
    if row is None or pd.isna(row.get("start_date")):
        return None
    tenure_days = (pd.Timestamp.today().normalize() - row["start_date"]).days
    if tenure_days < 365:
        return Alert("MANAGER_TENURE_SHORT", AlertSeverity.INFO.value, "manager",
                     AlertBasis.MANAGER_HISTORY.value,
                     {"tenure_days": int(tenure_days), "manager": row["manager_name"]},
                     f"Current manager ({row['manager_name']}) has run this scheme for "
                     f"{tenure_days} days (<1y) — this scheme's own track record under them is thin.")
    return None


# ==============================================================================
# NEWS ALERTS
# ==============================================================================
def _news_alerts(sentiment: Dict[str, Any]) -> List[Alert]:
    return [Alert("REGULATORY_NEWS_KEYWORD", AlertSeverity.WATCH.value, "news",
                  AlertBasis.NEWS_KEYWORD.value, {"headline": rf},
                  f"Regulatory keyword match on a headline (word-boundary match, not a verified "
                  f"enforcement action): {rf!r}")
           for rf in sentiment.get("red_flags", [])]


# ==============================================================================
# NFO ALERTS + DOSSIER
# ==============================================================================
def _manager_prior_funds(managers_df: Optional[pd.DataFrame], manager: str,
                         exclude_scheme: str) -> Dict[str, str]:
    """Other schemes this manager has run, per the hand-sourced managers.csv —
    skill attaches to the manager, not this (as-yet-unpriced) scheme code.

    Delegates to mf_managers so the matching is whitespace/case tolerant the same way
    the tenure lookup is; a name typed with a trailing space used to miss silently."""
    return mf_managers.prior_funds(managers_df, manager, exclude_scheme)


def _nfo(dossier: "FundDossier", gate: EligibilityGate, assessor: NFOAssessor,
        managers_df: Optional[pd.DataFrame]) -> Tuple[List[Alert], Optional[Dict[str, Any]], str]:
    # FundDossier carries no allotment_date field; the NAV series' own earliest
    # date is the best available proxy (real stores start `nav` at inception),
    # so the gate's NEWBORN/YOUNG age tiers still fire instead of silently
    # defaulting every fund with >=1 NAV point to EVALUABLE.
    navd = dossier.nav.dropna()
    allotment_date = navd.index.min().date() if len(navd) else None
    verdict = gate.assess(dossier.scheme_name, dossier.nav, allotment_date=allotment_date,
                         has_disclosure=dossier.holdings_available)
    status = verdict.status.value
    alerts: List[Alert] = []
    if status in ("PRE_LAUNCH", "NEWBORN"):
        alerts.append(Alert("NFO_NO_TRACK_RECORD", AlertSeverity.INFO.value, "nfo",
                            AlertBasis.COVERAGE.value, {"status": status}, verdict.reason))
        structural: Dict[str, str] = {}
        if np.isfinite(dossier.expense_ratio):
            structural["Expense Ratio"] = f"{dossier.expense_ratio:.2%}"
        if np.isfinite(dossier.aum_cr):
            structural["AUM"] = f"₹{dossier.aum_cr:,.0f} cr"
        prior_funds = _manager_prior_funds(managers_df, dossier.manager, dossier.scheme_name)
        dossier_obj = assessor.assess(dossier.scheme_name, dossier.amc, [dossier.manager],
                                      dossier.category, dossier.benchmark, verdict,
                                      manager_prior_funds=prior_funds, structural=structural)
        if not dossier_obj.manager_prior_funds:
            alerts.append(Alert("NFO_MANAGER_PROXY_UNAVAILABLE", AlertSeverity.INFO.value, "nfo",
                                AlertBasis.COVERAGE.value, {},
                                "No manager track-record proxy available (needs mf_cache/managers.csv)."))
        nfo_dossier = dict(scheme_name=dossier_obj.scheme_name, amc=dossier_obj.amc,
                           managers=dossier_obj.managers, category=dossier_obj.category,
                           benchmark=dossier_obj.benchmark, status=status,
                           manager_prior_funds=dossier_obj.manager_prior_funds,
                           structural=dossier_obj.structural, considerations=dossier_obj.considerations,
                           rendered=dossier_obj.render())
        return alerts, nfo_dossier, status
    if status == "YOUNG":
        alerts.append(Alert("NFO_SHORT_HISTORY", AlertSeverity.INFO.value, "nfo",
                            AlertBasis.COVERAGE.value, {"status": status}, verdict.reason))
    return alerts, None, status


_SEVERITY_ORDER = {AlertSeverity.HIGH.value: 0, AlertSeverity.WATCH.value: 1, AlertSeverity.INFO.value: 2}


class SentinelEngine:
    """System B. Runs per-fund alongside System A; never reads A's verdict/colours."""

    def __init__(self, validator: ValidationOrchestrator, managers_csv: str = "mf_cache/managers.csv") -> None:
        self.quant = QuantEngine(validator)
        self.v = validator
        self.gate = EligibilityGate()
        self.assessor = NFOAssessor()
        self._managers_df = _load_managers_csv(managers_csv)

    def run(self, dossier: "FundDossier", compliance: List["ComplianceFinding"],
            backtest: Dict[str, Any], sentiment: Dict[str, Any],
            benchmark: Optional[pd.Series]) -> SentinelReport:
        alerts: List[Alert] = []
        dormant: List[str] = []

        comp_alerts, checks_run = _compliance_alerts(compliance)
        alerts += comp_alerts

        nfo_alerts, nfo_dossier, status = _nfo(dossier, self.gate, self.assessor, self._managers_df)
        alerts += nfo_alerts

        if status == "EVALUABLE":
            fa, fdorm, fchecks = _factor_alerts(self.quant, dossier, benchmark)
            alerts += fa
            dormant += fdorm
            checks_run += fchecks
        else:
            dormant.append(f"FACTOR_RULES_SKIPPED_{status}")

        # Outside the eligibility gate on purpose — holdings structure needs a
        # disclosure, not NAV history, so it still speaks for a NEWBORN/YOUNG fund
        # whose factor rules were just skipped above.
        ha, hdorm, hchecks = _holdings_alerts(dossier)
        alerts += ha
        dormant += hdorm
        checks_run += hchecks

        ma, mdorm = _manager_alerts(backtest)
        alerts += ma
        dormant += mdorm
        if self._managers_df is None:
            dormant += ["MANAGER_CROSS_FUND_WEAK", "MANAGER_TENURE_SHORT"]
        else:
            t = _tenure_alert(self._managers_df, dossier)
            if t:
                alerts.append(t)
            # Cross-fund MACS aggregation needs orchestrating multiple evaluate()
            # calls across a manager's schemes — not wired in this pass (build gap).
            dormant.append("MANAGER_CROSS_FUND_WEAK")

        alerts += _news_alerts(sentiment)

        alerts.sort(key=lambda a: _SEVERITY_ORDER[a.severity])
        self.v._record("sentinel", "alerts_emitted", True,
                       f"{dossier.scheme_name}: {len(alerts)} alerts, {len(dormant)} dormant")
        return SentinelReport(alerts=alerts, nfo_dossier=nfo_dossier,
                              checks_run=checks_run, checks_dormant=dormant,
                              eligibility=status)


# ==============================================================================
# SELFTEST — registry invariant + live-computable rules trip on synthetic fixtures
# ==============================================================================
def _selftest() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", "2026-07-13")
    n = len(dates)

    def _path(mu, sigma, start=100.0):
        rets = rng.normal((mu - 0.5 * sigma ** 2) / 252, sigma / np.sqrt(252), n - 1)
        return pd.Series(start * np.cumprod(1 + np.concatenate([[0.0], rets])), index=dates)

    bench = _path(0.12, 0.16)
    validator = ValidationOrchestrator()
    quant = QuantEngine(validator)

    # 1. Registry invariant: constructing an Alert with a never-HIGH basis + HIGH must raise.
    raised = False
    try:
        Alert("X", AlertSeverity.HIGH.value, "factor", AlertBasis.GUARDRAIL_ONLY.value, {}, "x")
    except ValueError:
        raised = True
    assert raised, "FAIL: GUARDRAIL_ONLY alert accepted HIGH severity"
    print("[selftest] registry invariant: GUARDRAIL_ONLY/DESCRIPTIVE_RISK/etc cannot carry HIGH — PASS")

    # 2. A fund that closely tracks its benchmark with a mild positive tilt -> no WATCH/HIGH factor alerts.
    good = bench * (1 + rng.normal(0.0002, 0.0003, n)).cumprod()
    good = pd.Series(good.to_numpy(), index=dates)
    m_good = _fund_metrics(quant, good, bench)
    good_alerts = _equity_alerts(m_good, "Flexi Cap")
    bad_sev = [a for a in good_alerts if a.severity != AlertSeverity.INFO.value]
    assert not bad_sev, f"FAIL: benchmark-tracking fund tripped non-INFO alerts: {bad_sev}"
    print(f"[selftest] tracking fund: {len(good_alerts)} INFO-only alerts — PASS")

    # 3. A fund with amplified downside + real lag + deep drawdown -> the matching alerts fire.
    shock = np.zeros(n)
    shock[n // 2:n // 2 + 40] = -0.01
    bad = bench.to_numpy() * np.cumprod(1 + shock) * 0.8
    bad = pd.Series(bad, index=dates)
    m_bad = _fund_metrics(quant, bad, bench)
    bad_alerts = {a.code for a in _equity_alerts(m_bad, "Flexi Cap")}
    assert "EQ_DRAWDOWN_DEEP" in bad_alerts or "EQ_DRAWDOWN_VS_BENCH" in bad_alerts, \
        f"FAIL: shocked fund did not trip a drawdown alert: {bad_alerts}"
    print(f"[selftest] shocked fund tripped: {sorted(bad_alerts)} — PASS")

    # 4. Missing benchmark -> zero benchmark-relative alerts, no crash.
    m_nobench = _fund_metrics(quant, good, None)
    assert not np.isfinite(m_nobench["excess_cagr_3y"]) and not np.isfinite(m_nobench["beta"])
    nobench_alerts = _equity_alerts(m_nobench, "Flexi Cap")
    assert all(a.code not in ("EQ_BENCHMARK_LAG", "EQ_BETA_DRIFT", "EQ_DRAWDOWN_VS_BENCH")
              for a in nobench_alerts), "FAIL: benchmark-relative alert fired with no benchmark"
    print("[selftest] missing benchmark: no benchmark-relative alerts fired — PASS")

    # 5. Compliance mapping: critical -> HIGH, NOT-AVAILABLE warning -> INFO/COVERAGE, other warning -> WATCH.
    class _F:
        def __init__(self, rule_id, passed, severity, detail):
            self.rule_id, self.passed, self.severity, self.detail = rule_id, passed, severity, detail
    findings = [
        _F("min_equity>=65%", False, "critical", "equity 40%"),
        _F("cap_fidelity_min_equity_and_overlap_checks", False, "warning", "NOT AVAILABLE"),
        _F("scheme_name_matches_category", False, "warning", "mismatch"),
        _F("category_known", False, "warning", "Category 'Foo Bar' not in Feb-2026 registry"),
        _F("no_return_focused_words_in_name", True, "info", "ok"),
    ]
    comp_alerts, checks_run = _compliance_alerts(findings)
    by_code = {a.code: a for a in comp_alerts}
    assert by_code["SEBI_MIN_EQUITY_BREACH"].severity == AlertSeverity.HIGH.value
    assert by_code["SEBI_CHECKS_NOT_EVALUATED"].severity == AlertSeverity.INFO.value
    assert by_code["SEBI_CHECKS_NOT_EVALUATED"].basis == AlertBasis.COVERAGE.value
    assert by_code["NAME_CATEGORY_MISMATCH"].severity == AlertSeverity.WATCH.value
    assert by_code["SEBI_CATEGORY_UNKNOWN"].severity == AlertSeverity.INFO.value, \
        "FAIL: category_known (a coverage gap, not a verified breach) must not be WATCH/REGULATORY"
    assert by_code["SEBI_CATEGORY_UNKNOWN"].basis == AlertBasis.COVERAGE.value
    assert checks_run == 5 and len(comp_alerts) == 4
    print("[selftest] compliance mapping: critical->HIGH, NOT-AVAILABLE/category-unknown->INFO/COVERAGE, "
          "other warning->WATCH — PASS")

    # 6. Manager alerts: MACS<40 -> WATCH, 40<=MACS<60 -> INFO, >=60 -> nothing.
    assert _manager_alerts({"manager_alpha_consistency_score": 30})[0][0].severity == AlertSeverity.WATCH.value
    assert _manager_alerts({"manager_alpha_consistency_score": 50})[0][0].severity == AlertSeverity.INFO.value
    assert _manager_alerts({"manager_alpha_consistency_score": 75})[0] == []
    print("[selftest] manager MACS tiers: <40 WATCH, 40-59 INFO, >=60 silent — PASS")

    # 7. The SEBI single-issuer breach gets its OWN typed code, not the SEBI_OTHER[...]
    #    catch-all — the app branches on alert codes, so a rule_id embedded in the code
    #    string is not a stable contract. Severity/basis were already correct via
    #    `critical`; this pins the code.
    si_alerts, _ = _compliance_alerts(
        [_F("single_issuer<=10%", False, "critical", "1 equity issuer over SEBI's 10% ceiling: MEGA 22.0%")])
    assert si_alerts[0].code == "SEBI_SINGLE_ISSUER_BREACH", si_alerts[0].code
    assert si_alerts[0].severity == AlertSeverity.HIGH.value
    assert si_alerts[0].basis == AlertBasis.REGULATORY.value
    print("[selftest] single-issuer breach -> typed SEBI_SINGLE_ISSUER_BREACH, HIGH/REGULATORY — PASS")

    # 8. Holdings concentration alerts.
    class _B:
        def __init__(self, value): self.value = value

    class _D:
        def __init__(self, bucket, holdings, available=True):
            self.bucket, self.current_holdings, self.holdings_available = _B(bucket), holdings, available

    def _hframe(rows):
        df = pd.DataFrame(rows)
        df.index = df["name"].astype(str).str.upper()
        return df

    diversified = _hframe([dict(name=f"S{i:02d}", weight=0.030, asset_type="equity",
                                sector=["Financials", "IT", "Pharma", "Auto"][i % 4])
                           for i in range(30)])
    concentrated = _hframe([dict(name=f"C{i:02d}", weight=0.085, asset_type="equity", sector="IT")
                            for i in range(10)] + [dict(name=f"T{i:02d}", weight=0.015,
                                                        asset_type="equity", sector="Auto")
                                                   for i in range(10)])

    a, d, c = _holdings_alerts(_D("Equity", diversified))
    assert not a and not d and c == 1, f"FAIL: diversified book should be silent, got {a}/{d}"

    a, d, c = _holdings_alerts(_D("Equity", concentrated))
    assert len(a) == 1 and a[0].code == "HOLDINGS_CONCENTRATION_HIGH", a
    assert a[0].severity == AlertSeverity.WATCH.value, a[0].severity
    assert a[0].basis == AlertBasis.DESCRIPTIVE_RISK.value
    assert a[0].evidence["top10_weight"] == 0.85 and a[0].evidence["n_equity"] == 20
    assert not d and c == 1

    # No disclosure -> dormant, never a silent pass and never an alert.
    a, d, c = _holdings_alerts(_D("Equity", None, available=False))
    assert not a and d == ["HOLDINGS_CONCENTRATION_HIGH"] and c == 0, f"{a}/{d}/{c}"

    # Index/ETF: a concentrated top-10 is the INDEX's property, not a manager choice.
    a, d, c = _holdings_alerts(_D("Other (Index/ETF/FoF)", concentrated))
    assert not a and not d and c == 0, "FAIL: passive bucket must be mandate-inapplicable, not alerted"

    # DESCRIPTIVE_RISK is structurally barred from HIGH — a concentrated book can
    # never escalate to the push-notification tier the way a real breach does.
    raised = False
    try:
        Alert("HOLDINGS_CONCENTRATION_HIGH", AlertSeverity.HIGH.value, "holdings",
              AlertBasis.DESCRIPTIVE_RISK.value, {}, "x")
    except ValueError:
        raised = True
    assert raised, "FAIL: concentration alert accepted HIGH severity"
    print("[selftest] holdings: concentrated->WATCH, diversified silent, no-disclosure dormant, "
          "passive exempt, never HIGH — PASS")

    print("[selftest] PASS — Sentinel registry + live rules behave honestly")


if __name__ == "__main__":
    _selftest()
