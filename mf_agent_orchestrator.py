"""
====================================================================================
MASTER ORCHESTRATOR — AUTONOMOUS MULTI-AGENT INDIAN MF EVALUATION PIPELINE (v2)
====================================================================================
Re-architecture of the v1 monolithic pipeline (`mf_pipeline.py`, reviewed and
imported below for its hardened primitives) into an autonomous, agentic system:

    MasterOrchestrator
      ├── AGENT A  DataIngestionCategorizerAgent   (ingestion, SEBI auto-categorise)
      ├── AGENT B  HistoricalBacktesterAgent       (top-10 holdings audit, 3y/5y,
      │                                             Manager Alpha Consistency Score)
      ├── AGENT C  ProfileRiskScorerAgent          (dynamic runtime utility matrix)
      └── AGENT D  NewsSentimentResearcherAgent    (AMC / manager / sector signals)

Agents are specialised workers with pluggable EXECUTION BACKENDS. In production,
Agent A/B run on cheap code-execution models against live endpoints, and Agent D
binds to an LLM web-search tool. In this sealed environment every external
dependency is wrapped in an Adapter carrying `is_live=False` with a deterministic
mock backend, and the run ends with an explicit USER ACTION CHECKLIST listing the
exact keys/endpoints needed to flip each adapter live. Execution never blocks.

SEBI TRUE-TO-LABEL BASIS (verified against the circular dated 26-Feb-2026 which
supersedes Clause 2.6 of the 27-Jun-2024 Master Circular):
  * Categories expanded 36 -> 40 (13 equity / 17 debt / 7 hybrid / 2 other /
    1 life-cycle).
  * Sectoral/Thematic: portfolio overlap <= 50% vs other sectoral/thematic schemes
    AND other equity categories (large-cap schemes exempt); computed QUARTERLY as
    the average of DAILY overlap values; monthly disclosure; 3-year glide path.
  * Value & Contra split into separate categories; mutual overlap <= 50%;
    Dividend-Yield / Value / Contra minimum equity raised 65% -> 80%.
  * Market-cap fidelity: Large Cap >= 80% in large-caps; Mid >= 65% mid;
    Small >= 65% small (segment minima).
  * Solution-Oriented category DISCONTINUED w.e.f. 26-Feb-2026 (stop subscriptions,
    merge with similar scheme) — replaced by the new LIFE CYCLE fund category.
  * New Sectoral Debt Funds: >= 80% in debt instruments of a single sector.
  * Scheme name must equal category name; return-focused words prohibited.
====================================================================================
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- Reviewed & reused v1 primitives (hardened in the prior build) -----------
from mf_pipeline import (  # noqa: E402
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE_ANNUAL,
    QuantEngine,
    ValidationOrchestrator,
)
from mf_sentinel import SentinelEngine  # noqa: E402 — System B: typed alerts, never a fund number
import mf_holdings  # noqa: E402 — portfolio-structure facts + concentration screen (holdings-only)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
LOGGER = logging.getLogger("MFOrchestrator")
TODAY = pd.Timestamp("2026-07-13")


# ==============================================================================
# 1. INVESTOR PROFILE FRAMEWORK — dynamic runtime rules
# ==============================================================================
class LiquidityNeed(str, Enum):
    NONE = "none"          # fully locked-in capital, no interim dividends needed
    PARTIAL = "partial"    # occasional access possible
    HIGH = "high"          # may need capital at short notice


class RiskAppetite(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    HIGH = "high"          # wealth maximisation, preservation secondary


class InvestorProfile(BaseModel):
    """Runtime-mutable investor constraints driving every scoring function."""
    model_config = ConfigDict(frozen=True)

    horizon_years: float = Field(default=7.5, gt=0.25, le=40,
                                 description="Baseline default: 7-8y target inside a 5-10y band")
    liquidity_need: LiquidityNeed = Field(default=LiquidityNeed.NONE)
    risk_appetite: RiskAppetite = Field(default=RiskAppetite.HIGH)
    notes: str = Field(default="Secure Govt job + guaranteed pension; capital locked in; "
                               "inflation-beating growth prioritised over preservation; "
                               "category drift / unhedged thematic risk must be flagged.")

    @field_validator("horizon_years")
    @classmethod
    def _sane_horizon(cls, v: float) -> float:
        return round(float(v), 2)


def resolve_investor_profile(config: Optional[Dict[str, Any]] = None,
                             argv: Optional[Sequence[str]] = None,
                             interactive: bool = True) -> InvestorProfile:
    """
    MANDATORY runtime hook, executed at the start of EVERY pipeline run.
    Precedence: CLI arguments > config dict > interactive prompt > baseline default.
    Never blocks: in non-TTY / piped contexts the interactive step is skipped and
    the resolved (or default) profile is logged instead.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--horizon-years", type=float, default=None)
    parser.add_argument("--liquidity-need", choices=[e.value for e in LiquidityNeed], default=None)
    parser.add_argument("--risk-appetite", choices=[e.value for e in RiskAppetite], default=None)
    ns, _ = parser.parse_known_args(argv if argv is not None else [])

    merged: Dict[str, Any] = {}
    if config:
        merged.update({k: v for k, v in config.items() if v is not None})
    for cli_key, model_key in (("horizon_years", "horizon_years"),
                               ("liquidity_need", "liquidity_need"),
                               ("risk_appetite", "risk_appetite")):
        val = getattr(ns, cli_key)
        if val is not None:
            merged[model_key] = val

    if not merged and interactive and sys.stdin.isatty():
        print("\n--- Investor profile (Enter to accept baseline default) ---")
        h = input("Horizon in years [7.5]: ").strip()
        l = input("Liquidity need none/partial/high [none]: ").strip().lower()
        r = input("Risk appetite conservative/moderate/high [high]: ").strip().lower()
        if h:
            merged["horizon_years"] = float(h)
        if l in {e.value for e in LiquidityNeed}:
            merged["liquidity_need"] = l
        if r in {e.value for e in RiskAppetite}:
            merged["risk_appetite"] = r

    profile = InvestorProfile(**merged)
    LOGGER.info("Investor profile resolved: horizon=%.1fy liquidity=%s risk=%s (%s)",
                profile.horizon_years, profile.liquidity_need.value,
                profile.risk_appetite.value,
                "overridden at runtime" if merged else "baseline default")
    return profile


# ==============================================================================
# 2. SEBI FEB-2026 CATEGORISATION REGISTRY + TRUE-TO-LABEL VERIFIER
# ==============================================================================
class SEBIBucket(str, Enum):
    EQUITY = "Equity"
    DEBT = "Debt"
    HYBRID = "Hybrid"
    LIFE_CYCLE = "Life Cycle"          # NEW: replaces Solution-Oriented
    OTHER_PASSIVE = "Other (Index/ETF/FoF)"
    SOLUTION_ORIENTED_LEGACY = "Solution Oriented (DISCONTINUED)"


@dataclass(frozen=True)
class CategoryRule:
    bucket: SEBIBucket
    min_segment_weight: Optional[float] = None     # market-cap / sector fidelity floor
    segment: Optional[str] = None                  # 'large' | 'mid' | 'small' | 'single_sector_debt'
    min_equity: Optional[float] = None             # total-equity floor
    overlap_cap: Optional[float] = None            # 50% rule where applicable
    overlap_scope: Optional[str] = None            # who the cap is measured against
    notes: str = ""


SEBI_2026_RULES: Dict[str, CategoryRule] = {
    "Large Cap":  CategoryRule(SEBIBucket.EQUITY, 0.80, "large", 0.80,
                               notes="≥80% in large-cap stocks (top 100 by mkt cap)"),
    "Mid Cap":    CategoryRule(SEBIBucket.EQUITY, 0.65, "mid", 0.65,
                               notes="≥65% in mid-cap stocks (101st–250th)"),
    "Small Cap":  CategoryRule(SEBIBucket.EQUITY, 0.65, "small", 0.65,
                               notes="≥65% in small-cap stocks (251st onwards)"),
    "Flexi Cap":  CategoryRule(SEBIBucket.EQUITY, None, None, 0.65),
    "Value":      CategoryRule(SEBIBucket.EQUITY, None, None, 0.80, 0.50, "contra_sibling",
                               notes="Feb-2026: equity floor raised 65%→80%; ≤50% overlap with sibling Contra"),
    "Contra":     CategoryRule(SEBIBucket.EQUITY, None, None, 0.80, 0.50, "value_sibling"),
    "Dividend Yield": CategoryRule(SEBIBucket.EQUITY, None, None, 0.80,
                                   notes="Feb-2026: equity floor raised 65%→80%"),
    "Sectoral":   CategoryRule(SEBIBucket.EQUITY, 0.80, "declared_sector", 0.80, 0.50,
                               "all_equity_except_large_cap",
                               notes="≤50% quarterly-avg daily overlap vs sibling equity schemes "
                                     "(large-cap exempt); AMFI half-yearly sector list; 3y glide path"),
    "Thematic":   CategoryRule(SEBIBucket.EQUITY, 0.80, "declared_theme", 0.80, 0.50,
                               "all_equity_except_large_cap"),
    "Sectoral Debt": CategoryRule(SEBIBucket.DEBT, 0.80, "single_sector_debt",
                                  notes="NEW Feb-2026 category: ≥80% debt of a single sector"),
    "Corporate Bond": CategoryRule(SEBIBucket.DEBT),
    "Short Duration": CategoryRule(SEBIBucket.DEBT),
    "Aggressive Hybrid": CategoryRule(SEBIBucket.HYBRID, None, None, 0.65,
                                      notes="equity 65–80%"),
    "Conservative Hybrid": CategoryRule(SEBIBucket.HYBRID, None, None, None,
                                        notes="equity 10–25%"),
    "Arbitrage":  CategoryRule(SEBIBucket.HYBRID, None, None, 0.65),
    "Life Cycle": CategoryRule(SEBIBucket.LIFE_CYCLE,
                               notes="NEW planning-led multi-asset category replacing "
                                     "Solution-Oriented (discontinued 26-Feb-2026)"),
    "Index Fund": CategoryRule(SEBIBucket.OTHER_PASSIVE),
    "ETF":        CategoryRule(SEBIBucket.OTHER_PASSIVE),
    "Retirement/Children (legacy)": CategoryRule(
        SEBIBucket.SOLUTION_ORIENTED_LEGACY,
        notes="Subscriptions must STOP immediately; scheme to merge with a similar "
              "asset-allocation scheme with prior SEBI approval"),
}

RETURN_FOCUSED_NAME_WORDS = ("wealth booster", "multibagger", "assured", "guaranteed",
                             "high return", "supreme return", "double")


@dataclass
class ComplianceFinding:
    rule_id: str
    passed: bool
    severity: str          # info | warning | critical
    detail: str


class TrueToLabelVerifier:
    """Encodes the Feb-2026 circular as executable checks over live portfolio data."""

    def __init__(self, validator: ValidationOrchestrator) -> None:
        self.v = validator
        self.log = logging.getLogger("MFOrchestrator.TrueToLabel")

    # Overlap between two weight vectors = Σ min(w1_i, w2_i), the circular's basis.
    @staticmethod
    def portfolio_overlap(w1: pd.Series, w2: pd.Series) -> float:
        aligned = pd.concat([w1, w2], axis=1).fillna(0.0)
        return float(np.minimum(aligned.iloc[:, 0], aligned.iloc[:, 1]).sum())

    def quarterly_overlap(self, daily_w1: Dict[str, pd.Series],
                          daily_w2: Dict[str, pd.Series]) -> float:
        """Feb-2026 method: quarterly figure = average of DAILY overlap values."""
        days = sorted(set(daily_w1) & set(daily_w2))
        if not days:
            return np.nan
        vals = [self.portfolio_overlap(daily_w1[d], daily_w2[d]) for d in days]
        return float(np.mean(vals))

    def verify(self, fund: "FundDossier",
               sibling_daily_weights: Dict[str, Dict[str, pd.Series]]) -> List[ComplianceFinding]:
        findings: List[ComplianceFinding] = []
        rule = SEBI_2026_RULES.get(fund.category)
        if rule is None:
            findings.append(ComplianceFinding("category_known", False, "warning",
                                              f"Category {fund.category!r} not in Feb-2026 registry"))
            return findings

        # (a) Discontinued category surveillance
        if rule.bucket == SEBIBucket.SOLUTION_ORIENTED_LEGACY:
            findings.append(ComplianceFinding(
                "solution_oriented_discontinued", False, "critical",
                "Solution-Oriented category discontinued w.e.f. 26-Feb-2026: scheme must stop "
                "subscriptions and merge (SEBI approval) — evaluate successor Life Cycle scheme instead"))

        if not fund.holdings_available:
            # No manual holdings-disclosure CSV on file for this fund (real-data
            # mode). The (b)/(c)/(d) checks below all need real portfolio weights;
            # running them against an empty portfolio would report a fabricated
            # "0% equity / 0% cap-fidelity" FAIL that isn't true — the data is
            # simply missing. Say so honestly instead of scoring it.
            findings.append(ComplianceFinding(
                "cap_fidelity_min_equity_and_overlap_checks", False, "warning",
                "NOT AVAILABLE — no portfolio holdings disclosure on file "
                "(mf_cache/disclosures/<code>_<YYYY-MM>.csv missing). SEBI cap-fidelity, "
                "minimum-equity and 50% overlap checks require real holdings and are "
                "NOT EVALUATED here (not failed, not fabricated)."))
        else:
            holdings = fund.current_holdings
            eq = holdings[holdings["asset_type"] == "equity"]
            eq_w = float(eq["weight"].sum())

            # (b) Market-cap / sector segment fidelity
            if rule.min_segment_weight is not None and rule.segment in ("large", "mid", "small"):
                seg_w = float(eq.loc[eq["cap_band"] == rule.segment, "weight"].sum())
                ok = seg_w >= rule.min_segment_weight - 1e-9
                findings.append(ComplianceFinding(
                    f"cap_fidelity_{rule.segment}>= {rule.min_segment_weight:.0%}", ok,
                    "critical" if not ok else "info",
                    f"{rule.segment}-cap exposure {seg_w:.1%} vs floor {rule.min_segment_weight:.0%}"))
            if rule.segment in ("declared_sector", "declared_theme") and fund.declared_sector:
                sec_w = float(eq.loc[eq["sector"] == fund.declared_sector, "weight"].sum())
                ok = sec_w >= (rule.min_segment_weight or 0.80) - 1e-9
                findings.append(ComplianceFinding(
                    "sector_theme_fidelity>=80%", ok, "critical" if not ok else "info",
                    f"{fund.declared_sector} exposure {sec_w:.1%}"))
            if rule.segment == "single_sector_debt":
                debt = holdings[holdings["asset_type"] == "debt"]
                top_sec = debt.groupby("sector")["weight"].sum()
                share = float(top_sec.max()) if len(top_sec) else 0.0
                ok = share >= 0.80
                findings.append(ComplianceFinding("sectoral_debt>=80%_single_sector", ok,
                                                  "critical" if not ok else "info",
                                                  f"max single-sector debt share {share:.1%}"))

            # (c) Minimum-equity floors (incl. Feb-2026 raise for Value/Contra/Div-Yield)
            if rule.min_equity is not None:
                ok = eq_w >= rule.min_equity - 1e-9
                findings.append(ComplianceFinding(
                    f"min_equity>={rule.min_equity:.0%}", ok, "critical" if not ok else "info",
                    f"equity allocation {eq_w:.1%}"))

            # (c2) SEBI single-issuer 10% ceiling (MF Regulations) — a scheme may hold
            # at most 10% of NAV in the equity of one issuer. Index/ETF schemes are
            # exempt and sit in OTHER_PASSIVE, so restricting to the EQUITY/HYBRID
            # buckets skips them naturally. A breach on an active mandate is a real
            # regulatory line, so it rides the same critical -> SELL gate as (b)/(c).
            if rule.bucket in (SEBIBucket.EQUITY, SEBIBucket.HYBRID):
                hf = mf_holdings.analyze(holdings, available=True)
                for rid, passed, detail in mf_holdings.single_issuer_findings(hf):
                    findings.append(ComplianceFinding(
                        rid, passed, "critical" if not passed else "info", detail))

            # (d) 50% overlap ceiling — quarterly average of daily overlaps
            if rule.overlap_cap is not None and sibling_daily_weights:
                for sib_name, sib_daily in sibling_daily_weights.items():
                    q_ov = self.quarterly_overlap(fund.daily_equity_weights, sib_daily)
                    if np.isnan(q_ov):
                        continue
                    ok = q_ov <= rule.overlap_cap + 1e-9
                    findings.append(ComplianceFinding(
                        f"overlap_vs[{sib_name}]<=50%", ok, "critical" if not ok else "info",
                        f"quarterly-avg daily overlap {q_ov:.1%} (scope: {rule.overlap_scope}; "
                        f"3y glide path applies to pre-existing thematic schemes)"))

        # (e) Naming discipline: name==category; no return-focused words
        name_l = fund.scheme_name.lower()
        cat_in_name = fund.category.lower() in name_l
        findings.append(ComplianceFinding("scheme_name_matches_category", cat_in_name,
                                          "warning" if not cat_in_name else "info",
                                          f"name={fund.scheme_name!r} category={fund.category!r}"))
        bad_words = [w for w in RETURN_FOCUSED_NAME_WORDS if w in name_l]
        findings.append(ComplianceFinding("no_return_focused_words_in_name", not bad_words,
                                          "warning" if bad_words else "info",
                                          f"flagged={bad_words}"))
        for f in findings:
            self.v._record(f"sebi2026:{fund.scheme_name}", f.rule_id, f.passed, f.detail)
        return findings


# ==============================================================================
# 3. EXTERNAL DATA ADAPTERS — live-ready interfaces with deterministic mocks
# ==============================================================================
@dataclass
class AdapterSpec:
    name: str
    is_live: bool
    endpoint: str
    required_config: str
    used_by: str


ADAPTER_REGISTRY: List[AdapterSpec] = []


def register_adapter(spec: AdapterSpec) -> AdapterSpec:
    ADAPTER_REGISTRY.append(spec)
    if not spec.is_live:
        LOGGER.warning("Adapter %-24s running in MOCK mode (%s)", spec.name, spec.required_config)
    return spec


class MockMarketDataStore:
    """
    Deterministic synthetic Indian market: sector indices, ~60 NSE stocks with
    cap bands, 7.5 years of prices (sector-driven paths — the v1 learning that
    independent GBM paths break tracking/beta realism), fund NAVs, and monthly
    portfolio disclosures at t-5y, t-3y and current. Stands in for AMFI /
    exchange / registrar feeds until live keys are supplied.
    """

    SECTORS = ["Financials", "IT", "Energy", "Pharma", "Auto", "FMCG", "Infrastructure"]

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.dates = pd.bdate_range(TODAY - pd.DateOffset(years=8), TODAY)
        self.sector_indices = self._build_sector_indices()
        self.stocks = self._build_stocks()
        self._px = self._build_stock_prices()
        self.funds = self._build_funds()

    def _path(self, mu: float, sigma: float, start: float = 100.0,
              driver: Optional[pd.Series] = None, beta: float = 1.0,
              idio: float = 0.0) -> pd.Series:
        n = len(self.dates)
        dt = 1.0 / TRADING_DAYS_PER_YEAR
        if driver is None:
            rets = self.rng.normal((mu - 0.5 * sigma ** 2) * dt, sigma * math.sqrt(dt), n - 1)
            rets = np.concatenate([[0.0], rets])
        else:
            d = driver.pct_change().fillna(0.0).to_numpy()
            rets = mu * dt + beta * d + self.rng.normal(0, idio * math.sqrt(dt), n)
        rets = np.clip(rets, -0.18, 0.18)
        return pd.Series(start * np.cumprod(1 + rets), index=self.dates)

    def _build_sector_indices(self) -> Dict[str, pd.Series]:
        params = {"Financials": (0.13, 0.19), "IT": (0.11, 0.21), "Energy": (0.12, 0.20),
                  "Pharma": (0.115, 0.18), "Auto": (0.125, 0.22), "FMCG": (0.10, 0.14),
                  "Infrastructure": (0.14, 0.23)}
        idx = {s: self._path(mu, sig) for s, (mu, sig) in params.items()}
        idx["NIFTY 50 TRI"] = self._path(0.123, 0.165)
        idx["NIFTY SMALLCAP 250 TRI"] = self._path(0.15, 0.25)
        idx["NIFTY 500 TRI"] = self._path(0.128, 0.17)
        return idx

    def _build_stocks(self) -> pd.DataFrame:
        rows = []
        for i in range(60):
            sector = self.SECTORS[i % len(self.SECTORS)]
            band = "large" if i < 22 else ("mid" if i < 42 else "small")
            rows.append(dict(ticker=f"{sector[:3].upper()}{i:02d}.NS",
                             name=f"{sector} Co {i:02d}", sector=sector, cap_band=band,
                             # per-stock skill: some names genuinely beat their sector
                             true_alpha=float(self.rng.normal(0.0, 0.05)),
                             idio_vol=float(self.rng.uniform(0.10, 0.30))))
        return pd.DataFrame(rows).set_index("ticker")

    def _build_stock_prices(self) -> pd.DataFrame:
        cols = {}
        for t, row in self.stocks.iterrows():
            cols[t] = self._path(row["true_alpha"], 0.0, start=float(self.rng.uniform(50, 3000)),
                                 driver=self.sector_indices[row["sector"]],
                                 beta=float(self.rng.uniform(0.85, 1.15)), idio=row["idio_vol"])
        return pd.DataFrame(cols)

    # ---------------------- synthetic fund construction -----------------------
    def _disclosure(self, tickers: List[str], raw_w: np.ndarray) -> pd.DataFrame:
        w = raw_w / raw_w.sum() * 0.965
        df = self.stocks.loc[tickers, ["name", "sector", "cap_band"]].copy()
        df["weight"] = w
        df["asset_type"] = "equity"
        cash = pd.DataFrame([{"name": "TREPS/Cash", "sector": "Cash", "cap_band": "cash",
                              "weight": 0.035, "asset_type": "cash"}], index=["CASH"])
        return pd.concat([df, cash])

    def _pick(self, band: Optional[str], sector: Optional[str], n: int,
              skill: float) -> List[str]:
        pool = self.stocks
        if band:
            pool = pool[pool["cap_band"] == band]
        if sector:
            pool = pool[pool["sector"] == sector]
        # skill in [0,1]: skilled managers over-sample genuinely high-alpha stocks
        alpha_rank = pool["true_alpha"].rank(pct=True).to_numpy()
        p = (1 - skill) + skill * 2.2 * alpha_rank
        p = p / p.sum()
        return list(self.rng.choice(pool.index, size=min(n, len(pool)), replace=False, p=p))

    def _build_funds(self) -> Dict[str, Dict[str, Any]]:
        funds: Dict[str, Dict[str, Any]] = {}

        def make(name, isin, category, amc, manager, band=None, sector=None,
                 n=18, skill=0.5, ter=0.012, nav_alpha=0.0, declared_sector=None):
            snaps = {}
            for label, yrs in (("t-5y", 5), ("t-3y", 3), ("current", 0)):
                tickers = self._pick(band, sector, n, skill)
                snaps[label] = {"as_of": (TODAY - pd.DateOffset(years=yrs)).normalize(),
                                "holdings": self._disclosure(
                                    tickers, self.rng.dirichlet(np.ones(len(tickers)) * 2.5))}
            bench = declared_sector if declared_sector else \
                ("NIFTY SMALLCAP 250 TRI" if band == "small" else "NIFTY 50 TRI")
            nav = self._path(nav_alpha - ter, 0.0, start=float(self.rng.uniform(15, 400)),
                             driver=self.sector_indices[bench],
                             beta=1.0 if category in ("Index Fund", "ETF") else
                             float(self.rng.uniform(0.9, 1.05)),
                             idio=0.004 if category in ("Index Fund", "ETF")
                             else float(self.rng.uniform(0.03, 0.07)))
            funds[name] = dict(isin=isin, category=category, amc=amc, manager=manager,
                               declared_sector=declared_sector, benchmark=bench,
                               expense_ratio=ter, aum_cr=float(self.rng.uniform(1500, 40000)),
                               nav=nav, snapshots=snaps)
        make("Bharat Small Cap Fund", "INF001SC00010", "Small Cap", "Bharat AMC",
             "R. Iyer", band="small", n=20, skill=0.85, ter=0.008, nav_alpha=0.035)
        make("Bharat Banking & PSU Thematic Fund", "INF001TH00027", "Thematic", "Bharat AMC",
             "K. Menon", sector="Financials", n=14, skill=0.25, ter=0.019,
             nav_alpha=-0.01, declared_sector="Financials")
        make("Bharat Financial Services Sectoral Fund", "INF001SE00034", "Sectoral",
             "Bharat AMC", "K. Menon", sector="Financials", n=14, skill=0.30, ter=0.021,
             nav_alpha=-0.005, declared_sector="Financials")
        make("Alpha Nifty 50 Index Fund", "INF002IX00019", "Index Fund", "Alpha AMC",
             "Passive Desk", band="large", n=22, skill=0.0, ter=0.0015, nav_alpha=0.0)
        make("Deccan Retirement Fund", "INF003RT00042", "Retirement/Children (legacy)",
             "Deccan AMC", "S. Rao", band="large", n=16, skill=0.4, ter=0.017, nav_alpha=0.005)
        return funds

    # daily equity weight paths over last quarter (for the quarterly-overlap rule):
    def daily_weights_last_quarter(self, fund_name: str) -> Dict[str, pd.Series]:
        snap = self.funds[fund_name]["snapshots"]["current"]["holdings"]
        eq = snap[snap["asset_type"] == "equity"]["weight"]
        out: Dict[str, pd.Series] = {}
        qdays = self.dates[self.dates >= TODAY - pd.DateOffset(months=3)]
        for d in qdays:
            drift = pd.Series(self.rng.normal(1.0, 0.01, len(eq)), index=eq.index)
            w = (eq * drift).clip(lower=0)
            out[str(d.date())] = w / w.sum() * eq.sum()
        return out

    # ---- STORE INTERFACE ------------------------------------------------------
    # The agents talk to a store through this narrow, stable interface. Implementing
    # it here keeps the demo runnable offline; mf_datasources.py implements the SAME
    # interface against live AMFI / mfapi / yfinance / Anthropic, so swapping in real
    # data is a constructor change, not a rewrite.
    def resolve(self, query: str):
        q = query.strip().lower()
        for name, rec in self.funds.items():
            if q == rec["isin"].lower() or q in name.lower():
                return name
        for name in self.funds:
            if all(tok in name.lower() for tok in q.split()[:2]):
                return name
        return None

    def fund(self, name: str):
        return self.funds[name]

    def snapshots(self, name: str):
        return self.funds[name]["snapshots"]

    def stock_prices(self, tickers, start):
        cols = [t for t in tickers if t in self._px.columns]
        return self._px.loc[start:, cols]

    def sector_index(self, sector: str, start, basket=None):
        return self.sector_indices[sector].loc[start:]

    def benchmark_series(self, benchmark: str):
        return self.sector_indices[benchmark]

    def sibling_equity_schemes(self, scheme_name: str, amc: str):
        out = {}
        for name, rec in self.funds.items():
            if name == scheme_name or rec["amc"] != amc:
                continue
            if rec["category"] in ("Large Cap", "Index Fund", "ETF"):
                continue  # SEBI: large-cap schemes exempt from the overlap cap
            if SEBI_2026_RULES.get(rec["category"],
                                   CategoryRule(SEBIBucket.DEBT)).bucket != SEBIBucket.EQUITY:
                continue
            out[name] = self.daily_weights_last_quarter(name)
        return out

    def news(self, entities):
        return NewsSearchAdapter().fetch(entities)


# --------------------------- adapter facades ----------------------------------
# Registry-only adapters — the live product resolves/discloses/prices via the store
# interface (see MockMarketDataStore), so these carry no facade class; only their
# AdapterSpec is registered at import to populate the USER ACTION CHECKLIST.
# Live: AMFI NAVAll.txt + SEBI category master. Mock: MockMarketDataStore.
register_adapter(AdapterSpec(
    "Scheme_Registry", True, "https://mfdata.in/api/v1/search + api.mfapi.in",
    "NO KEY REQUIRED — implemented in live_adapters.MFDataClient/MFApiClient", "Agent A"))
# Live: Morningstar/ACE MF/CAMS monthly portfolios. Mock: synthetic snapshots.
register_adapter(AdapterSpec(
    "Portfolio_Disclosures", True,
    "https://mfdata.in/api/v1/families/{id}/holdings?month=YYYY-MM",
    "NO KEY REQUIRED — free monthly holdings snapshots; replaces paid Morningstar/ACE",
    "Agents A & B"))
# Live: yfinance NSE tickers (.NS). Mock: sector-driven synthetic prices.
register_adapter(AdapterSpec(
    "Stock_Price_History", True, "yfinance (NSE .NS tickers + yf.Search name resolution)",
    "NO KEY REQUIRED — pip install yfinance", "Agent B"))


class NewsSearchAdapter:
    """Live: LLM web-search tool / NewsAPI / GDELT. Mock: curated event feed."""
    spec = register_adapter(AdapterSpec(
        "News_Sentiment_Search", False,
        "https://api.anthropic.com/v1/messages with web_search_20250305 tool",
        "ANTHROPIC_API_KEY  <-- THE ONLY KEY THE SYSTEM NEEDS", "Agent D"))

    _MOCK_FEED = [
        dict(entity="Bharat AMC", days_ago=6, headline="Bharat AMC reports record SIP inflows; small-cap book crosses ₹40,000 cr", tone=+1),
        dict(entity="R. Iyer", days_ago=21, headline="Fund manager R. Iyer completes 9 years at helm; long-tenure stability cited", tone=+1),
        dict(entity="K. Menon", days_ago=11, headline="SEBI seeks clarification from Bharat AMC on overlapping thematic offerings", tone=-1),
        dict(entity="Financials", days_ago=4, headline="RBI tightens NBFC provisioning norms; financial sector funds under pressure", tone=-1),
        dict(entity="Financials", days_ago=15, headline="Credit growth moderates to 11% YoY; bank margins guidance trimmed", tone=-1),
        dict(entity="Deccan AMC", days_ago=30, headline="Deccan AMC to merge retirement scheme after SEBI discontinues solution-oriented category", tone=0),
        dict(entity="Alpha AMC", days_ago=9, headline="Alpha AMC cuts index fund TER to 15 bps in passive price war", tone=+1),
        dict(entity="IT", days_ago=7, headline="IT services order books rebound on GenAI integration deals", tone=+1),
    ]

    def fetch(self, entities: Sequence[str]) -> List[Dict[str, Any]]:
        ents = {e.lower() for e in entities}
        return [dict(item, date=str((TODAY - pd.Timedelta(days=item["days_ago"])).date()))
                for item in self._MOCK_FEED if item["entity"].lower() in ents]


# ==============================================================================
# 4. THE FOUR SUB-AGENTS
# ==============================================================================
@dataclass
class FundDossier:
    """Agent A's normalized output — the single source of truth downstream."""
    scheme_name: str
    isin: str
    category: str
    bucket: SEBIBucket
    amc: str
    manager: str
    benchmark: str
    declared_sector: Optional[str]
    expense_ratio: float
    aum_cr: float
    nav: pd.Series
    snapshots: Dict[str, Dict[str, Any]]
    current_holdings: pd.DataFrame
    daily_equity_weights: Dict[str, pd.Series]
    missing_params: List[str] = field(default_factory=list)
    # False only for real-data stores when no manual holdings-disclosure CSV was
    # found for this fund. Gates the holdings-dependent checks below (Manager
    # Alpha score, SEBI cap-fidelity) so they report NOT AVAILABLE instead of
    # computing off an empty/zero portfolio. Always True for MockMarketDataStore.
    holdings_available: bool = True
    # AMFI scheme code — empty for MockMarketDataStore (which has no trained
    # cohort membership anyway); populated by RealNAVStore.fund(). This is the
    # only key the cohort_q1 live-scoring path (mf_live_score.py) needs to look
    # up a fund's trained-universe membership.
    amfi_code: str = ""


class DataIngestionCategorizerAgent:
    """
    AGENT A — Data Ingestion & SEBI Categorizer.
    Single input (MF name OR ISIN) -> resolved dossier with auto-detected
    category. Detection order: registry label -> holdings-based inference
    (equity share, cap-band distribution, passive tracking signature). Any
    parameters it cannot infer are surfaced in `missing_params` for the
    orchestrator to prompt on (never silently guessed).
    """

    def __init__(self, store, validator: ValidationOrchestrator) -> None:
        self.store = store          # unified store API (mock shim OR LiveDataStore)
        self.v = validator
        self.log = logging.getLogger("MFOrchestrator.AgentA")

    def _infer_category(self, rec: Dict[str, Any], holdings: pd.DataFrame) -> str:
        if rec["category"] in SEBI_2026_RULES:
            return rec["category"]
        if holdings is None or holdings.empty:
            # No holdings disclosure to infer from (real-data mode with no manual
            # CSV drop-in). Trust the AMFI/manifest-sourced category label as-is
            # rather than guessing from data we don't have.
            return rec["category"]
        eq = holdings[holdings["asset_type"] == "equity"]
        eq_w = eq["weight"].sum()
        if eq_w < 0.30:
            return "Corporate Bond"
        band_w = eq.groupby("cap_band")["weight"].sum()
        top_sector = eq.groupby("sector")["weight"].sum()
        if top_sector.max() >= 0.80:
            return "Sectoral"
        if band_w.get("large", 0) >= 0.80:
            return "Large Cap"
        if band_w.get("small", 0) >= 0.65:
            return "Small Cap"
        if band_w.get("mid", 0) >= 0.65:
            return "Mid Cap"
        return "Flexi Cap" if eq_w >= 0.65 else "Aggressive Hybrid"

    def run(self, query: str) -> FundDossier:
        name = self.store.resolve(query)
        if name is None:
            raise LookupError(f"Fund not resolvable from query {query!r}")
        rec = self.store.fund(name)
        snaps = self.store.snapshots(name)
        current = snaps["current"]["holdings"]
        holdings_available = bool(snaps["current"].get("available", True))
        category = self._infer_category(rec, current)
        rule = SEBI_2026_RULES.get(category)
        bucket = rule.bucket if rule is not None else SEBIBucket.EQUITY
        missing = []
        if rule is None:
            missing.append(f"category_not_in_sebi_2026_registry:{category}")
        if not holdings_available:
            missing.append("holdings_unavailable:category_from_amfi_not_holdings_inferred")
        if bucket == SEBIBucket.OTHER_PASSIVE and "TRI" not in (rec.get("benchmark") or ""):
            missing.append("tri_benchmark_name")
        if category in ("Sectoral", "Thematic") and not rec.get("declared_sector"):
            missing.append("declared_sector_or_theme")
        if bucket == SEBIBucket.DEBT:
            missing.append("portfolio_ytm_and_duration_feed")
        dossier = FundDossier(
            scheme_name=name, isin=rec["isin"], category=category, bucket=bucket,
            amc=rec["amc"], manager=rec["manager"], benchmark=rec["benchmark"],
            declared_sector=rec.get("declared_sector"), expense_ratio=rec["expense_ratio"],
            aum_cr=rec["aum_cr"], nav=rec["nav"], snapshots=snaps,
            current_holdings=current,
            daily_equity_weights=self.store.daily_weights_last_quarter(name),
            missing_params=missing, holdings_available=holdings_available,
            amfi_code=rec.get("amfi_code", ""))
        self.v._record("agentA", "dossier_built", True,
                       f"{name} -> {category} ({bucket.value}); missing={missing}")
        self.log.info("Resolved %r -> %s | category=%s | missing params=%s",
                      query, name, category, missing or "none")
        return dossier


class HistoricalBacktesterAgent:
    """
    AGENT B — Historical Backtester & Stock Auditor (the core engine).

    For the t-3y and t-5y disclosed portfolios:
      1. Extract the reported TOP-10 holdings (the manager's revealed bets).
      2. Compute each stock's REALISED CAGR and max drawdown from disclosure
         date to today.
      3. Build the passive counterfactual: the same weights applied to each
         stock's OWN SECTOR index (i.e., what a sector-tracking passive clone
         of those exact bets would have returned).
      4. Manager Alpha Consistency Score (0–100):
             45% hit-rate (share of picks beating their sector index)
           + 35% weighted excess CAGR, scaled on a ±6% p.a. band
           + 20% persistence (positive excess in BOTH the 5y and 3y windows)
    """

    def __init__(self, store, validator: ValidationOrchestrator) -> None:
        self.store = store
        self.quant = QuantEngine(validator)
        self.v = validator
        self.log = logging.getLogger("MFOrchestrator.AgentB")

    def _stock_audit(self, holdings: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
        top10 = holdings[holdings["asset_type"] == "equity"].nlargest(10, "weight")
        px = self.store.stock_prices(list(top10.index), start)
        if px is None or px.empty:
            return pd.DataFrame()
        rows = []
        for t, h in top10.iterrows():
            if t not in px.columns or px[t].dropna().empty:
                continue
            s = px[t].dropna()
            # passive counterfactual: real sector index, else equal-weight basket
            # of this fund's OWN holdings in that sector
            peers = [x for x in top10.index[top10["sector"] == h["sector"]] if x in px.columns]
            basket = px[peers] if len(peers) > 1 else None
            sec = self.store.sector_index(h["sector"], start, basket)
            if sec is None or sec.dropna().empty:
                continue
            rows.append(dict(
                ticker=t, name=h["name"], sector=h["sector"], weight=h["weight"],
                stock_cagr=self.quant.cagr(s),
                stock_max_dd=self.quant.max_drawdown(s),
                sector_cagr=self.quant.cagr(sec),
                excess_cagr=self.quant.cagr(s) - self.quant.cagr(sec)))
        audit = pd.DataFrame(rows)
        if not audit.empty:
            audit["beat_sector"] = audit["excess_cagr"] > 0
        return audit

    def run(self, dossier: FundDossier) -> Dict[str, Any]:
        if not dossier.holdings_available:
            # No manual holdings-disclosure CSV on file: there is no real top-10
            # portfolio to audit at t-5y/t-3y. Report the score as unavailable
            # rather than computing it off an empty/zero portfolio.
            self.v._record("agentB", "backtest_complete", False,
                           f"{dossier.scheme_name}: holdings unavailable, MACS not computed")
            return dict(
                windows={}, manager_alpha_consistency_score=None,
                interpretation="NOT AVAILABLE — no historical portfolio holdings disclosure "
                               "on file (mf_cache/disclosures/<code>_<YYYY-MM>.csv missing); "
                               "this score requires real top-10 holdings and is never "
                               "computed from synthetic data.")
        results: Dict[str, Any] = {"windows": {}}
        window_excess: Dict[str, float] = {}
        for label in ("t-5y", "t-3y"):
            snap = dossier.snapshots[label]
            audit = self._stock_audit(snap["holdings"], snap["as_of"])
            if audit.empty:
                continue
            w = audit["weight"].to_numpy()
            weighted_excess = float(np.dot(w, audit["excess_cagr"]) / w.sum())
            window_excess[label] = weighted_excess
            results["windows"][label] = dict(
                as_of=str(snap["as_of"].date()),
                hit_rate=float(audit["beat_sector"].mean()),
                weighted_excess_cagr=weighted_excess,
                avg_stock_max_dd=float(np.dot(w, audit["stock_max_dd"]) / w.sum()),
                audit_table=audit.round(4))
        if not results["windows"]:
            # Holdings were on file but neither window could be audited (e.g. no
            # stock-price history for these tickers) — report unavailable rather
            # than let an empty-slice mean silently become NaN.
            self.v._record("agentB", "backtest_complete", False,
                           f"{dossier.scheme_name}: no window could be audited "
                           f"(stock price history unavailable), MACS not computed")
            results["manager_alpha_consistency_score"] = None
            results["interpretation"] = (
                "NOT AVAILABLE — holdings were on file but no window could be audited "
                "(stock price history unavailable for these tickers); this score is "
                "never fabricated.")
            return results
        hit = float(np.mean([results["windows"][k]["hit_rate"] for k in results["windows"]]))
        mean_excess = float(np.mean(list(window_excess.values()))) if window_excess else 0.0
        excess_component = float(np.clip((mean_excess + 0.06) / 0.12, 0, 1))
        persistence = 1.0 if window_excess and all(v > 0 for v in window_excess.values()) else \
                      (0.5 if any(v > 0 for v in window_excess.values()) else 0.0)
        score = round(100 * (0.45 * hit + 0.35 * excess_component + 0.20 * persistence), 1)
        results["manager_alpha_consistency_score"] = score
        results["interpretation"] = (
            "Manager's revealed stock picks genuinely beat passive sector-tracking clones"
            if score >= 60 else
            "Stock selection roughly matches a passive sector clone — alpha not evidenced"
            if score >= 40 else
            "Historical picks UNDERPERFORMED their own sectors — negative selection skill")
        self.v._record("agentB", "backtest_complete", True,
                       f"{dossier.scheme_name} MACS={score}")
        return results


class ProfileRiskScorerAgent:
    """
    AGENT C — Profile & Risk Scorer.
    Builds the DYNAMIC UTILITY MATRIX from the runtime InvestorProfile and
    scores the fund on it. Weight surface reacts to every profile field, so
    two runs with different runtime inputs rank the same fund differently.
    """

    def __init__(self, validator: ValidationOrchestrator) -> None:
        self.quant = QuantEngine(validator)
        self.v = validator
        self.log = logging.getLogger("MFOrchestrator.AgentC")

    def utility_weights(self, p: InvestorProfile) -> Dict[str, float]:
        # Base surface is calibrated to the HIGH risk appetite (conservative/
        # moderate subtract growth below); drift_thematic stays at 0.10 in every
        # branch because the profile mandates flagging category drift regardless.
        w = dict(growth=0.40, downside=0.15, consistency=0.20, cost=0.10,
                 liquidity=0.05, drift_thematic=0.10)
        if p.risk_appetite == RiskAppetite.CONSERVATIVE:
            w["growth"] -= 0.15; w["downside"] += 0.15
        elif p.risk_appetite == RiskAppetite.MODERATE:
            w["growth"] -= 0.07; w["downside"] += 0.07
        if p.horizon_years < 4:
            w["downside"] += 0.08; w["liquidity"] += 0.04; w["growth"] -= 0.12
        elif p.horizon_years >= 7:
            # was >= 8, which excluded the baseline profile's own 7.5y default
            # despite its declared 7-8y target band — long-horizon growth tilt
            # must cover the band it was written for
            w["growth"] += 0.05; w["downside"] -= 0.05
        if p.liquidity_need == LiquidityNeed.HIGH:
            # exit-sensitive: liquidity AND concentrated-thematic exposure both matter more
            w["liquidity"] += 0.12; w["drift_thematic"] += 0.04; w["growth"] -= 0.16
        elif p.liquidity_need == LiquidityNeed.PARTIAL:
            w["liquidity"] += 0.05; w["growth"] -= 0.05
        elif p.liquidity_need == LiquidityNeed.NONE:
            # locked-in capital: exit friction is not a selection criterion until
            # horizon end — keep a token residual for the terminal liquidation
            w["liquidity"] -= 0.04; w["growth"] += 0.04
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def run(self, dossier: FundDossier, profile: InvestorProfile,
            benchmark: pd.Series) -> Dict[str, Any]:
        nav = dossier.nav
        rets = self.quant.daily_returns(nav)
        brets = self.quant.daily_returns(benchmark)
        rr3 = self.quant.rolling_returns(nav, 3)
        eq = dossier.current_holdings[dossier.current_holdings["asset_type"] == "equity"]
        top_sector_w = float(eq.groupby("sector")["weight"].sum().max()) if len(eq) else 0.0

        # --- raw factor levels -> 0..1 sub-scores (bounded, monotone) ----------
        sd = ValidationOrchestrator.safe_divide
        cagr_full = self.quant.cagr(nav)
        growth = float(np.clip((cagr_full - 0.06) / 0.14, 0, 1))
        sortino = self.quant.sortino(rets)
        downside = float(np.clip((sortino + 0.5) / 2.5, 0, 1))
        consistency = float(np.clip(1.0 - (rr3.lt(0).mean() if len(rr3) else 0.5), 0, 1))
        # TER has no free real-data source (AMFI/mfapi don't publish it): held
        # neutral rather than let a NaN silently poison the whole utility score.
        cost = (float(np.clip(1.0 - dossier.expense_ratio / 0.0225, 0, 1))
                if np.isfinite(dossier.expense_ratio) else 0.5)
        # exit friction proxy: small/sectoral exposure raises days-to-liquidate
        small_w = float(eq.loc[eq["cap_band"] == "small", "weight"].sum()) if len(eq) else 0.0
        liquidity = float(np.clip(1.0 - 0.9 * small_w - 0.3 * max(top_sector_w - 0.4, 0), 0, 1))
        drift_pen = 0.0
        flags: List[str] = []
        if dossier.category in ("Sectoral", "Thematic") and top_sector_w > 0.75:
            drift_pen += 0.6
            flags.append(f"UNHEDGED THEMATIC CONCENTRATION: {top_sector_w:.0%} in "
                         f"{dossier.declared_sector} with no derivative hedge disclosed")
        beta, _ = self.quant.beta_alpha(rets, brets)
        if dossier.bucket == SEBIBucket.EQUITY and np.isfinite(beta) and abs(beta - 1) > 0.35:
            drift_pen += 0.4
            flags.append(f"CATEGORY DRIFT SIGNATURE: beta {beta:.2f} far from category norm")
        drift_thematic = float(np.clip(1.0 - drift_pen, 0, 1))

        # --- raw fact LEVELS for honest surfacing (not the bounded sub-scores) --
        # These are what the app/report shows as ground truth (CAGR 14%, maxDD -24%);
        # the 0..1 sub-scores above are only the screening transform of them.
        vol_1y = float(rets.tail(252).std() * np.sqrt(252)) if len(rets) >= 30 else float("nan")
        nav3 = nav.tail(756)
        max_dd_3y = float((nav3 / nav3.cummax() - 1.0).min()) if len(nav3) else float("nan")
        common = (nav.index.intersection(benchmark.index)
                  if benchmark is not None and len(benchmark) else nav.index[:0])
        excess_cagr_3y = (float(cagr_full - self.quant.cagr(benchmark.loc[common]))
                          if len(common) > 252 else float("nan"))
        facts = dict(cagr=float(cagr_full), vol_1y=vol_1y, max_dd_3y=max_dd_3y,
                     sortino_3y=float(sortino), excess_cagr_3y=excess_cagr_3y)
        # Holdings-structure facts (a disclosure snapshot, NOT NAV): surfaced as
        # ground-truth levels. Coverage-gated to NaN when no disclosure is on file so
        # the concentration metric greys out downstream — missing holdings can never
        # colour green (honesty invariant). These are facts + a transparent screen
        # metric only; they NEVER enter the cohort_q1 predictor (unbacktestable).
        hf = mf_holdings.analyze(dossier.current_holdings, available=dossier.holdings_available)
        _na = float("nan")
        facts.update(
            top10_weight=hf.top10_weight if hf.top10_weight is not None else _na,
            effective_n=hf.effective_n if hf.effective_n is not None else _na,
            top_sector_weight=hf.top_sector_weight if hf.top_sector_weight is not None else _na,
            n_holdings=float(hf.n_equity) if hf.available and hf.n_equity else _na)

        weights = self.utility_weights(profile)
        subs = dict(growth=growth, downside=downside, consistency=consistency,
                    cost=cost, liquidity=liquidity, drift_thematic=drift_thematic)
        utility = round(100 * sum(weights[k] * subs[k] for k in weights), 1)
        self.v._record("agentC", "utility_scored", True,
                       f"{dossier.scheme_name} U={utility} weights={ {k: round(v,2) for k,v in weights.items()} }")
        return dict(utility_score=utility, sub_scores={k: round(v, 3) for k, v in subs.items()},
                    weight_matrix={k: round(v, 3) for k, v in weights.items()}, facts=facts,
                    risk_flags=flags, beta=None if not np.isfinite(beta) else round(beta, 2))


class NewsSentimentResearcherAgent:
    """
    AGENT D — News & Sentiment Web Researcher.
    Entities queried: the AMC, the named fund manager, and every sector holding
    > 15% of the portfolio. Sentiment is tone-weighted with recency decay
    (half-life 14 days); regulatory-action keywords raise red flags regardless
    of net tone.
    """

    REGULATORY_TERMS = ("sebi seeks", "show cause", "penalty", "ban", "probe",
                        "investigation", "front-running", "discontinue")

    def __init__(self, store, validator: ValidationOrchestrator) -> None:
        self.store = store
        self.v = validator
        self.log = logging.getLogger("MFOrchestrator.AgentD")

    def run(self, dossier: FundDossier) -> Dict[str, Any]:
        eq = dossier.current_holdings[dossier.current_holdings["asset_type"] == "equity"]
        conc_sectors = [s for s, w in eq.groupby("sector")["weight"].sum().items() if w > 0.15]
        entities = [dossier.amc, dossier.manager, *conc_sectors]
        items = self.store.news(entities)
        if not items:
            return dict(net_sentiment=0.0, n_items=0, red_flags=[], items=[],
                        entities_queried=entities)
        weights, tones, red_flags = [], [], []
        for it in items:
            decay = 0.5 ** (it["days_ago"] / 14.0)
            weights.append(decay); tones.append(it["tone"])
            # word-boundary match: naive substring made "ban" fire on "bank margins",
            # a false regulatory flag that now (post-reframe) would force a SELL.
            hl = it["headline"].lower()
            if any(re.search(rf"\b{re.escape(term)}\b", hl) for term in self.REGULATORY_TERMS):
                red_flags.append(it["headline"])
        net = float(np.dot(weights, tones) / np.sum(weights))
        self.v._record("agentD", "sentiment_researched", True,
                       f"{dossier.scheme_name} net={net:+.2f} items={len(items)} flags={len(red_flags)}")
        return dict(net_sentiment=round(net, 2), n_items=len(items),
                    red_flags=red_flags, items=items, entities_queried=entities)


# ==============================================================================
# 5. RECOMMENDATION ENGINE + MASTER ORCHESTRATOR
# ==============================================================================
class RecommendationEngine:
    """Fuses A/B/C/D outputs into an institutional Buy/Sell/Hold with rationale."""

    def __init__(self, validator: ValidationOrchestrator) -> None:
        self.v = validator

    def run(self, dossier: FundDossier, compliance: List[ComplianceFinding],
            backtest: Dict[str, Any], profile_score: Dict[str, Any],
            sentiment: Dict[str, Any], profile: InvestorProfile) -> Dict[str, Any]:
        # HONEST FRAMING (2026-07 reframe): on NAV-only data this is a QUANT +
        # COMPLIANCE SCREEN, not a skill-verified buy engine. We deliberately do
        # NOT fuse a composite or emit BUY/SELL/HOLD. A weighted blend of a
        # below-chance utility (measured AUC 0.463), an unverifiable manager score
        # and absent-check compliance would launder MISSING EVIDENCE into a verdict
        # — exactly the honesty failure this reframe removes. Instead surface: the
        # raw fact levels, the profile-weighted sub-scores (a screening AID, not a
        # recommendation), explicit data-coverage gaps, and let compliance /
        # manager / sentiment concerns travel as their own findings, un-blended.
        critical = [f for f in compliance if not f.passed and f.severity == "critical"]
        warnings = [f for f in compliance if not f.passed and f.severity == "warning"]
        utility = profile_score["utility_score"]
        facts = profile_score["facts"]
        is_passive = dossier.bucket == SEBIBucket.OTHER_PASSIVE
        macs = backtest.get("manager_alpha_consistency_score", None)

        # Data-coverage gaps: absence of evidence is stated as a gap, never scored
        # as a pass (this is where the old ~90 "compliance" inflation is removed —
        # unevaluable SEBI checks are a coverage gap, not a near-full score).
        coverage_flags: List[str] = []
        if not is_passive and macs is None:
            coverage_flags.append(
                "HOLDINGS NOT AVAILABLE — manager-skill audit and SEBI cap-fidelity / "
                "overlap checks NOT EVALUATED (absence of evidence, not a pass)")
        if not np.isfinite(dossier.expense_ratio):
            coverage_flags.append(
                "EXPENSE_RATIO_UNAVAILABLE — no free real-data source; cost sub-score held neutral")
        if not np.isfinite(facts.get("excess_cagr_3y", float("nan"))):
            coverage_flags.append(
                "BENCHMARK_EXCESS_UNAVAILABLE — no aligned benchmark series for excess-return")

        observations, flags = [], []
        if macs is not None and macs >= 60:
            observations.append(f"Manager Alpha Consistency Score {macs}/100 — historical top-10 "
                                f"picks beat passive sector clones in both audit windows")
        elif macs is not None and macs < 40:
            flags.append(f"Manager Alpha Consistency Score {macs}/100 — revealed picks "
                         f"underperformed their own sectors; passive alternative superior")
        if np.isfinite(dossier.expense_ratio) and profile_score["sub_scores"]["cost"] >= 0.7:
            observations.append(f"Low cost structure (TER {dossier.expense_ratio:.2%}) compounds "
                                f"over the {profile.horizon_years:.0f}y horizon")
        elif np.isfinite(dossier.expense_ratio):
            flags.append(f"Elevated TER {dossier.expense_ratio:.2%} drags long-horizon compounding")
        if profile_score["sub_scores"]["growth"] >= 0.6:
            observations.append("Realised growth comfortably inflation-beating for the profile's "
                                "wealth-maximisation objective")
        for fl in profile_score["risk_flags"]:
            flags.append(fl)
        for f in critical:
            flags.append(f"SEBI Feb-2026 breach [{f.rule_id}]: {f.detail}")
        for f in warnings:
            flags.append(f"SEBI Feb-2026 warning [{f.rule_id}]: {f.detail}")
        if sentiment["net_sentiment"] > 0.3:
            observations.append(f"Positive news tone ({sentiment['net_sentiment']:+.2f}) across "
                                f"{sentiment['n_items']} recent items")
        elif sentiment["net_sentiment"] < -0.3:
            flags.append(f"Negative news tone ({sentiment['net_sentiment']:+.2f}); "
                         f"headlines: {[i['headline'][:60] for i in sentiment['items'][:2]]}")
        for rf in sentiment["red_flags"]:
            flags.append(f"REGULATORY RED FLAG: {rf}")
        if not observations:
            observations.append("No standout strengths versus category peers at current data quality")

        # ---- metric-level GREEN / RED coloring + an EXPLAINABLE tri-colour verdict.
        # This is NOT the old below-chance weighted composite as the verdict: each
        # metric is coloured by its own honest direction, and the overall signal is
        # a transparent RULE over those visible colours + hard compliance gates. The
        # weighted screen score rides along as a SUPPORTING datapoint (one banded
        # input), never the sole determinant. HOLD is the honest default under thin
        # evidence, so missing data lands on blue, not a fabricated green.
        def _band(x, good, bad, higher_better=True):
            if not (isinstance(x, (int, float)) and np.isfinite(x)):
                return "grey"
            if higher_better:
                return "green" if x >= good else ("red" if x <= bad else "amber")
            return "green" if x <= good else ("red" if x >= bad else "amber")

        metric_colors = {
            "cagr": _band(facts["cagr"], 0.10, 0.06),
            "excess_cagr_3y": _band(facts["excess_cagr_3y"], 0.01, -0.01),
            "sortino_3y": _band(facts["sortino_3y"], 0.75, 0.0),
            "max_dd_3y": _band(facts["max_dd_3y"], -0.20, -0.35),   # negative; shallower is better
            "consistency": _band(profile_score["sub_scores"]["consistency"], 0.60, 0.40),
            "expense": ("grey" if not np.isfinite(dossier.expense_ratio)
                        else _band(dossier.expense_ratio, 0.010, 0.020, higher_better=False)),
            # Portfolio concentration (top-10 equity weight, LOWER better). A holdings
            # fact NAV cannot see; greys out when no disclosure is on file. Surfaced as
            # a visible screen metric — NOT folded into the greens/reds verdict count
            # (a high-conviction book is a legitimate style, not a defect); a genuine
            # SEBI single-issuer breach instead drives the verdict through `compliance`.
            "concentration": _band(facts.get("top10_weight", float("nan")),
                                   mf_holdings.TOP10_DIVERSIFIED,
                                   mf_holdings.TOP10_CONCENTRATED, higher_better=False),
        }
        metric_colors["compliance"] = ("red" if critical else
                                       "amber" if (warnings or (not is_passive and macs is None))
                                       else "green")
        # supporting weighted screen score (profile-weighted sub-scores) → a banded
        # datapoint, honestly labelled a screening aid (measured below chance alone).
        score_band = "green" if utility >= 65 else ("red" if utility < 45 else "blue")
        metric_colors["screen_score"] = score_band

        core = ["cagr", "excess_cagr_3y", "sortino_3y", "max_dd_3y", "consistency"]
        greens = sum(metric_colors[k] == "green" for k in core)
        reds = sum(metric_colors[k] == "red" for k in core)
        critical_breach = bool(critical) or bool(sentiment["red_flags"])
        if critical_breach or reds >= 3 or (reds > greens and score_band == "red"):
            verdict, verdict_color = "SELL / AVOID", "red"
        elif greens >= 3 and reds == 0 and score_band != "red":
            verdict, verdict_color = "BUY", "green"
        else:
            verdict, verdict_color = "HOLD / WATCH", "blue"
        # BUY on the quant screen alone (holdings never on free data) carries its
        # caveat visibly — the user chose "BUY allowed, caveat shown".
        verdict_caveat = ""
        if verdict == "BUY" and not is_passive and macs is None:
            verdict_caveat = ("screen-favourable on NAV facts only — manager skill & "
                              "holdings NOT verified; not a prediction of outperformance")

        # Confidence must reflect the DATA ACTUALLY ON DISK, not an adapter's
        # self-declared intent. A mocked run must never report high confidence.
        try:
            from mf_datasources import readiness
            rdy = readiness()
            live_share = float((rdy["status"] == "READY").mean())
            # Agent B's holdings feed is the load-bearing input; without it the
            # Manager Alpha score is synthetic and confidence is capped hard.
            holdings_ready = bool(
                (rdy.loc[rdy["source"] == "Historical disclosures", "status"] == "READY").any())
        except Exception:  # noqa: BLE001
            live_share, holdings_ready = 0.0, False
        confidence = round(float(np.clip(0.15 + 0.85 * live_share, 0, 1)), 2)
        if not holdings_ready:
            confidence = min(confidence, 0.45)
        self.v._record("recommendation", "issued", True,
                       f"{dossier.scheme_name}: {verdict} ({verdict_color}) "
                       f"screen={utility} greens={greens} reds={reds}")
        return dict(
            verdict=verdict, verdict_color=verdict_color, verdict_caveat=verdict_caveat,
            metric_colors=metric_colors,
            kind="screen-based recommendation (transparent rule over the metrics below; "
                 "NAV-only data, manager skill unverified)",
            profile_fit_score=utility,
            sub_scores=profile_score["sub_scores"],
            weight_matrix=profile_score["weight_matrix"],
            facts=facts,
            coverage_flags=coverage_flags,
            manager_alpha=(macs if macs is not None else
                           ("N/A (passive mandate)" if is_passive else
                            "NOT AVAILABLE (holdings missing)")),
            sentiment_net=round(sentiment["net_sentiment"], 2),
            observations=observations, flags=flags,
            confidence=confidence,
            confidence_note=(
                "LOW CONFIDENCE: no live data on disk — run bootstrap.py"
                if live_share == 0 else
                "CAPPED at 0.45: historical disclosures missing, so the Manager "
                "Alpha Consistency Score is NOT based on real holdings"
                if not holdings_ready else ""))


class MasterOrchestrator:
    """
    Fable-5 master agent. Plans the run, resolves the runtime investor profile,
    executes Agent A synchronously (its dossier gates everything), then spawns
    Agents B, C and D concurrently on dedicated worker pools, and finally fuses
    the results into the institutional recommendation.
    """

    def __init__(self, seed: int = 7, store=None, live: bool = False) -> None:
        self.validator = ValidationOrchestrator()
        if store is not None:
            self.store = store
        elif live:
            from mf_realstore import RealNAVStore  # lazy: avoids a module-load cycle
            self.store = RealNAVStore()
            LOGGER.info("LIVE mode: RealNAVStore wired (AMFI/mfapi/CapBand/yfinance/"
                        "Anthropic via mf_cache/). Run bootstrap.py first to warm the cache.")
        else:
            self.store = MockMarketDataStore(seed=seed)
        self.agent_a = DataIngestionCategorizerAgent(self.store, self.validator)
        self.agent_b = HistoricalBacktesterAgent(self.store, self.validator)
        self.agent_c = ProfileRiskScorerAgent(self.validator)
        self.agent_d = NewsSentimentResearcherAgent(self.store, self.validator)
        self.verifier = TrueToLabelVerifier(self.validator)
        self.recommender = RecommendationEngine(self.validator)
        self.sentinel = SentinelEngine(self.validator)
        # Cohort-signal (cohort_q1) resources are loaded lazily on first use, not
        # here: they need mf_cache/phase_b/model_artifact_cohort.json + the
        # universe manifest + a full NAV panel, none of which the MOCK demo
        # ships with. A missing artifact must degrade evaluate() to "signal
        # unavailable", never crash __init__ for a plain mock run.
        self._cohort_manifest = None
        self._cohort_nav_panel = None
        self._cohort_inferencer = None
        self._cohort_engine = None
        self._cohort_unavailable: Optional[str] = None
        # Whole-market amfi_code -> category map, built at most ONCE per orchestrator
        # (a batch run scores ~136+ funds through the same instance). Sentinel value
        # None = "not attempted yet"; {} = "attempted and unavailable", which the gate
        # treats as a safe degrade rather than retrying the master read per fund.
        self._universe_index: Optional[Dict[str, str]] = None
        self.log = logging.getLogger("MFOrchestrator.Master")

    def _cohort_universe_index(self) -> Dict[str, str]:
        """Lazy whole-market category index for mf_live_score's trained-universe gate.

        Without it the gate still refuses out-of-universe funds — it just reports the
        coarser NOT_IN_UNIVERSE instead of the permanent OUT_OF_TRAINING_UNIVERSE. So
        a failure here costs precision in the coverage flag, never safety.
        """
        if self._universe_index is None:
            try:
                from mf_live_score import load_universe_index
                self._universe_index = load_universe_index()
            except Exception as exc:  # noqa: BLE001 — never break a run over a flag's precision
                self.log.warning("Universe index unavailable: %s", exc)
                self._universe_index = {}
        return self._universe_index

    def _cohort_resources(self):
        """Lazy-load + instance-cache the cohort_q1 live-scoring resources
        (mf_live_score.py). Builds the FeatureEngine once (holds a strong
        reference for this orchestrator's lifetime — no separate id()-keyed
        global cache needed). Returns None (logged once) if the Phase-B
        artifact hasn't been built yet or any other resource fails to load —
        never raises out of evaluate()."""
        if self._cohort_unavailable is not None:
            return None
        if self._cohort_inferencer is None:
            try:
                from mf_features import FeatureEngine
                from mf_infer import CohortInferencer
                from mf_labels import load_manifest, load_nav_panel
                self._cohort_manifest = load_manifest()
                self._cohort_nav_panel = load_nav_panel(self._cohort_manifest)
                self._cohort_inferencer = CohortInferencer()
                self._cohort_engine = FeatureEngine(self._cohort_manifest, self._cohort_nav_panel)
            except Exception as exc:  # noqa: BLE001 — a weak supporting signal must
                # never take down the whole screen; degrade to "unavailable".
                self._cohort_unavailable = str(exc)
                self.log.warning("Cohort signal unavailable this run: %s", exc)
                return None
        return (self._cohort_manifest, self._cohort_nav_panel,
                self._cohort_inferencer, self._cohort_engine)

    @property
    def cohort_model_id(self) -> Optional[str]:
        """Public accessor for external callers (e.g. mf_artifact.py) that need
        to stamp which model artifact scored this run — None until at least one
        evaluate() call has resolved the cohort resources (or if unavailable)."""
        return self._cohort_inferencer.artifact["version"] if self._cohort_inferencer else None

    def cohort_live_resources(self):
        """Public wrapper around _cohort_resources() for external batch/
        monitoring code (e.g. mf_ledger.py's PSI computation via mf_artifact.py)
        that needs the same cached (manifest, nav_panel, inferencer, engine)
        evaluate() uses, without reaching into a private method. Returns None
        if the cohort resources are unavailable (same degrade as evaluate())."""
        return self._cohort_resources()

    def _sibling_weights(self, dossier: FundDossier) -> Dict[str, Dict[str, pd.Series]]:
        """Same-AMC equity schemes in overlap scope (large-cap peers exempt per SEBI)."""
        rule = SEBI_2026_RULES.get(dossier.category)
        if rule is None or rule.overlap_cap is None:
            return {}
        return self.store.sibling_equity_schemes(dossier.scheme_name, dossier.amc)

    def evaluate(self, query: str,
                 profile_config: Optional[Dict[str, Any]] = None,
                 argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        try:
            profile = resolve_investor_profile(profile_config, argv, interactive=True)
            dossier = self.agent_a.run(query)                      # gate: must succeed
            if dossier.missing_params:
                self.log.warning("Agent A surfaced missing params %s — proceeding with "
                                 "documented defaults; prompt the user in interactive mode",
                                 dossier.missing_params)
            bench = self.store.benchmark_series(dossier.benchmark)

            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="SUB_AGENT") as pool:
                fut_b = pool.submit(self.agent_b.run, dossier)
                fut_c = pool.submit(self.agent_c.run, dossier, profile, bench)
                fut_d = pool.submit(self.agent_d.run, dossier)
                compliance = self.verifier.verify(dossier, self._sibling_weights(dossier))
                backtest, profile_score, sentiment = fut_b.result(), fut_c.result(), fut_d.result()

            rec = self.recommender.run(dossier, compliance, backtest,
                                       profile_score, sentiment, profile)
            sentinel_report = self.sentinel.run(dossier, compliance, backtest, sentiment, bench)

            # cohort_q1 — the one out-of-sample-validated signal (holdout AUC
            # ~0.578, lift ~1.76x — weak). A supporting datapoint alongside the
            # screen, never folded into the verdict rule (that's a separate,
            # deliberate decision — see mf-architecture-decisions memory).
            # Isolated in its own try/except: a bug or transient failure in this
            # weak, optional signal must never discard an otherwise-complete
            # evaluation (dossier/compliance/backtest/recommendation/sentinel
            # are already computed above by the time this runs).
            cohort_signal = None
            try:
                if dossier.amfi_code:
                    res = self._cohort_resources()
                    if res is None:
                        rec["coverage_flags"].append(
                            "COHORT SIGNAL NOT EVALUATED — MODEL_ARTIFACT_UNAVAILABLE"
                            + (f": {self._cohort_unavailable}" if self._cohort_unavailable else ""))
                    else:
                        from mf_live_score import score_live
                        manifest, nav_panel, inferencer, engine = res
                        cohort_signal = score_live(dossier.amfi_code, manifest=manifest,
                                                   nav_panel=nav_panel, inferencer=inferencer,
                                                   engine=engine, today=TODAY,
                                                   universe_index=self._cohort_universe_index())
                        if cohort_signal["status"] != "OK":
                            rec["coverage_flags"].append(
                                f"COHORT SIGNAL NOT EVALUATED — {cohort_signal['status']}"
                                + (f": {cohort_signal['note']}" if cohort_signal["note"] else ""))
            except Exception as exc:  # noqa: BLE001 — see comment above
                self.log.warning("Cohort signal errored for %r: %s", dossier.scheme_name, exc)
                rec["coverage_flags"].append("COHORT SIGNAL NOT EVALUATED — unexpected error")

            self.validator.register_iteration(context=f"evaluate:{dossier.scheme_name}")
            return dict(profile=profile.model_dump(mode="json"), dossier=dossier, compliance=compliance,
                        backtest=backtest, profile_score=profile_score,
                        sentiment=sentiment, recommendation=rec, sentinel=sentinel_report,
                        cohort_signal=cohort_signal)
        except Exception as exc:  # noqa: BLE001 — institutional runs must degrade cleanly
            self.log.error("Evaluation failed for %r: %s\n%s", query, exc,
                           traceback_format(exc))
            return dict(error=str(exc), query=query)

    # ------------------------------ reporting --------------------------------
    @staticmethod
    def print_report(result: Dict[str, Any]) -> None:
        if "error" in result:
            print(f"\n!! EVALUATION FAILED: {result['error']}")
            return
        d: FundDossier = result["dossier"]
        r = result["recommendation"]
        p = result["profile"]
        print("\n" + "=" * 86)
        print(f"  {d.scheme_name}  |  {d.isin}  |  {d.category} ({d.bucket.value})")
        aum_str = f"₹{d.aum_cr:,.0f} cr" if np.isfinite(d.aum_cr) else "NOT AVAILABLE"
        ter_str = f"{d.expense_ratio:.2%}" if np.isfinite(d.expense_ratio) else "NOT AVAILABLE"
        print(f"  AMC: {d.amc}  |  Manager: {d.manager}  |  AUM {aum_str}  "
              f"|  TER {ter_str}")
        print("=" * 86)
        print(f"  Investor profile: {p['horizon_years']}y horizon | liquidity="
              f"{p['liquidity_need']} | risk={p['risk_appetite']}")
        def _pct(x, nd=2, signed=False):
            if not (isinstance(x, (int, float)) and np.isfinite(x)):
                return "n/a"
            return f"{x:+.{nd}%}" if signed else f"{x:.{nd}%}"
        dot = {"green": "🟢", "red": "🔴", "blue": "🔵", "amber": "🟡", "grey": "⚪"}
        mc = r["metric_colors"]
        print(f"\n  >>> RECOMMENDATION: {dot[r['verdict_color']]} {r['verdict']}"
              f"   (confidence {r['confidence']})")
        if r["verdict_caveat"]:
            print(f"      ⚠ {r['verdict_caveat']}")
        print(f"      basis: {r['kind']}")
        if r["confidence_note"]:
            print(f"      {r['confidence_note']}")
        f = r["facts"]
        print(f"      CAGR {dot[mc['cagr']]} {_pct(f['cagr'])} | "
              f"excess/bench(3y) {dot[mc['excess_cagr_3y']]} {_pct(f['excess_cagr_3y'], signed=True)} | "
              f"Sortino {dot[mc['sortino_3y']]} {f['sortino_3y']:.2f} | "
              f"maxDD(3y) {dot[mc['max_dd_3y']]} {_pct(f['max_dd_3y'], 1)} | "
              f"consistency {dot[mc['consistency']]} {r['sub_scores']['consistency']:.2f}")
        exp_str = _pct(d.expense_ratio) if np.isfinite(d.expense_ratio) else "n/a"
        en = f.get("effective_n", float("nan"))
        en_str = f"{en:.0f}" if isinstance(en, (int, float)) and np.isfinite(en) else "n/a"
        print(f"      expense {dot[mc['expense']]} {exp_str} | "
              f"compliance {dot[mc['compliance']]} | vol(1y) {_pct(f['vol_1y'])} | "
              f"concentration {dot[mc['concentration']]} top10={_pct(f.get('top10_weight', float('nan')), 1)} "
              f"(eff.N {en_str})")
        print(f"      Supporting screen score: {dot[mc['screen_score']]} "
              f"{r['profile_fit_score']}/100  (weighted screening aid — supporting datapoint, not the verdict)")
        print(f"      Manager Alpha: {r['manager_alpha']}")
        cs = result.get("cohort_signal")
        if cs is not None and cs["status"] == "OK":
            print(f"      Cohort signal ({cs['target']}): p={cs['probability']:.2f}, "
                  f"percentile={cs['cohort_percentile']:.0%} within {cs['cohort_n']}-fund cohort "
                  f"'{cs['cohort_key']}' @ {cs['anchor']} — {cs['note']}")
        for cf in r["coverage_flags"]:
            print(f"      [DATA GAP] {cf}")
        if d.bucket == SEBIBucket.OTHER_PASSIVE:
            print("\n  Manager Alpha Consistency Score: N/A — passive mandate; "
                  "evaluated on tracking fidelity, cost and compliance instead")
        else:
            mac = result["backtest"].get("manager_alpha_consistency_score")
            mac_str = f"{mac}/100 — " if mac is not None else ""
            print(f"\n  Manager Alpha Consistency Score: "
                  f"{mac_str}{result['backtest'].get('interpretation','')}")
        for label, w in result["backtest"].get("windows", {}).items():
            print(f"    [{label} @ {w['as_of']}] hit-rate {w['hit_rate']:.0%} | "
                  f"wtd excess CAGR {w['weighted_excess_cagr']:+.2%} | "
                  f"wtd stock max-DD {w['avg_stock_max_dd']:.1%}")
        print("\n  OBSERVATIONS:")
        for x in r["observations"]:
            print(f"    + {x}")
        print("  FLAGS / RISKS:")
        for x in r["flags"]:
            print(f"    - {x}")
        fails = [f for f in result["compliance"] if not f.passed]
        print(f"\n  SEBI Feb-2026 true-to-label: {len(result['compliance'])} checks, "
              f"{len(fails)} adverse")
        for f in fails:
            print(f"    [{f.severity.upper()}] {f.rule_id}: {f.detail}")
        s = result["sentiment"]
        print(f"\n  News sentiment: net {s['net_sentiment']:+.2f} over {s['n_items']} items "
              f"(entities: {', '.join(s['entities_queried'])})")
        sr = result["sentinel"]
        dot_sev = {"HIGH": "🔴", "WATCH": "🟡", "INFO": "⚪"}
        print(f"\n  SENTINEL ({sr.checks_run} checks run, {len(sr.checks_dormant)} dormant, "
              f"{len(sr.alerts)} alerts):")
        for a in sr.alerts:
            print(f"    {dot_sev[a.severity]} [{a.severity}] {a.code} ({a.basis}): {a.detail}")
        if sr.nfo_dossier:
            print(sr.nfo_dossier["rendered"])


def traceback_format(exc: BaseException) -> str:
    import traceback as tb
    return "".join(tb.format_exception(type(exc), exc, exc.__traceback__))


def print_user_action_checklist() -> None:
    print("\n" + "#" * 86)
    print("#  USER ACTION CHECKLIST")
    print("#" * 86)
    todo = [a for a in ADAPTER_REGISTRY if not a.is_live]
    done = [a for a in ADAPTER_REGISTRY if a.is_live]
    print("\n  ALREADY IMPLEMENTED — no action needed (free, no-auth, coded & tested):")
    for a in done:
        print(f"    [DONE] {a.name:<24} {a.endpoint}")
    print("\n  YOUR REMAINING ACTIONS:")
    for i, a in enumerate(todo, 1):
        print(f"    {i}. {a.name}  (used by {a.used_by})\n"
              f"       endpoint : {a.endpoint}\n"
              f"       requires : {a.required_config}")
    n = len(todo)
    print(f"    {n+1}. OPTIONAL — AMFI cap-band XLSX for exact SEBI 80%/65% fidelity checks:\n"
          "       download from amfiindia.com/otherdata/categorisation-of-stocks,\n"
          "       then  export AMFI_CAP_XLSX=/path/to/file.xlsx\n"
          "       (skip it and the pipeline falls back to market-cap cut-offs automatically)")
    print(f"    {n+2}. Run:  python3 bootstrap.py    (probes every source, caches, reports readiness)")


# ==============================================================================
# 6. END-TO-END DEMONSTRATION RUN
# ==============================================================================
def main() -> None:
    orch = MasterOrchestrator()

    print("\n############ RUN 1: baseline default profile | query by NAME ############")
    r1 = orch.evaluate("Bharat Small Cap Fund")
    MasterOrchestrator.print_report(r1)

    print("\n############ RUN 2: SAME thematic fund, TWO different runtime profiles ####")
    r2a = orch.evaluate("Bharat Banking & PSU Thematic Fund")   # default profile
    MasterOrchestrator.print_report(r2a)
    r2b = orch.evaluate("Bharat Banking & PSU Thematic Fund",
                        profile_config=dict(horizon_years=2.5,
                                            liquidity_need="high",
                                            risk_appetite="conservative"))
    MasterOrchestrator.print_report(r2b)

    print("\n############ RUN 3: query by ISIN | passive fund ############")
    r3 = orch.evaluate("INF002IX00019")
    MasterOrchestrator.print_report(r3)

    print("\n############ RUN 4: legacy Solution-Oriented scheme (discontinued) ########")
    r4 = orch.evaluate("Deccan Retirement Fund")
    MasterOrchestrator.print_report(r4)

    print(f"\nValidation harness: {len(orch.validator.records)} checks, "
          f"{orch.validator.failure_count} adverse findings recorded "
          f"(adverse = surveillance hits, not code errors)")
    print_user_action_checklist()




if __name__ == "__main__":
    main()
