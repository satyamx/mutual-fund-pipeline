"""
====================================================================================
INDIAN MUTUAL FUND UNIVERSE — INGESTION / QUANT / RISK / SCORING PIPELINE
====================================================================================
Production-grade, object-oriented pipeline covering 100% of the SEBI-categorised
Indian mutual fund universe: Equity, Debt, Hybrid, Solution-Oriented, Index/ETF.

ARCHITECTURE / SUB-AGENT ALLOCATION MAP
---------------------------------------
The pipeline is orchestrated by `PipelineOrchestrator`, which dispatches work to
three logically isolated, asynchronously executed "sub-agents" (thread pools):

  * SUB-AGENT A — IngestionEngine
      AMFI NAVAll.txt dump parsing, SEBI metadata schema-mapping, benchmark
      alignment, portfolio-disclosure normalisation.
  * SUB-AGENT B — QuantEngine
      Fully vectorised return mathematics: stepped rolling returns, alpha, beta,
      Sharpe, Sortino, Treynor, Information Ratio, capture ratios, tracking
      error/difference vs TRI benchmarks, active share, arbitrage spread velocity.
  * SUB-AGENT C — RiskStressTester
      Credit-quality blending (CRISIL/ICRA/CARE scale), corporate-group
      concentration surveillance (NBFC-stress style), duration shock simulation,
      equity liquidity / days-to-liquidate impact-cost modelling.

Episodic verification is provided by `ValidationOrchestrator`, a programmatic
self-verification harness that fires (a) every N=10 pipeline iterations and
(b) on every registered DataFrame mutation, checking math sanity, schema
conformance and data continuity against a strict baseline.

Dependencies: pandas>=2, numpy>=1.26, scipy>=1.11, pydantic>=2
====================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy import stats as scistats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
LOGGER = logging.getLogger("MFPipeline")

# ------------------------------------------------------------------------------
# GLOBAL MARKET CONSTANTS (Indian market conventions)
# ------------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252          # NSE/BSE trading calendar approximation
RISK_FREE_RATE_ANNUAL: float = 0.0675     # Proxy: ~91-day T-Bill / repo corridor
DEFAULT_ADV_PARTICIPATION: float = 0.20   # Max % of ADV consumable per day on exit
SEBI_SINGLE_ISSUER_LIMIT: float = 0.10    # 10% single-issuer debt exposure cap
SEBI_GROUP_EXPOSURE_LIMIT: float = 0.20   # 20% corporate-group cap (pre board approval)
EPISODIC_CHECK_INTERVAL: int = 10         # ValidationOrchestrator cadence


class SEBICategory(str, Enum):
    """Top-level SEBI scheme categorisation buckets."""
    EQUITY = "Equity"
    DEBT = "Debt"
    HYBRID = "Hybrid"
    SOLUTION_ORIENTED = "Solution Oriented"
    INDEX_ETF = "Index/ETF"


class EquitySubCategory(str, Enum):
    LARGE_CAP = "Large Cap"
    MID_CAP = "Mid Cap"
    SMALL_CAP = "Small Cap"
    FLEXI_CAP = "Flexi Cap"
    ELSS = "ELSS"
    FOCUSED = "Focused"
    SECTORAL_THEMATIC = "Sectoral/Thematic"


class DebtSubCategory(str, Enum):
    LIQUID = "Liquid"
    ULTRA_SHORT = "Ultra Short Duration"
    SHORT_DURATION = "Short Duration"
    CORPORATE_BOND = "Corporate Bond"
    CREDIT_RISK = "Credit Risk"
    BANKING_PSU = "Banking & PSU"
    GILT = "Gilt"
    DYNAMIC_BOND = "Dynamic Bond"


class HybridSubCategory(str, Enum):
    AGGRESSIVE = "Aggressive Hybrid"
    CONSERVATIVE = "Conservative Hybrid"
    BALANCED_ADVANTAGE = "Balanced Advantage"
    ARBITRAGE = "Arbitrage"
    EQUITY_SAVINGS = "Equity Savings"
    MULTI_ASSET = "Multi Asset"


class CreditRating(str, Enum):
    """Domestic long/short-term rating scale union (CRISIL / ICRA / CARE / IND-Ra)."""
    SOVEREIGN = "SOV"      # G-Secs, SDLs, T-Bills
    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB = "BBB"
    BELOW_BBB = "<BBB"
    A1_PLUS = "A1+"        # Short-term (CP/CD) top grade
    A1 = "A1"
    A2 = "A2"
    UNRATED = "UNRATED"
    CASH = "CASH"          # TREPS / repo / net current assets


# Internal numeric map — LOWER score == SAFER credit. Used by RiskStressTester.
CREDIT_RATING_SCORE: Dict[CreditRating, float] = {
    CreditRating.SOVEREIGN: 0.0,
    CreditRating.CASH:      0.0,
    CreditRating.AAA:       1.0,
    CreditRating.A1_PLUS:   1.0,
    CreditRating.AA_PLUS:   2.0,
    CreditRating.A1:        2.0,
    CreditRating.AA:        3.0,
    CreditRating.AA_MINUS:  4.0,
    CreditRating.A2:        4.0,
    CreditRating.A_PLUS:    5.0,
    CreditRating.A:         6.0,
    CreditRating.A_MINUS:   7.0,
    CreditRating.BBB:       8.0,
    CreditRating.BELOW_BBB: 10.0,
    CreditRating.UNRATED:   9.0,
}

# Recognised TRI benchmark identifiers (Index/ETF tracking must resolve to TRI).
TRI_BENCHMARKS: Tuple[str, ...] = (
    "NIFTY 50 TRI", "NIFTY NEXT 50 TRI", "NIFTY 100 TRI", "NIFTY MIDCAP 150 TRI",
    "NIFTY SMALLCAP 250 TRI", "NIFTY 500 TRI", "SENSEX TRI", "NIFTY BANK TRI",
)
ARBITRAGE_BENCHMARK: str = "NIFTY 50 ARBITRAGE INDEX"


# ==============================================================================
# MODULE 1 — DataModels (Pydantic schemas)
# ==============================================================================
class FundMetadata(BaseModel):
    """SEBI/AMFI master record for a single scheme (direct-growth plan basis)."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    amfi_code: str = Field(..., min_length=4, max_length=8, description="AMFI scheme code")
    isin: Optional[str] = Field(default=None, description="ISIN (growth option)")
    scheme_name: str = Field(..., min_length=3)
    amc_name: str = Field(..., min_length=2)
    category: SEBICategory
    sub_category: str = Field(..., min_length=2)
    benchmark: str = Field(..., min_length=3, description="Declared benchmark index")
    aum_cr: float = Field(..., gt=0, description="Assets under management, ₹ crore")
    expense_ratio: float = Field(..., ge=0.0, le=0.0275, description="TER (SEBI slab-capped)")
    inception_date: date

    @field_validator("amfi_code")
    @classmethod
    def _numeric_code(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError(f"AMFI code must be numeric, got {v!r}")
        return v

    @field_validator("isin")
    @classmethod
    def _isin_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.fullmatch(r"INF[0-9A-Z]{9}", v):
            raise ValueError(f"Malformed Indian MF ISIN: {v!r}")
        return v

    @model_validator(mode="after")
    def _benchmark_rules(self) -> "FundMetadata":
        # Indian nuance 1: passive vehicles must be evaluated vs Total Return variants.
        if self.category == SEBICategory.INDEX_ETF and self.benchmark.upper() not in TRI_BENCHMARKS:
            raise ValueError(
                f"Index/ETF scheme {self.scheme_name!r} must declare a TRI benchmark, "
                f"got {self.benchmark!r}. Allowed: {TRI_BENCHMARKS}"
            )
        # Indian nuance 2: Arbitrage funds must not hide behind debt benchmarks.
        if self.sub_category == HybridSubCategory.ARBITRAGE.value and \
                self.benchmark.upper() != ARBITRAGE_BENCHMARK:
            raise ValueError(
                f"Arbitrage scheme {self.scheme_name!r} must be benchmarked to "
                f"{ARBITRAGE_BENCHMARK!r}, got {self.benchmark!r}"
            )
        return self


class NAVSeries(BaseModel):
    """Validated NAV time-series payload for one scheme."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    amfi_code: str
    dates: List[date]
    navs: List[float]

    @model_validator(mode="after")
    def _integrity(self) -> "NAVSeries":
        if len(self.dates) != len(self.navs):
            raise ValueError("dates/navs length mismatch")
        if len(self.navs) < 30:
            raise ValueError(f"NAV history too short ({len(self.navs)} obs); need >= 30")
        arr = np.asarray(self.navs, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite NAV values detected")
        if np.any(arr <= 0):
            raise ValueError("Non-positive NAV values detected")
        if any(self.dates[i] >= self.dates[i + 1] for i in range(len(self.dates) - 1)):
            raise ValueError("NAV dates must be strictly increasing")
        # Fat-finger guard: single-day NAV jump > 25% flags a corrupt print.
        rel = np.abs(np.diff(arr) / arr[:-1])
        if np.any(rel > 0.25):
            raise ValueError("Suspect NAV discontinuity (>25% single-step move)")
        return self

    def to_series(self) -> pd.Series:
        s = pd.Series(self.navs, index=pd.DatetimeIndex(pd.to_datetime(self.dates)), name=self.amfi_code)
        s.index.name = "date"
        return s.astype(float)


class Holding(BaseModel):
    """One line item of a monthly portfolio disclosure."""
    model_config = ConfigDict(str_strip_whitespace=True)

    instrument: str = Field(..., min_length=2)
    weight: float = Field(..., gt=0, le=1.0, description="Fraction of NAV")
    asset_type: str = Field(..., description="equity|debt|derivative|cash")
    rating: CreditRating = Field(default=CreditRating.UNRATED)
    corporate_group: Optional[str] = Field(default=None, description="Promoter/conglomerate group")
    market_cap_cr: Optional[float] = Field(default=None, gt=0, description="Equity: mkt cap ₹cr")
    avg_daily_traded_value_cr: Optional[float] = Field(default=None, gt=0, description="Equity: ADV ₹cr")
    modified_duration: Optional[float] = Field(default=None, ge=0, le=40, description="Debt line MD")
    macaulay_duration: Optional[float] = Field(default=None, ge=0, le=40)
    ytm: Optional[float] = Field(default=None, ge=-0.02, le=0.30, description="Debt line YTM")
    is_hedged: bool = Field(default=False, description="Equity leg fully hedged via derivatives")

    @field_validator("asset_type")
    @classmethod
    def _asset_type_domain(cls, v: str) -> str:
        allowed = {"equity", "debt", "derivative", "cash"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"asset_type must be one of {allowed}, got {v!r}")
        return v


class PortfolioDisclosure(BaseModel):
    """Full monthly disclosure for one scheme."""
    amfi_code: str
    as_of: date
    holdings: List[Holding] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _weights_sum(self) -> "PortfolioDisclosure":
        total = sum(h.weight for h in self.holdings)
        if not (0.95 <= total <= 1.05):  # allow small cash-drag / derivative notional drift
            raise ValueError(f"Holding weights sum to {total:.4f}; expected ≈ 1.0")
        return self

    def to_frame(self) -> pd.DataFrame:
        rows = [h.model_dump() for h in self.holdings]
        df = pd.DataFrame(rows)
        df.insert(0, "amfi_code", self.amfi_code)
        df.insert(1, "as_of", pd.Timestamp(self.as_of))
        return df


# ==============================================================================
# MODULE 6 (defined early — dependency of all engines) — ValidationOrchestrator
# ==============================================================================
@dataclass
class ValidationRecord:
    when: datetime
    stage: str
    check: str
    passed: bool
    detail: str = ""


class ValidationOrchestrator:
    """
    Programmatic self-verification harness.

    Responsibilities
    ----------------
    1. Episodic verification: `register_iteration()` fires a full audit every
       `EPISODIC_CHECK_INTERVAL` (=10) pipeline iterations.
    2. Mutation verification: `guard_mutation()` snapshots a frame checksum
       before a mutation and audits continuity (row counts, date monotonicity,
       column schema) after it.
    3. Math safety: `safe_divide()` guarantees zero-division-free vectorised
       ratios everywhere in QuantEngine / ScoringEngine.
    4. Baseline schema enforcement for NAV panels and holdings tables.
    """

    NAV_PANEL_BASELINE: Dict[str, str] = {"index": "datetime64", "values": "float"}
    HOLDINGS_BASELINE: Tuple[str, ...] = (
        "amfi_code", "as_of", "instrument", "weight", "asset_type", "rating",
    )

    def __init__(self) -> None:
        self._records: List[ValidationRecord] = []
        self._iteration: int = 0
        self._lock = threading.Lock()
        self._watched_frames: Dict[str, pd.DataFrame] = {}
        self.log = logging.getLogger("MFPipeline.Validation")

    # -------------------------- core plumbing --------------------------------
    def _record(self, stage: str, check: str, passed: bool, detail: str = "") -> None:
        with self._lock:
            self._records.append(ValidationRecord(datetime.now(), stage, check, passed, detail))
        if not passed:
            self.log.warning("VALIDATION FAIL | %s | %s | %s", stage, check, detail)

    @property
    def records(self) -> List[ValidationRecord]:
        return list(self._records)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self._records if not r.passed)

    # -------------------------- math safety ----------------------------------
    @staticmethod
    def safe_divide(numer: Any, denom: Any, fill: float = np.nan) -> Any:
        """Vectorised division with zero/NaN denominators mapped to `fill`."""
        numer_arr = np.asarray(numer, dtype=float)
        denom_arr = np.asarray(denom, dtype=float)
        out = np.full(np.broadcast(numer_arr, denom_arr).shape, fill, dtype=float)
        mask = np.isfinite(denom_arr) & (np.abs(denom_arr) > 1e-12)
        np.divide(numer_arr, denom_arr, out=out, where=mask)
        if np.isscalar(numer) and np.isscalar(denom):
            return float(out)
        return out

    # ----------------------- frame-level audits ------------------------------
    @staticmethod
    def _checksum(df: pd.DataFrame) -> str:
        h = hashlib.sha256()
        h.update(str(df.shape).encode())
        h.update(",".join(map(str, df.columns)).encode())
        if len(df):
            h.update(pd.util.hash_pandas_object(df.head(50), index=True).values.tobytes())
        return h.hexdigest()

    def audit_nav_panel(self, panel: pd.DataFrame, stage: str = "nav_panel") -> bool:
        ok = True
        if not isinstance(panel.index, pd.DatetimeIndex):
            self._record(stage, "index_is_datetime", False, str(type(panel.index))); ok = False
        else:
            self._record(stage, "index_is_datetime", True)
            mono = panel.index.is_monotonic_increasing
            self._record(stage, "dates_monotonic", bool(mono))
            ok &= bool(mono)
        vals = panel.to_numpy(dtype=float, na_value=np.nan)
        finite_or_nan = np.all(np.isnan(vals) | np.isfinite(vals))
        self._record(stage, "no_inf_values", bool(finite_or_nan)); ok &= bool(finite_or_nan)
        positive = bool(np.nanmin(vals) > 0) if vals.size else True
        self._record(stage, "navs_strictly_positive", positive); ok &= positive
        dead_cols = int(panel.isna().all().sum())
        self._record(stage, "no_all_nan_columns", dead_cols == 0, f"{dead_cols} dead columns")
        ok &= dead_cols == 0
        return ok

    def audit_holdings(self, holdings: pd.DataFrame, stage: str = "holdings") -> bool:
        missing = [c for c in self.HOLDINGS_BASELINE if c not in holdings.columns]
        self._record(stage, "baseline_schema", not missing, f"missing={missing}")
        wsum = holdings.groupby("amfi_code")["weight"].sum()
        in_band = bool(((wsum >= 0.95) & (wsum <= 1.05)).all())
        self._record(stage, "weights_sum_band", in_band, f"range=[{wsum.min():.3f},{wsum.max():.3f}]")
        return (not missing) and in_band

    def audit_benchmark_alignment(self, fund_returns: pd.Series, bench_returns: pd.Series,
                                  stage: str = "alignment") -> bool:
        joined = pd.concat([fund_returns, bench_returns], axis=1, join="inner").dropna()
        coverage = self.safe_divide(len(joined), max(len(fund_returns.dropna()), 1), fill=0.0)
        ok = bool(coverage >= 0.90)
        self._record(stage, "benchmark_date_coverage>=90%", ok, f"coverage={coverage:.2%}")
        return ok

    # ---------------------- episodic verification ----------------------------
    def watch(self, name: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._watched_frames[name] = df

    def register_iteration(self, context: str = "") -> None:
        """Call once per fund processed. Every 10th call triggers a full audit."""
        with self._lock:
            self._iteration += 1
            fire = (self._iteration % EPISODIC_CHECK_INTERVAL == 0)
        if fire:
            self.log.info("Episodic verification fired at iteration %d (%s)", self._iteration, context)
            for name, df in list(self._watched_frames.items()):
                if isinstance(df.index, pd.DatetimeIndex):
                    self.audit_nav_panel(df, stage=f"episodic:{name}")
                elif "weight" in df.columns:
                    self.audit_holdings(df, stage=f"episodic:{name}")

    # ---------------------- mutation verification ----------------------------
    def guard_mutation(self, name: str, before: pd.DataFrame,
                       mutate: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """Run a mutation with pre/post continuity checks."""
        pre_hash = self._checksum(before)
        pre_rows, pre_cols = before.shape
        after = mutate(before.copy())
        post_hash = self._checksum(after)
        changed = pre_hash != post_hash
        self._record(f"mutation:{name}", "mutation_registered", True,
                     f"changed={changed} rows {pre_rows}->{after.shape[0]}")
        if isinstance(after.index, pd.DatetimeIndex):
            self._record(f"mutation:{name}", "post_mutation_dates_monotonic",
                         bool(after.index.is_monotonic_increasing))
        row_ratio = self.safe_divide(after.shape[0], max(pre_rows, 1), fill=0.0)
        self._record(f"mutation:{name}", "row_continuity>=50%", bool(row_ratio >= 0.5),
                     f"ratio={row_ratio:.2f}")
        return after

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame([r.__dict__ for r in self._records])
        if df.empty:
            return df
        return df.groupby(["stage", "check"], as_index=False)["passed"].agg(["count", "sum"])


# ==============================================================================
# MODULE 2 — IngestionEngine        [SUB-AGENT A: Ingestion & Schema Mapping]
# ==============================================================================
class IngestionEngine:
    """
    SUB-AGENT A. Converts raw feeds into clean, uniform DataFrames:

      * AMFI `NAVAll.txt` daily dumps  -> long NAV table -> wide NAV panel
      * SEBI metadata records          -> validated FundMetadata registry
      * Benchmark index level streams  -> aligned benchmark panel
      * Monthly portfolio disclosures  -> normalised holdings table

    All output frames use a business-day DatetimeIndex, forward-filled up to a
    bounded staleness limit (Indian exchange holidays), and are registered with
    the ValidationOrchestrator for episodic auditing.
    """

    AMFI_LINE = re.compile(
        r"^(?P<code>\d{4,8});(?P<isin1>[^;]*);(?P<isin2>[^;]*);(?P<name>[^;]+);"
        r"(?P<nav>[0-9.]+|N\.A\.);(?P<date>\d{2}-[A-Za-z]{3}-\d{4})\s*$"
    )
    MAX_FFILL_DAYS = 5  # never carry a stale NAV across more than a long weekend

    def __init__(self, validator: ValidationOrchestrator) -> None:
        self.validator = validator
        self.metadata: Dict[str, FundMetadata] = {}
        self.log = logging.getLogger("MFPipeline.SubAgentA")

    # ----------------------- SEBI metadata mapping ---------------------------
    def register_metadata(self, records: Sequence[Dict[str, Any]]) -> Dict[str, FundMetadata]:
        ok, failed = 0, 0
        for rec in records:
            try:
                meta = FundMetadata(**rec)
                self.metadata[meta.amfi_code] = meta
                ok += 1
            except Exception as exc:  # noqa: BLE001 - quarantine bad master rows
                failed += 1
                self.log.warning("Metadata rejected (%s): %s", rec.get("amfi_code", "?"), exc)
        self.validator._record("ingest:metadata", "records_validated", failed == 0,
                               f"ok={ok} failed={failed}")
        return self.metadata

    # ----------------------- AMFI text-dump parsing --------------------------
    def parse_amfi_dump(self, raw_text: str) -> pd.DataFrame:
        """Parse an AMFI NAVAll.txt payload into a long frame [amfi_code, date, nav]."""
        rows: List[Tuple[str, pd.Timestamp, float]] = []
        for line in raw_text.splitlines():
            m = self.AMFI_LINE.match(line.strip())
            if not m or m.group("nav") == "N.A.":
                continue  # category headers, AMC banners, blank lines, suspended NAVs
            rows.append((m.group("code"),
                         pd.to_datetime(m.group("date"), format="%d-%b-%Y"),
                         float(m.group("nav"))))
        df = pd.DataFrame(rows, columns=["amfi_code", "date", "nav"])
        self.validator._record("ingest:amfi_dump", "lines_parsed", len(df) > 0, f"rows={len(df)}")
        return df

    # ----------------------- NAV panel construction --------------------------
    def build_nav_panel(self, series_payloads: Sequence[NAVSeries]) -> pd.DataFrame:
        """Union of validated NAVSeries -> wide panel (date x amfi_code)."""
        cols = [p.to_series() for p in series_payloads]
        panel = pd.concat(cols, axis=1).sort_index()

        def _to_business_days(df: pd.DataFrame) -> pd.DataFrame:
            bdays = pd.bdate_range(df.index.min(), df.index.max())
            return df.reindex(bdays).ffill(limit=self.MAX_FFILL_DAYS)

        panel = self.validator.guard_mutation("nav_panel_bday_ffill", panel, _to_business_days)
        panel.index.name = "date"
        self.validator.audit_nav_panel(panel, stage="ingest:nav_panel")
        self.validator.watch("nav_panel", panel)
        return panel

    def build_benchmark_panel(self, benchmark_levels: Dict[str, pd.Series]) -> pd.DataFrame:
        frames = []
        for name, s in benchmark_levels.items():
            s = s.copy()
            s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
            s.name = name.upper()
            frames.append(s.sort_index())
        panel = pd.concat(frames, axis=1)
        bdays = pd.bdate_range(panel.index.min(), panel.index.max())
        panel = panel.reindex(bdays).ffill(limit=self.MAX_FFILL_DAYS)
        panel.index.name = "date"
        self.validator.audit_nav_panel(panel, stage="ingest:benchmark_panel")
        self.validator.watch("benchmark_panel", panel)
        return panel

    # ----------------------- holdings normalisation --------------------------
    def build_holdings_table(self, disclosures: Sequence[PortfolioDisclosure]) -> pd.DataFrame:
        frames = [d.to_frame() for d in disclosures]
        table = pd.concat(frames, ignore_index=True)
        table["rating"] = table["rating"].map(lambda r: r.value if isinstance(r, CreditRating) else str(r))
        self.validator.audit_holdings(table, stage="ingest:holdings")
        self.validator.watch("holdings", table)
        return table


# ==============================================================================
# MODULE 3 — QuantEngine            [SUB-AGENT B: Vectorised Calculation Engine]
# ==============================================================================
class QuantEngine:
    """
    SUB-AGENT B. All routines are vectorised over the full NAV panel; per-fund
    dispatch is embarrassingly parallel and is fanned out by the orchestrator's
    SUB_AGENT_B thread pool.
    """

    def __init__(self, validator: ValidationOrchestrator,
                 rf_annual: float = RISK_FREE_RATE_ANNUAL) -> None:
        self.validator = validator
        self.rf_annual = rf_annual
        self.rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        self.log = logging.getLogger("MFPipeline.SubAgentB")

    # ----------------------------- primitives --------------------------------
    @staticmethod
    def daily_returns(levels: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return levels.pct_change().iloc[1:]

    def cagr(self, levels: pd.Series) -> float:
        levels = levels.dropna()
        if len(levels) < 2:
            return np.nan
        years = (levels.index[-1] - levels.index[0]).days / 365.25
        if years <= 0:
            return np.nan
        gross = self.validator.safe_divide(levels.iloc[-1], levels.iloc[0])
        return float(gross ** (1.0 / years) - 1.0) if np.isfinite(gross) else np.nan

    def annualised_return(self, rets: pd.Series) -> float:
        rets = rets.dropna()
        if rets.empty:
            return np.nan
        gross = float(np.prod(1.0 + rets.to_numpy()))
        return gross ** (TRADING_DAYS_PER_YEAR / len(rets)) - 1.0

    @staticmethod
    def annualised_vol(rets: pd.Series) -> float:
        rets = rets.dropna()
        return float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if len(rets) > 2 else np.nan

    # ----------------------- rolling returns (stepped) -----------------------
    def rolling_returns(self, levels: pd.Series, window_years: int,
                        step: str = "daily") -> pd.Series:
        """
        Stepped rolling CAGR — NOT point-to-point. For every observation date t,
        computes annualised return over [t - window, t]. `step='weekly'`
        subsamples the daily-stepped series to Fridays.
        """
        window = window_years * TRADING_DAYS_PER_YEAR
        levels = levels.dropna()
        if len(levels) <= window:
            return pd.Series(dtype=float, name=f"roll_{window_years}y")
        gross = self.validator.safe_divide(levels.to_numpy()[window:], levels.to_numpy()[:-window])
        ann = np.power(gross, 1.0 / window_years, where=np.isfinite(gross),
                       out=np.full_like(gross, np.nan)) - 1.0
        out = pd.Series(ann, index=levels.index[window:], name=f"roll_{window_years}y")
        if step == "weekly":
            out = out[out.index.dayofweek == 4]  # Friday steps
        return out

    def rolling_return_stats(self, levels: pd.Series) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for yrs in (3, 5):
            rr = self.rolling_returns(levels, yrs, step="daily")
            if rr.empty:
                stats[f"rr{yrs}y_mean"] = np.nan
                stats[f"rr{yrs}y_p25"] = np.nan
                stats[f"rr{yrs}y_negative_share"] = np.nan
            else:
                stats[f"rr{yrs}y_mean"] = float(rr.mean())
                stats[f"rr{yrs}y_p25"] = float(rr.quantile(0.25))
                stats[f"rr{yrs}y_negative_share"] = float((rr < 0).mean())
        return stats

    # ------------------------- CAPM-style metrics ----------------------------
    def beta_alpha(self, fund_rets: pd.Series, bench_rets: pd.Series) -> Tuple[float, float]:
        df = pd.concat([fund_rets, bench_rets], axis=1, join="inner").dropna()
        if len(df) < 60:
            return np.nan, np.nan
        f, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
        var_b = np.var(b, ddof=1)
        beta = float(self.validator.safe_divide(np.cov(f, b, ddof=1)[0, 1], var_b))
        ann_f = self.annualised_return(df.iloc[:, 0])
        ann_b = self.annualised_return(df.iloc[:, 1])
        alpha = ann_f - (self.rf_annual + beta * (ann_b - self.rf_annual))  # Jensen's alpha
        return beta, float(alpha)

    def sharpe(self, fund_rets: pd.Series) -> float:
        excess = fund_rets.dropna() - self.rf_daily
        vol = excess.std(ddof=1)
        return float(self.validator.safe_divide(excess.mean(), vol) * math.sqrt(TRADING_DAYS_PER_YEAR))

    def sortino(self, fund_rets: pd.Series, mar_annual: Optional[float] = None) -> float:
        mar_daily = ((1 + (self.rf_annual if mar_annual is None else mar_annual))
                     ** (1 / TRADING_DAYS_PER_YEAR) - 1)
        excess = fund_rets.dropna() - mar_daily
        downside = np.minimum(excess.to_numpy(), 0.0)
        dd = math.sqrt(float(np.mean(np.square(downside))))
        return float(self.validator.safe_divide(excess.mean(), dd) * math.sqrt(TRADING_DAYS_PER_YEAR))

    def treynor(self, fund_rets: pd.Series, bench_rets: pd.Series) -> float:
        beta, _ = self.beta_alpha(fund_rets, bench_rets)
        ann = self.annualised_return(fund_rets)
        return float(self.validator.safe_divide(ann - self.rf_annual, beta))

    def tracking_error(self, fund_rets: pd.Series, bench_rets: pd.Series) -> float:
        df = pd.concat([fund_rets, bench_rets], axis=1, join="inner").dropna()
        if len(df) < 60:
            return np.nan
        active = df.iloc[:, 0] - df.iloc[:, 1]
        return float(active.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

    def information_ratio(self, fund_rets: pd.Series, bench_rets: pd.Series) -> float:
        te = self.tracking_error(fund_rets, bench_rets)
        df = pd.concat([fund_rets, bench_rets], axis=1, join="inner").dropna()
        if df.empty:
            return np.nan
        active_ann = self.annualised_return(df.iloc[:, 0]) - self.annualised_return(df.iloc[:, 1])
        return float(self.validator.safe_divide(active_ann, te))

    def tracking_difference_tri(self, fund_levels: pd.Series, tri_levels: pd.Series) -> float:
        """Indian nuance: annualised tracking difference vs the TRI variant."""
        joined = pd.concat([fund_levels, tri_levels], axis=1, join="inner").dropna()
        if len(joined) < TRADING_DAYS_PER_YEAR:
            return np.nan
        return self.cagr(joined.iloc[:, 0]) - self.cagr(joined.iloc[:, 1])

    def capture_ratios(self, fund_rets: pd.Series, bench_rets: pd.Series) -> Tuple[float, float]:
        df = pd.concat([fund_rets, bench_rets], axis=1, join="inner").dropna()
        f, b = df.iloc[:, 0], df.iloc[:, 1]

        def _compounded(x: pd.Series) -> float:
            return float(np.prod(1.0 + x.to_numpy()) ** (self.validator.safe_divide(1.0, max(len(x), 1))) - 1.0)

        up, down = b > 0, b < 0
        ucr = self.validator.safe_divide(_compounded(f[up]), _compounded(b[up])) if up.sum() > 10 else np.nan
        dcr = self.validator.safe_divide(_compounded(f[down]), _compounded(b[down])) if down.sum() > 10 else np.nan
        return float(ucr), float(dcr)

    # --------------------------- portfolio-level -----------------------------
    @staticmethod
    def active_share(fund_weights: pd.Series, bench_weights: pd.Series) -> float:
        aligned = pd.concat([fund_weights, bench_weights], axis=1).fillna(0.0)
        return float(0.5 * (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum())

    def arbitrage_spread_velocity(self, fund_levels: pd.Series,
                                  arb_index_levels: pd.Series,
                                  lookback: int = 63) -> Dict[str, float]:
        """
        Indian nuance: arbitrage funds live off cash–futures spreads. We measure
        the rolling 3-month annualised active spread vs the Nifty 50 Arbitrage
        Index and its *velocity* (slope of the spread over the last quarter) —
        decaying spreads flag roll-yield compression.
        """
        joined = pd.concat([fund_levels, arb_index_levels], axis=1, join="inner").dropna()
        rets = self.daily_returns(joined)
        active = (rets.iloc[:, 0] - rets.iloc[:, 1])
        roll_spread = active.rolling(lookback).mean() * TRADING_DAYS_PER_YEAR
        roll_spread = roll_spread.dropna()
        if len(roll_spread) < lookback:
            return {"arb_spread_ann": np.nan, "arb_spread_velocity": np.nan}
        recent = roll_spread.iloc[-lookback:]
        slope, *_ = scistats.linregress(np.arange(len(recent)), recent.to_numpy())
        return {"arb_spread_ann": float(roll_spread.iloc[-1]),
                "arb_spread_velocity": float(slope * TRADING_DAYS_PER_YEAR)}

    def max_drawdown(self, levels: pd.Series) -> float:
        levels = levels.dropna()
        running_max = levels.cummax()
        dd = self.validator.safe_divide(levels.to_numpy(), running_max.to_numpy()) - 1.0
        return float(np.nanmin(dd)) if dd.size else np.nan

    # ------------------------ full metric assembly ---------------------------
    def compute_fund_metrics(self, meta: FundMetadata, fund_levels: pd.Series,
                             bench_levels: pd.Series,
                             arb_index_levels: Optional[pd.Series] = None) -> Dict[str, float]:
        fund_rets = self.daily_returns(fund_levels.dropna())
        bench_rets = self.daily_returns(bench_levels.dropna())
        self.validator.audit_benchmark_alignment(fund_rets, bench_rets,
                                                 stage=f"quant:{meta.amfi_code}")
        beta, alpha = self.beta_alpha(fund_rets, bench_rets)
        ucr, dcr = self.capture_ratios(fund_rets, bench_rets)
        metrics: Dict[str, float] = {
            "cagr_full": self.cagr(fund_levels),
            "ann_vol": self.annualised_vol(fund_rets),
            "sharpe": self.sharpe(fund_rets),
            "sortino": self.sortino(fund_rets),
            "beta": beta,
            "alpha": alpha,
            "treynor": self.treynor(fund_rets, bench_rets),
            "info_ratio": self.information_ratio(fund_rets, bench_rets),
            "tracking_error": self.tracking_error(fund_rets, bench_rets),
            "upside_capture": ucr,
            "downside_capture": dcr,
            "max_drawdown": self.max_drawdown(fund_levels),
        }
        metrics.update(self.rolling_return_stats(fund_levels))
        if meta.category == SEBICategory.INDEX_ETF:
            metrics["tracking_difference_tri"] = self.tracking_difference_tri(fund_levels, bench_levels)
        if meta.sub_category == HybridSubCategory.ARBITRAGE.value and arb_index_levels is not None:
            metrics.update(self.arbitrage_spread_velocity(fund_levels, arb_index_levels))
        return metrics


# ==============================================================================
# MODULE 4 — RiskStressTester       [SUB-AGENT C: Risk & Stress Simulator]
# ==============================================================================
class RiskStressTester:
    """
    SUB-AGENT C. Portfolio-disclosure-driven risk analytics:

      Debt   : credit-quality blending on the CRISIL/ICRA/CARE numeric scale,
               single-issuer & corporate-group concentration surveillance
               (hidden NBFC-stress style exposures), MD/Macaulay/YTM roll-ups,
               and parallel yield-shock stress P&L.
      Equity : portfolio liquidity score — weighted average market cap and
               simulated days-to-liquidate under a bounded ADV participation.
      Hybrid : dynamic equity allocation, hedged vs unhedged split, and
               aggressive/conservative band classification.
    """

    YIELD_SHOCKS_BPS: Tuple[int, ...] = (50, 100, 200)

    def __init__(self, validator: ValidationOrchestrator,
                 issuer_limit: float = SEBI_SINGLE_ISSUER_LIMIT,
                 group_limit: float = SEBI_GROUP_EXPOSURE_LIMIT,
                 adv_participation: float = DEFAULT_ADV_PARTICIPATION) -> None:
        self.validator = validator
        self.issuer_limit = issuer_limit
        self.group_limit = group_limit
        self.adv_participation = adv_participation
        self.log = logging.getLogger("MFPipeline.SubAgentC")

    # ------------------------------ debt -------------------------------------
    def credit_risk_index(self, holdings: pd.DataFrame) -> float:
        """Exposure-weighted credit score on a 0 (sovereign) .. 10 (junk) scale."""
        debt = holdings[holdings["asset_type"].isin(["debt", "cash"])]
        if debt.empty:
            return np.nan
        scores = debt["rating"].map(lambda r: CREDIT_RATING_SCORE.get(CreditRating(r), 9.0))
        w = debt["weight"].to_numpy()
        return float(self.validator.safe_divide(np.dot(w, scores.to_numpy()), w.sum()))

    def duration_profile(self, holdings: pd.DataFrame) -> Dict[str, float]:
        debt = holdings[holdings["asset_type"] == "debt"].dropna(subset=["modified_duration"])
        if debt.empty:
            return {"modified_duration": np.nan, "macaulay_duration": np.nan, "ytm": np.nan}
        w = debt["weight"].to_numpy()
        wsum = w.sum()
        sd = self.validator.safe_divide
        return {
            "modified_duration": float(sd(np.dot(w, debt["modified_duration"].to_numpy()), wsum)),
            "macaulay_duration": float(sd(np.dot(w, debt["macaulay_duration"].fillna(
                debt["modified_duration"]).to_numpy()), wsum)),
            "ytm": float(sd(np.dot(w, debt["ytm"].fillna(0.0).to_numpy()), wsum)),
        }

    def yield_shock_pnl(self, portfolio_md: float, debt_weight: float) -> Dict[str, float]:
        """First-order parallel-shift stress: ΔNAV ≈ -w_debt · MD · Δy."""
        out = {}
        for bps in self.YIELD_SHOCKS_BPS:
            shock = bps / 10_000.0
            pnl = -debt_weight * (portfolio_md if np.isfinite(portfolio_md) else 0.0) * shock
            out[f"stress_pnl_{bps}bps"] = float(pnl)
        return out

    def concentration_flags(self, holdings: pd.DataFrame) -> Dict[str, Any]:
        """Single-issuer and corporate-group concentration surveillance."""
        debt = holdings[holdings["asset_type"] == "debt"]
        if debt.empty:
            return {"max_issuer_weight": 0.0, "max_group_weight": 0.0,
                    "issuer_breaches": 0, "group_breaches": 0, "top_group": None}
        issuer_w = debt.groupby("instrument")["weight"].sum()
        grp = debt.assign(corporate_group=debt["corporate_group"].fillna("STANDALONE"))
        group_w = grp[grp["corporate_group"] != "STANDALONE"] \
            .groupby("corporate_group")["weight"].sum()
        max_group = float(group_w.max()) if len(group_w) else 0.0
        return {
            "max_issuer_weight": float(issuer_w.max()),
            "max_group_weight": max_group,
            "issuer_breaches": int((issuer_w > self.issuer_limit).sum()),
            "group_breaches": int((group_w > self.group_limit).sum()) if len(group_w) else 0,
            "top_group": group_w.idxmax() if len(group_w) else None,
        }

    # ----------------------------- equity ------------------------------------
    def liquidity_profile(self, holdings: pd.DataFrame, aum_cr: float) -> Dict[str, float]:
        """
        Indian nuance for small/mid caps: days-to-liquidate each position when
        consuming at most `adv_participation` of its average daily traded value,
        exposure-weighted; plus weighted-average market cap.
        """
        eq = holdings[(holdings["asset_type"] == "equity")
                      & holdings["avg_daily_traded_value_cr"].notna()]
        if eq.empty:
            return {"wavg_market_cap_cr": np.nan, "days_to_liquidate_wavg": np.nan,
                    "days_to_liquidate_p90": np.nan}
        position_value = eq["weight"].to_numpy() * aum_cr
        sellable_per_day = self.adv_participation * eq["avg_daily_traded_value_cr"].to_numpy()
        dtl = self.validator.safe_divide(position_value, sellable_per_day)
        w = eq["weight"].to_numpy()
        sd = self.validator.safe_divide
        wavg_mcap = sd(np.dot(w, eq["market_cap_cr"].fillna(0.0).to_numpy()), w.sum())
        return {
            "wavg_market_cap_cr": float(wavg_mcap),
            "days_to_liquidate_wavg": float(sd(np.dot(w, dtl), w.sum())),
            "days_to_liquidate_p90": float(np.nanpercentile(dtl, 90)),
        }

    # ----------------------------- hybrid ------------------------------------
    def hybrid_profile(self, holdings: pd.DataFrame) -> Dict[str, Any]:
        eq_w = float(holdings.loc[holdings["asset_type"] == "equity", "weight"].sum())
        hedged_w = float(holdings.loc[(holdings["asset_type"] == "equity")
                                      & holdings["is_hedged"], "weight"].sum())
        unhedged = eq_w - hedged_w
        sd = self.validator.safe_divide
        if unhedged >= 0.65:
            band = "aggressive"
        elif unhedged >= 0.40:
            band = "balanced"
        elif unhedged >= 0.10:
            band = "conservative"
        else:
            band = "arbitrage/defensive"
        return {
            "equity_allocation": eq_w,
            "hedged_ratio": float(sd(hedged_w, eq_w, fill=0.0)),
            "unhedged_equity": unhedged,
            "hybrid_band": band,
        }

    # ------------------------ full risk assembly -----------------------------
    def run(self, meta: FundMetadata, holdings: pd.DataFrame) -> Dict[str, Any]:
        h = holdings[holdings["amfi_code"] == meta.amfi_code]
        out: Dict[str, Any] = {}
        debt_weight = float(h.loc[h["asset_type"] == "debt", "weight"].sum())
        if meta.category in (SEBICategory.DEBT, SEBICategory.HYBRID, SEBICategory.SOLUTION_ORIENTED):
            out["credit_risk_index"] = self.credit_risk_index(h)
            dur = self.duration_profile(h)
            out.update(dur)
            out.update(self.yield_shock_pnl(dur["modified_duration"], debt_weight))
            out.update(self.concentration_flags(h))
        if meta.category in (SEBICategory.EQUITY, SEBICategory.INDEX_ETF,
                             SEBICategory.HYBRID, SEBICategory.SOLUTION_ORIENTED):
            out.update(self.liquidity_profile(h, meta.aum_cr))
        if meta.category in (SEBICategory.HYBRID, SEBICategory.SOLUTION_ORIENTED):
            out.update(self.hybrid_profile(h))
        return out


# ==============================================================================
# MODULE 5 — ScoringEngine (multi-factor, SEBI-class-weighted composite)
# ==============================================================================
class ScoringEngine:
    """
    Reusable multi-factor ranking system. For each SEBI bucket, a metric matrix
    with direction and weight is defined; metrics are percentile-ranked within
    the peer group (so scores are peer-relative), direction-adjusted, weighted,
    and rescaled to a normalised 1–100 composite.

    direction = +1  -> higher is better (alpha, sortino, ...)
    direction = -1  -> lower is better (tracking error, credit risk, ...)
    """

    FACTOR_MAP: Dict[SEBICategory, Dict[str, Tuple[float, int]]] = {
        SEBICategory.EQUITY: {
            # Step-2 reweight (2026-07): the nine testable metrics carry the
            # pre-registered "minimal surgery" variant that cleared the purged-
            # CPCV no-regression guardrail (mf_factor_backtest.py) — legacy
            # proportions kept exactly, scaled x0.90 so the map sums to 1.0
            # after the config-argued untestable changes below. Honest caveat:
            # no pre-registered variant (this one included) measurably BEAT the
            # old weights' within-cohort rank AUC; all sit at ~0.46, i.e. the
            # fixed composite has no demonstrated within-cohort selection skill.
            "rr3y_mean":            (0.135, +1),
            "rr5y_mean":            (0.09,  +1),
            "sortino":              (0.135, +1),
            "info_ratio":           (0.135, +1),
            "alpha":                (0.09,  +1),
            "upside_capture":       (0.0675, +1),
            "downside_capture":     (0.0675, -1),
            "max_drawdown":         (0.045, +1),  # less-negative drawdown is better
            # Robust Phase-B model finding (coefficients.json cohort runs):
            # beta_3y -0.902/-0.960, sign+magnitude stable across both cohort
            # targets — lower systematic risk ranks better within cohort.
            "beta":                 (0.135, -1),
            # days_to_liquidate_wavg dropped (was 0.10): investor liquidity is
            # LOCKED for the 7.5y horizon — exit friction is not a selection
            # criterion. expense_ratio raised 0.05 -> 0.10: the one a-priori
            # causal forward-looking factor (cost drag is arithmetic and does
            # not mean-revert); a 1% TER gap compounds to ~7.5% over 7.5y.
            "expense_ratio":        (0.10, -1),
        },
        SEBICategory.INDEX_ETF: {
            "tracking_error":          (0.35, -1),
            "tracking_difference_tri": (0.30, +1),  # less negative TD vs TRI wins
            "expense_ratio":           (0.20, -1),
            "days_to_liquidate_wavg":  (0.15, -1),
        },
        SEBICategory.DEBT: {
            "credit_risk_index":  (0.25, -1),
            "ytm":                (0.15, +1),
            "sharpe":             (0.15, +1),
            "rr3y_mean":          (0.10, +1),
            "max_group_weight":   (0.10, -1),
            "max_issuer_weight":  (0.10, -1),
            "stress_pnl_200bps":  (0.10, +1),   # smaller loss under +200bps wins
            "expense_ratio":      (0.05, -1),
        },
        SEBICategory.HYBRID: {
            "sortino":            (0.20, +1),
            "rr3y_mean":          (0.15, +1),
            "alpha":              (0.15, +1),
            "credit_risk_index":  (0.125, -1),
            "max_drawdown":       (0.125, +1),
            "downside_capture":   (0.10, -1),
            "arb_spread_ann":     (0.10, +1),   # only populated for arbitrage peers
            "expense_ratio":      (0.05, -1),
        },
        SEBICategory.SOLUTION_ORIENTED: {
            "rr5y_mean":          (0.25, +1),
            "sortino":            (0.20, +1),
            "alpha":              (0.15, +1),
            "max_drawdown":       (0.15, +1),
            "credit_risk_index":  (0.15, -1),
            "expense_ratio":      (0.10, -1),
        },
    }

    def __init__(self, validator: ValidationOrchestrator) -> None:
        self.validator = validator
        self.log = logging.getLogger("MFPipeline.Scoring")

    def _percentile_rank(self, col: pd.Series, direction: int) -> pd.Series:
        """Peer percentile in [0,1]; NaNs get the peer median (neutral 0.5)."""
        ranked = col.rank(pct=True, ascending=(direction > 0))
        return ranked.fillna(0.5)

    def score_category(self, metrics: pd.DataFrame, category: SEBICategory) -> pd.DataFrame:
        peers = metrics[metrics["category"] == category.value].copy()
        if peers.empty:
            return peers
        factors = self.FACTOR_MAP[category]
        available = {m: (w, d) for m, (w, d) in factors.items() if m in peers.columns
                     and peers[m].notna().any()}
        weight_sum = sum(w for w, _ in available.values())
        self.validator._record(f"scoring:{category.value}", "factor_coverage",
                               weight_sum >= 0.5,
                               f"{len(available)}/{len(factors)} factors, wsum={weight_sum:.2f}")
        composite = pd.Series(0.0, index=peers.index)
        for metric, (weight, direction) in available.items():
            composite += (weight / weight_sum) * self._percentile_rank(peers[metric], direction)
        # Normalise composite in [0,1] -> 1..100 band with min-max stretch.
        lo, hi = composite.min(), composite.max()
        spread = self.validator.safe_divide(composite - lo, (hi - lo), fill=0.5)
        peers["composite_score"] = np.round(1.0 + 99.0 * np.asarray(spread), 1)
        peers["peer_rank"] = peers["composite_score"].rank(ascending=False, method="min").astype(int)
        return peers.sort_values("composite_score", ascending=False)

    def score_universe(self, metrics: pd.DataFrame) -> pd.DataFrame:
        frames = [self.score_category(metrics, cat) for cat in SEBICategory]
        scored = pd.concat([f for f in frames if not f.empty], axis=0)
        self.validator._record("scoring:universe", "all_funds_scored",
                               len(scored) == len(metrics),
                               f"{len(scored)}/{len(metrics)}")
        in_band = scored["composite_score"].between(1, 100).all()
        self.validator._record("scoring:universe", "scores_in_1_100_band", bool(in_band))
        return scored


# ==============================================================================
# ORCHESTRATION — parallel sub-agent dispatch + episodic verification loop
# ==============================================================================
class PipelineOrchestrator:
    """
    End-to-end conductor. Concurrency map:

        SUB_AGENT_A (2 workers)  — ingestion & schema mapping (I/O bound)
        SUB_AGENT_B (8 workers)  — per-fund quant metric computation
        SUB_AGENT_C (4 workers)  — per-fund risk & stress simulation

    Every completed fund registers an iteration with the ValidationOrchestrator,
    which fires the full episodic audit every 10 iterations.
    """

    def __init__(self) -> None:
        self.validator = ValidationOrchestrator()
        self.ingestion = IngestionEngine(self.validator)          # SUB-AGENT A
        self.quant = QuantEngine(self.validator)                  # SUB-AGENT B
        self.risk = RiskStressTester(self.validator)              # SUB-AGENT C
        self.scoring = ScoringEngine(self.validator)
        self.log = logging.getLogger("MFPipeline.Orchestrator")

    def _quant_task(self, meta: FundMetadata, nav_panel: pd.DataFrame,
                    bench_panel: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        fund_levels = nav_panel[meta.amfi_code]
        bench_key = meta.benchmark.upper()
        if bench_key not in bench_panel.columns:
            raise KeyError(f"Benchmark {bench_key} missing for {meta.scheme_name}")
        arb = bench_panel.get(ARBITRAGE_BENCHMARK)
        metrics = self.quant.compute_fund_metrics(meta, fund_levels,
                                                  bench_panel[bench_key], arb)
        return meta.amfi_code, metrics

    def _risk_task(self, meta: FundMetadata,
                   holdings: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        return meta.amfi_code, self.risk.run(meta, holdings)

    def run(self, metadata_records: Sequence[Dict[str, Any]],
            nav_payloads: Sequence[NAVSeries],
            benchmark_levels: Dict[str, pd.Series],
            disclosures: Sequence[PortfolioDisclosure]) -> pd.DataFrame:

        # ---- SUB-AGENT A: ingestion (parallel over independent feeds) --------
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="SUB_AGENT_A") as pool_a:
            fut_meta = pool_a.submit(self.ingestion.register_metadata, metadata_records)
            fut_bench = pool_a.submit(self.ingestion.build_benchmark_panel, benchmark_levels)
            fut_hold = pool_a.submit(self.ingestion.build_holdings_table, disclosures)
            metadata = fut_meta.result()
            bench_panel = fut_bench.result()
            holdings = fut_hold.result()
        nav_panel = self.ingestion.build_nav_panel(nav_payloads)
        universe = [m for code, m in metadata.items() if code in nav_panel.columns]
        self.log.info("Ingestion complete: %d funds, %d benchmark series, %d holding rows",
                      len(universe), bench_panel.shape[1], len(holdings))

        # ---- SUB-AGENT B + C: fan out quant & risk concurrently --------------
        quant_out: Dict[str, Dict[str, float]] = {}
        risk_out: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="SUB_AGENT_B") as pool_b, \
             ThreadPoolExecutor(max_workers=4, thread_name_prefix="SUB_AGENT_C") as pool_c:
            futs = {pool_b.submit(self._quant_task, m, nav_panel, bench_panel): ("quant", m)
                    for m in universe}
            futs.update({pool_c.submit(self._risk_task, m, holdings): ("risk", m)
                         for m in universe})
            for fut in as_completed(futs):
                kind, meta = futs[fut]
                try:
                    code, payload = fut.result()
                    (quant_out if kind == "quant" else risk_out)[code] = payload
                except Exception as exc:  # noqa: BLE001 - quarantine, never crash the batch
                    errors.append(f"{kind}:{meta.amfi_code}:{exc}")
                    self.log.error("Task failed %s %s: %s", kind, meta.amfi_code, exc)
                self.validator.register_iteration(context=f"{kind}:{meta.amfi_code}")
        self.validator._record("orchestrator", "zero_task_failures", not errors,
                               "; ".join(errors[:5]))

        # ---- assemble the master metric matrix --------------------------------
        rows = []
        for m in universe:
            row: Dict[str, Any] = {
                "amfi_code": m.amfi_code, "scheme_name": m.scheme_name,
                "amc": m.amc_name, "category": m.category.value,
                "sub_category": m.sub_category, "benchmark": m.benchmark,
                "aum_cr": m.aum_cr, "expense_ratio": m.expense_ratio,
            }
            row.update(quant_out.get(m.amfi_code, {}))
            row.update(risk_out.get(m.amfi_code, {}))
            rows.append(row)
        matrix = pd.DataFrame(rows).set_index("amfi_code")

        # ---- scoring -----------------------------------------------------------
        scored = self.scoring.score_universe(matrix)
        self.log.info("Pipeline complete. Validation failures: %d",
                      self.validator.failure_count)
        return scored


# ==============================================================================
# END-TO-END EXECUTION HARNESS — synthetic full-universe demonstration
# ==============================================================================
# The generator below fabricates a statistically realistic 40-scheme universe
# spanning all five SEBI buckets so the pipeline can be executed end-to-end in
# a sealed environment. In production, swap `build_synthetic_universe()` for
# live AMFI NAVAll.txt pulls (IngestionEngine.parse_amfi_dump), NSE TRI index
# feeds and monthly SEBI portfolio-disclosure files — the interfaces are
# identical.
# ==============================================================================

def _gbm_levels(rng: np.random.Generator, dates: pd.DatetimeIndex,
                mu: float, sigma: float, start: float = 100.0) -> pd.Series:
    n = len(dates)
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    shocks = rng.normal((mu - 0.5 * sigma ** 2) * dt, sigma * math.sqrt(dt), size=n - 1)
    levels = start * np.exp(np.concatenate([[0.0], np.cumsum(shocks)]))
    return pd.Series(levels, index=dates)


def _driven_levels(rng: np.random.Generator, bench: pd.Series, beta: float,
                   ann_alpha: float, idio_vol: float, start: float) -> pd.Series:
    """Fund path driven by its benchmark: r_f = α/252 + β·r_b + ε (realistic
    correlation structure so betas, tracking error and capture ratios behave)."""
    b_rets = bench.pct_change().fillna(0.0).to_numpy()
    eps = rng.normal(0.0, idio_vol / math.sqrt(TRADING_DAYS_PER_YEAR), size=len(b_rets))
    f_rets = ann_alpha / TRADING_DAYS_PER_YEAR + beta * b_rets + eps
    f_rets = np.clip(f_rets, -0.20, 0.20)  # respect NAVSeries fat-finger guard
    levels = start * np.cumprod(1.0 + f_rets)
    return pd.Series(levels, index=bench.index)


def build_synthetic_universe(seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-07-01", "2026-06-30")

    # ---------------- benchmarks (all TRI where relevant) ---------------------
    bench_specs = {
        "NIFTY 50 TRI": (0.125, 0.165), "NIFTY NEXT 50 TRI": (0.135, 0.20),
        "NIFTY MIDCAP 150 TRI": (0.155, 0.22), "NIFTY SMALLCAP 250 TRI": (0.16, 0.26),
        "NIFTY 500 TRI": (0.13, 0.17),
        "CRISIL COMPOSITE BOND INDEX": (0.070, 0.02),
        "CRISIL SHORT TERM BOND INDEX": (0.068, 0.012),
        "NIFTY 50 ARBITRAGE INDEX": (0.062, 0.008),
        "CRISIL HYBRID 35+65 AGGRESSIVE INDEX": (0.105, 0.115),
    }
    benchmarks = {k: _gbm_levels(rng, dates, mu, sig) for k, (mu, sig) in bench_specs.items()}

    groups = ["HDFC Group", "Tata Group", "Adani Group", "Birla Group", "Bajaj Group",
              "Shriram Group", "Piramal Group", None, None, None]
    ratings_pool = [CreditRating.SOVEREIGN, CreditRating.AAA, CreditRating.AA_PLUS,
                    CreditRating.AA, CreditRating.AA_MINUS, CreditRating.A_PLUS,
                    CreditRating.A1_PLUS, CreditRating.A]

    def equity_holdings(n_stocks: int, cap_band: str) -> List[Holding]:
        raw = rng.dirichlet(np.ones(n_stocks) * 3.0) * 0.97
        caps = {"large": (40_000, 6_00_000, 80, 900), "mid": (12_000, 45_000, 15, 120),
                "small": (1_500, 14_000, 2, 25)}[cap_band]
        hs = [Holding(instrument=f"{cap_band.title()}Co {i:02d}", weight=float(w),
                      asset_type="equity",
                      market_cap_cr=float(rng.uniform(caps[0], caps[1])),
                      avg_daily_traded_value_cr=float(rng.uniform(caps[2], caps[3])))
              for i, w in enumerate(raw)]
        hs.append(Holding(instrument="TREPS/Cash", weight=0.03, asset_type="cash",
                          rating=CreditRating.CASH))
        return hs

    def debt_holdings(n_bonds: int, risk_tilt: float, md_center: float) -> List[Holding]:
        raw = rng.dirichlet(np.ones(n_bonds) * 2.0) * 0.95
        hs = []
        for i, w in enumerate(raw):
            rating = ratings_pool[min(int(rng.exponential(risk_tilt)), len(ratings_pool) - 1)]
            md = float(np.clip(rng.normal(md_center, md_center * 0.3), 0.05, 15))
            hs.append(Holding(
                instrument=f"Bond Issuer {i:02d}", weight=float(w), asset_type="debt",
                rating=rating, corporate_group=str(rng.choice(groups)) if rng.random() < 0.6 else None,
                modified_duration=md, macaulay_duration=md * 1.07,
                ytm=float(0.065 + 0.004 * CREDIT_RATING_SCORE[rating] + rng.normal(0, 0.002))))
        hs.append(Holding(instrument="TREPS/Cash", weight=0.05, asset_type="cash",
                          rating=CreditRating.CASH))
        return hs

    def hybrid_holdings(eq_w: float, hedged_frac: float) -> List[Holding]:
        hs = []
        n_eq = 12
        eq_raw = rng.dirichlet(np.ones(n_eq)) * eq_w
        for i, w in enumerate(eq_raw):
            hs.append(Holding(instrument=f"HybridEq {i:02d}", weight=float(w),
                              asset_type="equity", is_hedged=bool(rng.random() < hedged_frac),
                              market_cap_cr=float(rng.uniform(30_000, 4_00_000)),
                              avg_daily_traded_value_cr=float(rng.uniform(60, 700))))
        debt_w = 0.97 - eq_w
        for h in debt_holdings(8, 0.8, 2.5):
            hs.append(h.model_copy(update={"weight": max(h.weight * debt_w / 1.0, 1e-4)}))
        total = sum(h.weight for h in hs)
        return [h.model_copy(update={"weight": h.weight / total}) for h in hs]

    metadata_records, nav_payloads, disclosures = [], [], []
    amcs = ["Alpha AMC", "Bharat AMC", "Chola AMC", "Deccan AMC", "Everest AMC"]
    code = 100001

    def add_fund(name, cat, sub, bench, beta, ann_alpha, idio_vol, holdings,
                 ter: Optional[float] = None):
        nonlocal code
        ter = float(rng.uniform(0.004, 0.0225)) if ter is None else ter
        metadata_records.append(dict(
            amfi_code=str(code), isin=f"INF{rng.integers(0, 10**9):09d}",
            scheme_name=name, amc_name=str(rng.choice(amcs)), category=cat,
            sub_category=sub, benchmark=bench,
            aum_cr=float(rng.uniform(400, 45_000)),
            expense_ratio=ter,
            inception_date=date(2015, 1, 1)))
        levels = _driven_levels(rng, benchmarks[bench], beta, ann_alpha - ter,
                                idio_vol, start=float(rng.uniform(10, 300)))
        nav_payloads.append(NAVSeries(amfi_code=str(code),
                                      dates=[d.date() for d in dates],
                                      navs=levels.tolist()))
        disclosures.append(PortfolioDisclosure(amfi_code=str(code),
                                               as_of=date(2026, 5, 31), holdings=holdings))
        code += 1

    # -- Equity (12): active funds — beta ≈ 0.9-1.1, α spread, real idio noise ---
    for i in range(4):
        add_fund(f"Large Cap Fund {i+1}", SEBICategory.EQUITY, "Large Cap",
                 "NIFTY 50 TRI", rng.uniform(0.90, 1.08), rng.uniform(-0.02, 0.035),
                 rng.uniform(0.03, 0.06), equity_holdings(30, "large"))
    for i in range(4):
        add_fund(f"Mid Cap Fund {i+1}", SEBICategory.EQUITY, "Mid Cap",
                 "NIFTY MIDCAP 150 TRI", rng.uniform(0.92, 1.10), rng.uniform(-0.02, 0.05),
                 rng.uniform(0.05, 0.09), equity_holdings(40, "mid"))
    for i in range(4):
        add_fund(f"Small Cap Fund {i+1}", SEBICategory.EQUITY, "Small Cap",
                 "NIFTY SMALLCAP 250 TRI", rng.uniform(0.90, 1.12), rng.uniform(-0.03, 0.06),
                 rng.uniform(0.06, 0.11), equity_holdings(55, "small"))

    # -- Index / ETF (6): β=1, α = -drag, tiny idio noise -> low TE, negative TD --
    for i in range(3):
        add_fund(f"Nifty 50 Index Fund {i+1}", SEBICategory.INDEX_ETF, "Index Fund",
                 "NIFTY 50 TRI", 1.0, -rng.uniform(0.0002, 0.0015),
                 rng.uniform(0.001, 0.006), equity_holdings(50, "large"),
                 ter=float(rng.uniform(0.0005, 0.0035)))
    for i in range(3):
        add_fund(f"Nifty Next 50 ETF {i+1}", SEBICategory.INDEX_ETF, "ETF",
                 "NIFTY NEXT 50 TRI", 1.0, -rng.uniform(0.0002, 0.0020),
                 rng.uniform(0.001, 0.008), equity_holdings(50, "large"),
                 ter=float(rng.uniform(0.0005, 0.0045)))

    # -- Debt (12) ----------------------------------------------------------------
    for i in range(4):
        add_fund(f"Corporate Bond Fund {i+1}", SEBICategory.DEBT, "Corporate Bond",
                 "CRISIL COMPOSITE BOND INDEX", rng.uniform(0.85, 1.05),
                 rng.uniform(-0.004, 0.008), rng.uniform(0.002, 0.006),
                 debt_holdings(20, 0.7, 3.0), ter=float(rng.uniform(0.002, 0.007)))
    for i in range(4):
        add_fund(f"Credit Risk Fund {i+1}", SEBICategory.DEBT, "Credit Risk",
                 "CRISIL COMPOSITE BOND INDEX", rng.uniform(0.80, 1.05),
                 rng.uniform(0.004, 0.018), rng.uniform(0.006, 0.014),
                 debt_holdings(16, 2.2, 2.2), ter=float(rng.uniform(0.006, 0.016)))
    for i in range(4):
        add_fund(f"Short Duration Fund {i+1}", SEBICategory.DEBT, "Short Duration",
                 "CRISIL SHORT TERM BOND INDEX", rng.uniform(0.85, 1.05),
                 rng.uniform(-0.003, 0.005), rng.uniform(0.001, 0.004),
                 debt_holdings(22, 0.6, 1.4), ter=float(rng.uniform(0.002, 0.006)))

    # -- Hybrid (8) -----------------------------------------------------------------
    for i in range(3):
        add_fund(f"Aggressive Hybrid Fund {i+1}", SEBICategory.HYBRID, "Aggressive Hybrid",
                 "CRISIL HYBRID 35+65 AGGRESSIVE INDEX", rng.uniform(0.90, 1.05),
                 rng.uniform(-0.015, 0.03), rng.uniform(0.02, 0.05),
                 hybrid_holdings(0.70, 0.05))
    for i in range(2):
        add_fund(f"Conservative Hybrid Fund {i+1}", SEBICategory.HYBRID, "Conservative Hybrid",
                 "CRISIL HYBRID 35+65 AGGRESSIVE INDEX", rng.uniform(0.30, 0.45),
                 rng.uniform(0.01, 0.03), rng.uniform(0.01, 0.02),
                 hybrid_holdings(0.20, 0.05))
    for i in range(3):
        add_fund(f"Arbitrage Fund {i+1}", SEBICategory.HYBRID, "Arbitrage",
                 "NIFTY 50 ARBITRAGE INDEX", rng.uniform(0.85, 1.0),
                 rng.uniform(-0.004, 0.006), rng.uniform(0.001, 0.004),
                 hybrid_holdings(0.68, 0.98), ter=float(rng.uniform(0.002, 0.01)))

    # -- Solution Oriented (2) ---------------------------------------------------------
    for i, nm in enumerate(["Retirement Fund", "Children's Gift Fund"]):
        add_fund(nm, SEBICategory.SOLUTION_ORIENTED, "Retirement/Children",
                 "NIFTY 500 TRI", rng.uniform(0.55, 0.75), rng.uniform(-0.01, 0.02),
                 rng.uniform(0.02, 0.04), hybrid_holdings(0.55, 0.05))

    return metadata_records, nav_payloads, benchmarks, disclosures


SAMPLE_AMFI_DUMP = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Large Cap Fund)

Alpha AMC Mutual Fund

100001;INF123456789;INF987654321;Alpha Large Cap Fund - Direct Growth;284.5612;06-Jul-2026
100002;INF223456789;-;Alpha Large Cap Fund - Regular Growth;N.A.;06-Jul-2026

Open Ended Schemes(Debt Scheme - Corporate Bond Fund)

Bharat AMC Mutual Fund

100013;INF323456789;INF887654321;Bharat Corporate Bond Fund - Direct Growth;38.9021;06-Jul-2026
"""


def main() -> None:
    orch = PipelineOrchestrator()

    # Demonstrate AMFI text-dump parsing (Sub-Agent A raw feed path).
    amfi_long = orch.ingestion.parse_amfi_dump(SAMPLE_AMFI_DUMP)
    LOGGER.info("AMFI dump demo parsed %d valid NAV rows (N.A./headers skipped)", len(amfi_long))

    # Full end-to-end run on the synthetic universe.
    meta, navs, benchmarks, disclosures = build_synthetic_universe()
    scored = orch.run(meta, navs, benchmarks, disclosures)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 12)
    show = ["scheme_name", "category", "sub_category", "composite_score", "peer_rank",
            "sortino", "alpha", "tracking_error", "credit_risk_index"]
    for cat in SEBICategory:
        block = scored[scored["category"] == cat.value]
        if block.empty:
            continue
        print(f"\n================ {cat.value.upper()} — TOP RANKED ================")
        print(block[[c for c in show if c in block.columns]].head(5).to_string())

    print("\n================ VALIDATION SUMMARY ================")
    print(orch.validator.summary().to_string())
    print(f"\nTotal validation checks: {len(orch.validator.records)} | "
          f"failures: {orch.validator.failure_count}")

    scored.to_csv("/home/claude/mf_universe_scored.csv")
    LOGGER.info("Scored universe written to mf_universe_scored.csv")


if __name__ == "__main__":
    main()
