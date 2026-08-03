"""
====================================================================================
mf_benchmarks.py — bias-aware hybrid benchmark resolver
====================================================================================
Resolves a fund's category/sector to a forward benchmark return over a given
(anchor, anchor+3y) window, per docs/phase_b_design.md §1.3 with the CONFIRMED
hybrid extension: prefer a REAL index over the peer-proxy wherever the cached
real-index history covers the window.

Source preference per (fund, anchor, 3y window):
  (a) fund_NAV real index  — an index-tracking fund's own (cleaned) NAV series,
      i.e. genuine total return, if it covers the whole window.
  (b) adj_pri real index   — a yfinance price-return (PRI) index series, if it
      covers the whole window; annualized return is bumped by
      DIVIDEND_YIELD_ADDBACK to approximate total return (PRI understates TRI by
      the index's dividend yield, ~1-1.5%/yr for Nifty indices).
  (c) peer_proxy           — leave-one-out (LOO) median of same-category
      (diversified) or same-sector (thematic, >=3 funds) peer forward returns;
      thematic sectors with <3 funds pool into one LOO "small-sector" group
      (design §1.3). Minimum 2 peers after LOO exclusion, else no benchmark.
  (d) none                 — no valid benchmark; the excess leg (condA) is False.

Every result carries benchmark_source in {fund_nav, adj_pri, peer_proxy, none}
and a more granular benchmark_quality tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("MFBenchmarks")

# GIT-TRACKED, deliberately NOT under mf_cache/ (2026-08-03). No script in this
# repo regenerates any of this: mf_benchmarks has no builder and bootstrap.py never
# writes it, so the availability sheet (partly hand-annotated — its `notes` column
# maps each index to a specific AMFI code or yfinance ticker) and the 20 index
# parquets are unbackfillable. Living in mf_cache/ meant gitignored AND restored
# from an evictable actions/cache, i.e. one eviction from gone — the same trap
# already fixed for ledger/ and overrides/universe_manifest.csv. It also meant
# every cold CI runner scored the whole universe with no benchmark at all.
BENCH_DIR = Path(__file__).parent / "benchmarks"
AVAILABILITY_CSV = BENCH_DIR / "benchmark_availability.csv"

# Dividend-yield addback applied to PRI (price-return) index series to
# approximate TRI (total-return). Tunable module constant, per design.
DIVIDEND_YIELD_ADDBACK = 0.013  # 1.3%/yr, documented estimate for Nifty sector indices

# ---- fund category -> benchmark index name (manifest `category`) -----------------
# Large & Mid Cap has no matching single real index in benchmarks/; it
# falls straight through to peer-proxy (documented, not silently guessed).
CATEGORY_BENCHMARK: Dict[str, Optional[str]] = {
    "Flexi Cap": "Nifty 500",
    "Large Cap": "Nifty 100",
    "Mid Cap": "Nifty Midcap 150",
    "Small Cap": "Nifty Smallcap 250",
    "Multi Cap": "Nifty 500",
    "ELSS": "Nifty 500",
    "Value/Contra": "Nifty 500",
    "Focused": "Nifty 500",
    "Large & Mid Cap": None,
}

# ---- thematic sector -> benchmark index name (manifest `sector`) -----------------
SECTOR_BENCHMARK: Dict[str, Optional[str]] = {
    "Banking/Financial Services": "Nifty Bank",
    "FMCG/Consumption": "Nifty FMCG",
    "Infrastructure": "Nifty Infrastructure",
    "Pharma/Healthcare": "Nifty Pharma",
    "IT/Technology": "Nifty IT",
    "Manufacturing": "Nifty India Manufacturing",
    "MNC": "Nifty MNC",
    "Auto/Transportation": "Nifty Auto",
    "Energy/Power": "Nifty Energy",
    "PSU": "Nifty PSE",
    "Commodities/Natural Resources": "Nifty Commodities",
    "Housing/Real Estate": "Nifty Realty",
    "Defence": "Nifty India Defence",
    "ESG": "Nifty100 ESG Sector Leaders",
    "Exports": None,
    "Quant": None,
    "Business Cycle": None,
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


@lru_cache(maxsize=1)
def _availability() -> pd.DataFrame:
    """The benchmark availability sheet, or an EMPTY one if it is missing.

    This used to be a bare read_csv, so an absent sheet raised FileNotFoundError
    out of every benchmark lookup. That is not a degrade — it propagated through
    MasterOrchestrator.evaluate()'s outer handler and DISCARDED each fund's
    otherwise-complete evaluation. On the first real CI run (2026-08-03, run
    30774345802) it dropped 118 of 136 funds while the workflow still reported
    success, because BENCH_DIR then lived under gitignored, cache-restored
    mf_cache/. BENCH_DIR is git-tracked now, so absence should no longer happen
    on a clean checkout — the guard stays because the failure mode it prevents is
    silent and universe-wide, and "it can't happen now" is what was assumed before.

    Empty is the honest degrade and needs no downstream change: _benchmark_meta()
    becomes {}, load_benchmark_series() returns None for every name (its existing
    "source == NONE" path), and the caller falls back to PeerProxyResolver — the
    same route already taken by a sector with no mapped index. The fund is scored
    with excess-vs-benchmark reported as unavailable, never fabricated.
    """
    if not AVAILABILITY_CSV.exists():
        LOGGER.warning(
            "No benchmark availability sheet at %s — every real-index lookup will "
            "report unavailable and fall back to the peer proxy. Benchmark-relative "
            "facts (excess_cagr_*) degrade to NOT AVAILABLE; they are never guessed.",
            AVAILABILITY_CSV)
        return pd.DataFrame(columns=["benchmark_name", "source", "return_type"])
    return pd.read_csv(AVAILABILITY_CSV)


@lru_cache(maxsize=1)
def _benchmark_meta() -> Dict[str, Dict[str, str]]:
    """benchmark_name -> {source, return_type}, from benchmark_availability.csv."""
    df = _availability()
    return {r.benchmark_name: dict(source=r.source, return_type=r.return_type)
            for r in df.itertuples()}


@lru_cache(maxsize=32)
def load_benchmark_series(name: str) -> Optional[pd.Series]:
    """Cleaned, sorted, single-column value series for a real index, or None if
    the index has no cached data (source == NONE in the availability sheet)."""
    meta = _benchmark_meta().get(name)
    if meta is None or meta["source"] == "NONE":
        return None
    path = BENCH_DIR / f"{_slug(name)}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    col = "close" if "close" in df.columns else "nav"
    s = df[col].dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    s.name = name
    return s


# ==============================================================================
# Shared as-of / forward-return engine (mirrors design §1.1-1.2 exactly; also
# used by mf_labels.py for the fund's own R_fwd so both sides share one rule).
# ==============================================================================
@dataclass
class ForwardReturn:
    value: float
    d0: pd.Timestamp
    d1: pd.Timestamp
    yrs: float


def _asof(series: pd.Series, target: pd.Timestamp, tol_days: int = 10
          ) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """Last value on or before `target`, within `tol_days` calendar days."""
    pos = series.index.searchsorted(target, side="right") - 1
    if pos < 0:
        return None, None
    d0 = series.index[pos]
    if (target - d0).days > tol_days:
        return None, None
    return d0, float(series.iloc[pos])


def forward_return(series: pd.Series, t: pd.Timestamp, years: int = 3,
                    tol_days: int = 10, min_yrs: float = 2.75) -> Optional[ForwardReturn]:
    """Annualized forward return from `t` to `t + years`, per design §1.2:
    NAV0 = last value <= t (within tol_days); NAV1 = last value <= t+years
    (within tol_days); require yrs >= min_yrs; else None (no sample)."""
    d0, v0 = _asof(series, t, tol_days)
    if d0 is None or v0 <= 0:
        return None
    d1, v1 = _asof(series, t + pd.DateOffset(years=years), tol_days)
    if d1 is None:
        return None
    yrs = (d1 - d0).days / 365.25
    if yrs < min_yrs:
        return None
    return ForwardReturn(value=(v1 / v0) ** (1.0 / yrs) - 1.0, d0=d0, d1=d1, yrs=yrs)


def real_index_forward(name: Optional[str], t: pd.Timestamp
                        ) -> Optional[Tuple[float, str]]:
    """(annualized_return, 'fund_nav'|'adj_pri') if `name`'s real-index series
    covers the [t, t+3y] window, else None."""
    if not name:
        return None
    series = load_benchmark_series(name)
    if series is None:
        return None
    res = forward_return(series, t)
    if res is None:
        return None
    meta = _benchmark_meta()[name]
    if meta["return_type"] == "fund_NAV":
        return res.value, "fund_nav"
    if meta["return_type"] == "PRI":
        return res.value + DIVIDEND_YIELD_ADDBACK, "adj_pri"
    return None


# ==============================================================================
# Peer/sector proxy — leave-one-out median, per design §1.3
# ==============================================================================
class PeerProxyResolver:
    """LOO median forward return over a fund's peer pool: same manifest
    `category` for diversified funds; same `sector` for thematic funds with
    >=3 funds in that sector; a single pooled LOO group across all thematic
    sectors with <3 funds otherwise (design's "pooled_fallback")."""

    def __init__(self, manifest: pd.DataFrame):
        self.manifest = manifest.set_index("amfi_code")
        them = manifest[manifest["category"] == "Sectoral/Thematic"]
        sector_n = them.groupby("sector")["amfi_code"].size()
        self.small_sectors = set(sector_n[sector_n < 3].index)

        self.groups: Dict[Tuple, list] = {}
        for code, row in self.manifest.iterrows():
            self.groups.setdefault(self._group_key(code), []).append(code)

    def _group_key(self, code: str) -> Tuple:
        row = self.manifest.loc[code]
        if row["category"] != "Sectoral/Thematic":
            return ("category", row["category"])
        if row["sector"] in self.small_sectors:
            return ("pooled_thematic",)
        return ("sector", row["sector"])

    def quality(self, code: str) -> str:
        kind = self._group_key(code)[0]
        return {"category": "peer_category", "sector": "peer_sector",
                "pooled_thematic": "pooled_fallback"}[kind]

    def loo_median(self, code: str, t: pd.Timestamp, fwd_lookup: Dict[Tuple[str, pd.Timestamp], float]
                    ) -> Tuple[Optional[float], int]:
        peers = self.groups[self._group_key(code)]
        vals = [fwd_lookup[(p, t)] for p in peers
                if p != code and (p, t) in fwd_lookup and not pd.isna(fwd_lookup[(p, t)])]
        if len(vals) < 2:
            return None, len(vals)
        return float(np.median(vals)), len(vals)


# ==============================================================================
# Top-level resolvers used by mf_labels.py
# ==============================================================================
def _benchmark_name(category: str, sector: Optional[str]) -> Optional[str]:
    if category == "Sectoral/Thematic":
        return SECTOR_BENCHMARK.get(sector)
    return CATEGORY_BENCHMARK.get(category)


def resolve_hybrid(code: str, category: str, sector: Optional[str], t: pd.Timestamp,
                    peer_resolver: PeerProxyResolver,
                    fwd_lookup: Dict[Tuple[str, pd.Timestamp], float]) -> dict:
    """(a) fund_nav -> (b) adj_pri -> (c) peer_proxy -> (d) none."""
    name = _benchmark_name(category, sector)
    real = real_index_forward(name, t)
    if real is not None:
        value, source = real
        quality = "fund_nav" if source == "fund_nav" else "adj_pri_addback"
        return dict(benchmark_fwd=value, benchmark_source=source,
                    benchmark_quality=quality, peer_n=None)
    med, n = peer_resolver.loo_median(code, t, fwd_lookup)
    if med is None:
        return dict(benchmark_fwd=np.nan, benchmark_source="none",
                    benchmark_quality="no_benchmark", peer_n=n)
    return dict(benchmark_fwd=med, benchmark_source="peer_proxy",
                benchmark_quality=peer_resolver.quality(code), peer_n=n)


def resolve_pure_peer(code: str, t: pd.Timestamp, peer_resolver: PeerProxyResolver,
                       fwd_lookup: Dict[Tuple[str, pd.Timestamp], float]) -> dict:
    """Original design default (§1.3): peer-proxy only, never a real index."""
    med, n = peer_resolver.loo_median(code, t, fwd_lookup)
    if med is None:
        return dict(benchmark_fwd=np.nan, benchmark_source="none",
                    benchmark_quality="no_benchmark", peer_n=n)
    return dict(benchmark_fwd=med, benchmark_source="peer_proxy",
                benchmark_quality=peer_resolver.quality(code), peer_n=n)
