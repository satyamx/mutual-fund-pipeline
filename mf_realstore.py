"""
====================================================================================
mf_realstore.py — RealNAVStore: production data-store swap-in for the orchestrator
====================================================================================
Implements the SAME narrow interface the agents talk to on MockMarketDataStore
(resolve / fund / snapshots / stock_prices / sector_index / benchmark_series /
sibling_equity_schemes / news / daily_weights_last_quarter — see
mf_agent_orchestrator._bind_store_interface), backed by the real adapters in
mf_datasources.py and the cached real benchmark indices in mf_benchmarks.py.

HONESTY CONTRACT (do not violate this):
  We have NO free source of historical or current portfolio HOLDINGS. This store
  never invents them. Holdings-dependent methods (`snapshots`, and transitively
  the Manager Alpha backtest / SEBI cap-fidelity checks) return an explicit
  "unavailable" shape unless a manually-supplied disclosure CSV is present at
  mf_cache/disclosures/<amfi_code>_<YYYY-MM>.csv (columns: instrument,weight,
  asset_type,sector,cap_band — see bootstrap.py). The dependent agents
  (mf_agent_orchestrator.HistoricalBacktesterAgent, TrueToLabelVerifier) already
  key off FundDossier.holdings_available to degrade honestly instead of scoring
  an empty portfolio.

Similarly, expense ratio (TER), AUM and fund-manager name have NO free real-data
source either (AMFI's NAVAll.txt and mfapi.in's metadata carry neither) — they
are surfaced as NaN / an explicit "NOT AVAILABLE" string, never guessed.
====================================================================================
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from mf_datasources import (
    CACHE, AMFIRegistryLive, AnthropicNewsAgent, CapBandAdapter, MFAPIAdapter,
    YFinanceAdapter,
)
from mf_benchmarks import CATEGORY_BENCHMARK, SECTOR_BENCHMARK, load_benchmark_series
from mf_labels import MANIFEST_PATH

LOGGER = logging.getLogger("MFOrchestrator.RealStore")

# Re-exported alias, not a second definition: the manifest is hand-curated and
# git-tracked under overrides/ (see mf_labels.MANIFEST_PATH for why). Two
# independent literals here is exactly how a moved file silently keeps working
# in one module and breaks in the other.
UNIVERSE_MANIFEST = MANIFEST_PATH
DISCLOSURES_DIR = CACHE / "disclosures"


class AmbiguousFundQuery(LookupError):
    """Raised when a query token-matches more than one manifest fund. Distinct
    from "not found" on purpose: silently answering an ambiguous query with the
    first candidate is what let three funds be scored as the wrong scheme."""

# AMFI publishes category as free text inside the NAVAll.txt banner lines, e.g.
# "Open Ended Schemes(Equity Scheme - Flexi Cap Fund)". Extracted from real,
# structured AMFI text — not guessed — and mapped to the label vocabulary
# mf_agent_orchestrator.SEBI_2026_RULES expects. Categories with no entry in
# that registry (e.g. ELSS, Large & Mid Cap) are passed through unmapped; the
# orchestrator already degrades a not-in-registry category to a single honest
# "category not in Feb-2026 registry" finding rather than crashing.
_RAW_SUFFIX_TO_LABEL: Dict[str, str] = {
    "FLEXI CAP FUND": "Flexi Cap",
    "LARGE CAP FUND": "Large Cap",
    "MID CAP FUND": "Mid Cap",
    "SMALL CAP FUND": "Small Cap",
    "LARGE & MID CAP FUND": "Large & Mid Cap",
    "MULTI CAP FUND": "Multi Cap",
    "VALUE FUND": "Value",
    "CONTRA FUND": "Contra",
    "DIVIDEND YIELD FUND": "Dividend Yield",
    "FOCUSED FUND": "Focused",
    "ELSS": "ELSS",
    "SECTORAL/ THEMATIC": "Thematic",
    "THEMATIC FUND": "Thematic",
    "AGGRESSIVE HYBRID FUND": "Aggressive Hybrid",
    "CONSERVATIVE HYBRID FUND": "Conservative Hybrid",
    "ARBITRAGE FUND": "Arbitrage",
    "INDEX FUNDS": "Index Fund",
    "EQUITY ETF": "ETF",
    "DEBT ETF": "ETF",
}


def _bucket_from_raw(category_raw: Optional[str]):
    """Best-effort SEBIBucket from AMFI's raw banner text (real text, regex-parsed,
    not guessed). Imported lazily to avoid a module-load cycle with the
    orchestrator (which imports RealNAVStore lazily too, inside __init__)."""
    from mf_agent_orchestrator import SEBIBucket
    if not category_raw:
        return SEBIBucket.EQUITY
    r = category_raw.upper()
    if "DEBT SCHEME" in r:
        return SEBIBucket.DEBT
    if "HYBRID SCHEME" in r:
        return SEBIBucket.HYBRID
    if "EXCHANGE TRADED" in r or "INDEX FUND" in r or "FUND OF FUNDS" in r:
        return SEBIBucket.OTHER_PASSIVE
    return SEBIBucket.EQUITY


def _category_from_raw(category_raw: Optional[str]) -> str:
    if not category_raw:
        return "Unknown"
    m = re.search(r"\(([^)]+)\)", category_raw)
    inner = m.group(1) if m else category_raw
    frag = inner.split(" - ")[-1].strip().upper()
    return _RAW_SUFFIX_TO_LABEL.get(frag, inner.split(" - ")[-1].strip())


class RealNAVStore:
    """Production data store for MasterOrchestrator(live=True)."""

    HOLDINGS_COLUMNS = ["name", "sector", "cap_band", "weight", "asset_type"]

    def __init__(self, manifest_path: Path = UNIVERSE_MANIFEST) -> None:
        self.amfi = AMFIRegistryLive()
        self.mfapi = MFAPIAdapter()
        self.capband = CapBandAdapter()
        self.yf = YFinanceAdapter()
        self.news_agent = AnthropicNewsAgent()
        self._manifest = self._load_manifest(manifest_path)
        self._resolved: Dict[str, Dict[str, Any]] = {}
        self._fund_cache: Dict[str, Dict[str, Any]] = {}
        self.log = LOGGER

    @staticmethod
    def _load_manifest(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path, dtype={"amfi_code": str})
        LOGGER.warning("No universe manifest at %s — resolving purely against live "
                       "AMFI scheme master.", path)
        return pd.DataFrame(columns=["amfi_code", "scheme_name", "amc", "category", "sector"])

    # ---------------------------------------------------------------- resolve
    def resolve(self, query: str) -> Optional[str]:
        q = query.strip()
        if not q:
            return None
        try:
            row = self._manifest_match(q)
        except AmbiguousFundQuery:
            # Deliberately NOT falling through to the live registry: a query that
            # names two funds in the curated manifest is ambiguous, full stop, and
            # deferring it to a fuzzier index would just relocate the guess.
            return None
        if row is not None:
            name = row["scheme_name"]
            sector = row.get("sector")
            self._resolved[name] = dict(
                source="manifest", amfi_code=str(row["amfi_code"]), amc=row.get("amc"),
                category=row.get("category"),
                declared_sector=(sector if isinstance(sector, str) and sector.strip() else None))
            return name
        amfi_row = self.amfi.resolve(q)
        if amfi_row is None:
            return None
        name = amfi_row["scheme_name"]
        self._resolved[name] = dict(
            source="amfi_live", amfi_code=str(amfi_row["amfi_code"]), amc=amfi_row.get("amc"),
            category=_category_from_raw(amfi_row.get("category_raw")),
            category_raw=amfi_row.get("category_raw"),
            declared_sector=None, isin=amfi_row.get("isin"))
        return name

    def _manifest_match(self, q: str) -> Optional[pd.Series]:
        """Exact amfi_code, then exact scheme_name, then a token-subset fallback
        that REFUSES rather than guessing when it matches more than one fund.

        The fallback is a subset test, so a full scheme name can match a *longer*
        one whose extra words it doesn't contain. Three funds in the current
        136-row manifest do exactly that — "Franklin India Mid Cap Fund" is a
        token-subset of "Franklin India Large & Mid Cap Fund", likewise Nippon
        and Tata — and until 2026-08-04 this method took `hit.iloc[0]`, i.e.
        whichever collided first in manifest order. Every one of the three
        resolved to the WRONG fund, silently: the batch scored 136 names into 117
        distinct funds and reported 0 errors, and three predictions against the
        wrong scheme reached the append-only ledger.

        Hence: an exact-name branch (the batch passes full manifest names, so it
        never reaches the fallback at all), and an explicit refusal on ambiguity.
        Returning the first of several candidates is the defect — a caller that
        gets None degrades to a visible resolution error, which is the honest
        outcome for a query that genuinely names two funds.
        """
        if self._manifest.empty:
            return None
        names = self._manifest["scheme_name"].str.upper()
        q_up = q.upper()
        hit = self._manifest[self._manifest["amfi_code"].astype(str) == q]
        if hit.empty:
            hit = self._manifest[names == q_up]
        if hit.empty:
            toks = [t for t in re.split(r"\s+", q_up) if t]
            mask = pd.Series(True, index=self._manifest.index)
            for t in toks:
                mask &= names.str.contains(re.escape(t), na=False)
            hit = self._manifest[mask]
            if len(hit) > 1:
                raise AmbiguousFundQuery(
                    f"{q!r} matches {len(hit)} manifest funds — "
                    + "; ".join(f"{r.scheme_name} [{r.amfi_code}]"
                                for r in hit.head(4).itertuples())
                    + ". Name the fund exactly or pass its AMFI code.")
        return None if hit.empty else hit.iloc[0]

    # ------------------------------------------------------------------ fund
    def fund(self, name: str) -> Dict[str, Any]:
        if name in self._fund_cache:
            return self._fund_cache[name]
        info = self._resolved.get(name)
        if info is None:
            raise LookupError(f"fund() called before a successful resolve() for {name!r}")
        code = info["amfi_code"]
        nav, meta, _rep = self.mfapi.nav_series(code)
        if nav is None or nav.empty:
            raise LookupError(
                f"No NAV history cached or fetchable for {name!r} (AMFI code {code}). "
                f"Run: bootstrap.py --funds \"{name}\"")
        category = info["category"] or "Unknown"
        if category == "Sectoral/Thematic":
            # Both "Sectoral" and "Thematic" carry identical rule parameters in
            # SEBI_2026_RULES (segment 0.80 / overlap 0.50 / same scope) — the
            # registry's split of AMFI's single merged category is cosmetic, so
            # collapsing to one umbrella label loses no compliance information.
            category = "Thematic"
        declared_sector = info.get("declared_sector")
        bench_name = (SECTOR_BENCHMARK.get(declared_sector) if declared_sector
                     else CATEGORY_BENCHMARK.get(category))
        isin = info.get("isin") or (meta or {}).get("isin_growth") or ""
        rec = dict(
            isin=isin,
            amfi_code=code,
            category=category,
            amc=info.get("amc") or (meta or {}).get("fund_house") or "Unknown",
            manager="NOT AVAILABLE (no free data source publishes fund-manager names)",
            declared_sector=declared_sector,
            benchmark=bench_name,          # may be None — no real index for this category
            expense_ratio=float("nan"),    # NOT AVAILABLE — no free TER source
            aum_cr=float("nan"),           # NOT AVAILABLE — no free AUM source
            nav=nav,
            category_source=info["source"],
        )
        self._fund_cache[name] = rec
        return rec

    # ------------------------------------------------------------- snapshots
    def snapshots(self, name: str) -> Dict[str, Dict[str, Any]]:
        from mf_agent_orchestrator import TODAY
        info = self._resolved.get(name) or {}
        code = info.get("amfi_code")
        empty_holdings = pd.DataFrame(columns=self.HOLDINGS_COLUMNS)
        files = self._disclosure_files(code) if code else []
        if not files:
            self.log.warning(
                "No manually-supplied holdings-disclosure CSV for %s (code %s) under "
                "mf_cache/disclosures/<code>_<YYYY-MM>.csv — Manager Alpha score and "
                "SEBI cap-fidelity checks will report NOT AVAILABLE, not fabricated.",
                name, code)
            return {label: dict(as_of=TODAY, holdings=empty_holdings.copy(), available=False)
                    for label in ("t-5y", "t-3y", "current")}
        return self._build_snapshots_from_files(files, empty_holdings, TODAY)

    @staticmethod
    def _disclosure_files(code: str) -> List[Path]:
        if not DISCLOSURES_DIR.exists():
            return []
        return sorted(DISCLOSURES_DIR.glob(f"{code}_*.csv"))

    def _build_snapshots_from_files(self, files: List[Path], empty_holdings: pd.DataFrame,
                                    today: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        parsed = []
        for p in files:
            m = re.match(r"^\d+_(\d{4})-(\d{2})\.csv$", p.name)
            if m:
                parsed.append((pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1), p))
        if not parsed:
            self.log.warning("Disclosure file(s) found but none matched the "
                             "<code>_<YYYY-MM>.csv naming convention: %s", [p.name for p in files])
            return {label: dict(as_of=today, holdings=empty_holdings.copy(), available=False)
                    for label in ("t-5y", "t-3y", "current")}
        targets = {"t-5y": today - pd.DateOffset(years=5),
                  "t-3y": today - pd.DateOffset(years=3),
                  "current": today}
        out: Dict[str, Dict[str, Any]] = {}
        for label, target in targets.items():
            as_of, path = min(parsed, key=lambda x: abs((x[0] - target).days))
            out[label] = dict(as_of=as_of, holdings=self._load_disclosure_csv(path), available=True)
        return out

    def _load_disclosure_csv(self, path: Path) -> pd.DataFrame:
        """Parse instrument,weight,asset_type,sector,cap_band and overlay REAL
        AMFI cap-band classification (design §6.3) over whatever the manual CSV
        recorded, for equity rows — the live classification is more current
        than a one-off manual disclosure snapshot."""
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}

        def col(*names: str) -> Optional[pd.Series]:
            for n in names:
                if n in cols:
                    return df[cols[n]]
            return None

        instrument = col("instrument", "name", "symbol")
        weight = col("weight")
        asset_type = col("asset_type", "type")
        sector = col("sector")
        cap_band = col("cap_band", "cap band", "capband")
        if instrument is None or weight is None:
            self.log.error("Disclosure CSV %s missing required instrument/weight columns "
                           "— treating as empty rather than guessing.", path)
            return pd.DataFrame(columns=self.HOLDINGS_COLUMNS)

        instrument = instrument.astype(str).str.strip()
        out = pd.DataFrame({
            "name": instrument,
            "weight": pd.to_numeric(weight, errors="coerce"),
            "asset_type": (asset_type.astype(str).str.strip().str.lower()
                          if asset_type is not None else "equity"),
            "sector": sector.astype(str).str.strip() if sector is not None else None,
            "cap_band": (cap_band.astype(str).str.strip().str.lower()
                        if cap_band is not None else None),
        })
        out.index = instrument.str.upper()
        out = out.dropna(subset=["weight"])

        bands = self.capband.band_lookup()
        if bands:
            eq_mask = out["asset_type"] == "equity"
            live_band = out.index.to_series().map(bands.get)
            overlay = eq_mask & live_band.notna()
            out.loc[overlay, "cap_band"] = live_band[overlay]
        return out

    # ------------------------------------------------- daily weights / peers
    def daily_weights_last_quarter(self, name: str) -> Dict[str, pd.Series]:
        # No daily/near-daily portfolio feed exists (monthly manual disclosures
        # at best) — honestly empty rather than a synthetic daily-drift path.
        return {}

    def sibling_equity_schemes(self, scheme_name: str, amc: str) -> Dict[str, Dict[str, pd.Series]]:
        # Requires disclosed daily equity weights for peer schemes too, which we
        # don't have (see daily_weights_last_quarter) — {} degrades the 50%
        # overlap check to "not evaluated" honestly rather than fabricating peers.
        return {}

    # -------------------------------------------------------- stock / sector
    def stock_prices(self, tickers: Sequence[str], start) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        px = self.yf.prices(list(tickers), start=str(pd.Timestamp(start).date()))
        return px if px is not None else pd.DataFrame()

    def sector_index(self, sector: str, start, basket=None) -> pd.Series:
        name = SECTOR_BENCHMARK.get(sector)
        if name is None:
            self.log.warning("No real index mapped for sector %r; sector benchmark "
                             "unavailable (not synthesized).", sector)
            return pd.Series(dtype=float)
        s = load_benchmark_series(name)
        if s is None:
            self.log.warning("Real index series for %r not present under "
                             "benchmarks/; sector benchmark unavailable.", name)
            return pd.Series(dtype=float)
        return s.loc[pd.Timestamp(start):]

    # -------------------------------------------------------------- benchmark
    def benchmark_series(self, benchmark: Optional[str]) -> pd.Series:
        if not benchmark:
            self.log.warning("No real index mapped for this fund's category — beta/alpha "
                             "vs benchmark reported unavailable, not compared to a "
                             "synthetic index.")
            return pd.Series(dtype=float)
        s = load_benchmark_series(benchmark)
        if s is None:
            self.log.warning("Real index series for %r not present under "
                             "benchmarks/ — benchmark unavailable.", benchmark)
            return pd.Series(dtype=float)
        return s

    # ------------------------------------------------------------------ news
    def news(self, entities: Sequence[str]) -> List[Dict[str, Any]]:
        from mf_agent_orchestrator import TODAY
        if not self.news_agent.health():
            return []  # no ANTHROPIC_API_KEY: neutral, never invented (matches AnthropicNewsAgent)
        amc = entities[0] if entities else ""
        manager = entities[1] if len(entities) > 1 else ""
        sectors = list(entities[2:])
        result = self.news_agent.research(amc, manager, sectors)
        items = []
        for it in result.get("items", []):
            try:
                days_ago = max(0, (TODAY - pd.Timestamp(it["date"])).days)
            except Exception:  # noqa: BLE001 — malformed date from the LLM's JSON
                days_ago = 0
            items.append(dict(it, days_ago=days_ago))
        return items


# ==============================================================================
# Selftest — resolution only. Hermetic: a synthetic manifest, no network, no cache.
# ==============================================================================
def _selftest() -> None:
    import tempfile

    # The three real collisions from the 136-fund manifest, reduced to their
    # shape: a full scheme name that is a token-SUBSET of a longer one. Ordered
    # so the wrong answer comes first, which is exactly how iloc[0] picked it.
    rows = [
        dict(amfi_code="118510", scheme_name="Franklin India Large & Mid Cap Fund - Direct - Growth",
             amc="Franklin", category="Large & Mid Cap", sector=""),
        dict(amfi_code="118509", scheme_name="Franklin India Mid Cap Fund - Direct - Growth",
             amc="Franklin", category="Mid Cap", sector=""),
        dict(amfi_code="120505", scheme_name="Parag Parikh Flexi Cap Fund - Direct - Growth",
             amc="PPFAS", category="Flexi Cap", sector=""),
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        store = RealNAVStore(manifest_path=p)

        # ---- the regression: a full name must resolve to ITSELF, never to the
        # longer scheme it is a token-subset of.
        for r in rows:
            hit = store._manifest_match(r["scheme_name"])
            assert hit is not None, f"FAIL: {r['scheme_name']!r} did not resolve"
            assert str(hit["amfi_code"]) == r["amfi_code"], (
                f"FAIL: {r['scheme_name']!r} resolved to {hit['amfi_code']} "
                f"(expected {r['amfi_code']}) — the subset collision is back")
        assert len({str(store._manifest_match(r["scheme_name"])["amfi_code"]) for r in rows}) == 3, \
            "FAIL: three distinct names must resolve to three distinct funds"
        print("[selftest] exact scheme_name beats the token-subset fallback — PASS")

        # ---- an ambiguous FRAGMENT must refuse, not pick the first candidate.
        try:
            store._manifest_match("Franklin India Mid Cap")
            raise AssertionError("FAIL: an ambiguous fragment must raise, not guess")
        except AmbiguousFundQuery:
            pass
        # ...and resolve() must swallow it as an honest miss WITHOUT falling
        # through to the fuzzier live registry (which would relocate the guess).
        assert store.resolve("Franklin India Mid Cap") is None, \
            "FAIL: an ambiguous query must not resolve via the AMFI fallback"
        print("[selftest] ambiguous fragment refuses + does not fall through to AMFI — PASS")

        # ---- unambiguous fragments and code lookups still work.
        assert str(store._manifest_match("Parag Parikh")["amfi_code"]) == "120505"
        assert str(store._manifest_match("118509")["amfi_code"]) == "118509"
        assert store._manifest_match("No Such Fund Anywhere") is None
        print("[selftest] unambiguous fragment / amfi_code / miss all unchanged — PASS")

    print("[selftest] PASS — manifest resolution is exact-first and refuses to guess")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RealNAVStore resolution selftest")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        _selftest()
    else:
        ap.print_help()
