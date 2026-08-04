"""
====================================================================================
mf_datasources.py — PRODUCTION DATA LAYER
====================================================================================
Replaces the mock adapters in mf_agent_orchestrator.py with real, live-wired
sources. Every adapter:
  * caches to ./mf_cache/ (so a re-run is instant and works fully offline),
  * degrades to cache -> mock rather than raising,
  * reports .health() so the orchestrator can compute honest confidence.

REAL-DATA LESSONS BAKED IN (found by inspecting live AMFI/mfapi payloads, NOT
assumed): raw Indian NAV series contain
  (a) IDCW payout step-downs  — an IDCW-plan NAV legitimately falls ~5-8% on the
      record date. A naive "reject >25% move" validator throws; a naive returns
      calc books a fake loss. Fix: prefer the GROWTH plan ISIN, and where an
      IDCW series must be used, detect payout steps and reconstruct a total-
      return series rather than dropping the observation.
  (b) Isolated bad prints    — e.g. 117.87 wedged between neighbours of ~104.5,
      reverting the next day. Fix: median-filter spike detector that removes
      only NON-PERSISTENT jumps (a real repricing persists; a typo reverts).
Getting this wrong silently corrupts every downstream CAGR, Sharpe and alpha,
so the cleaner is applied at ingestion and its actions are logged, never hidden.
====================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("MFData")
CACHE = Path(os.environ.get("MF_CACHE_DIR", "./mf_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

AMFI_NAVALL = "https://www.amfiindia.com/spages/NAVAll.txt"
AMFI_CAPLIST_PAGE = "https://www.amfiindia.com/otherdata/categorisation-of-stocks"
MFAPI_BASE = "https://api.mfapi.in"
ANTHROPIC_MESSAGES = "https://api.anthropic.com/v1/messages"


def _requests():
    try:
        import requests  # noqa: PLC0415
        return requests
    except ImportError:
        return None


def _get_json(url: str, timeout: int = 20, retries: int = 3) -> Optional[Any]:
    r = _requests()
    if r is None:
        return None
    for attempt in range(retries):
        try:
            resp = r.get(url, timeout=timeout,
                         headers={"User-Agent": "mf-pipeline/2.0"})
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                LOGGER.warning("GET %s failed after %d tries: %s", url, retries, exc)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _get_text(url: str, timeout: int = 60) -> Optional[str]:
    r = _requests()
    if r is None:
        return None
    try:
        resp = r.get(url, timeout=timeout, headers={"User-Agent": "mf-pipeline/2.0"})
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GET %s failed: %s", url, exc)
        return None


def _get_bytes(url: str, referer: str, timeout: int = 60) -> Optional[bytes]:
    r = _requests()
    if r is None:
        return None
    try:
        resp = r.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Referer": referer})
        resp.raise_for_status()
        return resp.content
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GET %s failed: %s", url, exc)
        return None


# ==============================================================================
# NAV CLEANER — the piece that real data proved we need
# ==============================================================================
@dataclass
class CleaningReport:
    spikes_removed: int = 0
    payout_steps_detected: int = 0
    payout_dates: List[str] = None
    rows_in: int = 0
    rows_out: int = 0
    ambiguous: List[str] = None

    def __post_init__(self):
        if self.payout_dates is None:
            self.payout_dates = []
        if self.ambiguous is None:
            self.ambiguous = []

    def summary(self) -> str:
        return (f"{self.rows_in}->{self.rows_out} obs | "
                f"{self.spikes_removed} bad prints removed | "
                f"{self.payout_steps_detected} IDCW payout steps reconstructed")


class NAVCleaner:
    """
    Two anomalies that look alike numerically but must NEVER share a threshold.

    BAD PRINT (typo): jump that REVERTS on the next observation. Deleted.
    IDCW PAYOUT: step-DOWN that PERSISTS because cash left the fund.

    The naive fix — "treat any big persistent drop as a payout" — is WRONG, and
    testing against real data proved it: the observed payout in scheme 119551 was
    only -6%, while genuine market crashes of -6% happen all the time. Magnitude
    cannot separate them. Two signals can:

      1. PLAN TYPE. Growth plans never pay out, so payout logic is gated behind
         `is_idcw_plan` and is simply never armed for a growth-plan series. This
         is why the resolver hunts for the DIRECT-GROWTH code first: the cleanest
         fix for the IDCW problem is to not analyse an IDCW plan at all.
      2. IDIOSYNCRASY. A payout is fund-specific; a crash is market-wide. When a
         benchmark series is supplied, a drop is only treated as a payout if the
         benchmark did NOT fall with it.

    Anything ambiguous is surfaced in the report, never silently "corrected".
    """

    SPIKE_THRESHOLD = 0.10        # bad-print candidate
    REVERSION_TOLERANCE = 0.5     # reverts >=50% of the jump => typo
    PAYOUT_THRESHOLD = 0.03       # IDCW steps are commonly 3-8%
    BENCHMARK_TOLERANCE = 0.015   # benchmark moved <1.5% => the drop was fund-specific

    def clean(self, s: pd.Series, is_idcw_plan: bool = False,
              benchmark: Optional[pd.Series] = None) -> Tuple[pd.Series, CleaningReport]:
        s = s.dropna().sort_index()
        rep = CleaningReport(rows_in=len(s))
        if len(s) < 5:
            rep.rows_out = len(s)
            return s, rep

        vals = s.to_numpy(dtype=float)
        idx = list(s.index)
        ret = np.diff(vals) / vals[:-1]

        # ---- pass 1: delete non-persistent spikes (bad prints). Always on. ----
        drop: set[int] = set()
        for i in range(len(ret) - 1):
            if abs(ret[i]) < self.SPIKE_THRESHOLD:
                continue
            if ret[i] * ret[i + 1] < 0 and abs(ret[i + 1]) >= self.REVERSION_TOLERANCE * abs(ret[i]):
                drop.add(i + 1)
        if drop:
            keep = [k for k in range(len(vals)) if k not in drop]
            vals, idx = vals[keep], [idx[k] for k in keep]
            rep.spikes_removed = len(drop)
            ret = np.diff(vals) / vals[:-1]

        # ---- pass 2: payout reconstruction. GATED — never armed for growth. ---
        if not is_idcw_plan:
            rep.rows_out = len(vals)
            return pd.Series(vals, index=pd.DatetimeIndex(idx), name=s.name), rep

        bench_ret = None
        if benchmark is not None:
            b = benchmark.reindex(pd.DatetimeIndex(idx)).ffill()
            bench_ret = b.pct_change().to_numpy()[1:]

        tr_factor = np.ones(len(vals))
        for i in range(len(ret)):
            if ret[i] > -self.PAYOUT_THRESHOLD:
                continue
            if bench_ret is not None and np.isfinite(bench_ret[i]):
                # market fell too => a crash, not a payout. Leave it alone.
                if bench_ret[i] < -self.BENCHMARK_TOLERANCE:
                    rep.ambiguous.append(
                        f"{pd.Timestamp(idx[i+1]).date()}: fund {ret[i]:+.1%} vs "
                        f"benchmark {bench_ret[i]:+.1%} -> treated as MARKET MOVE")
                    continue
            rep.payout_steps_detected += 1
            rep.payout_dates.append(str(pd.Timestamp(idx[i + 1]).date()))
            tr_factor[i + 1:] *= (1.0 / (1.0 + ret[i]))

        cleaned = pd.Series(vals * tr_factor, index=pd.DatetimeIndex(idx), name=s.name)
        rep.rows_out = len(cleaned)
        if rep.spikes_removed or rep.payout_steps_detected:
            LOGGER.info("NAV clean [%s]: %s", s.name, rep.summary())
        return cleaned, rep


# ==============================================================================
# ADAPTERS
# ==============================================================================
class MFAPIAdapter:
    """
    api.mfapi.in — free, no key, no auth. Schema verified against a live call:
      {"meta": {"fund_house","scheme_type","scheme_category","scheme_code",
                "scheme_name","isin_growth","isin_div_reinvestment"},
       "data": [{"date":"DD-MM-YYYY","nav":"106.79770"}, ...]}   # newest first
    """

    def __init__(self, cleaner: Optional[NAVCleaner] = None) -> None:
        self.cleaner = cleaner or NAVCleaner()
        self.live = False

    def _cache_path(self, code: str) -> Path:
        return CACHE / f"mfapi_{code}.json"

    def fetch_raw(self, scheme_code: str, max_age_hours: int = 20) -> Optional[Dict[str, Any]]:
        p = self._cache_path(scheme_code)
        if p.exists() and (time.time() - p.stat().st_mtime) < max_age_hours * 3600:
            return json.loads(p.read_text())
        payload = _get_json(f"{MFAPI_BASE}/mf/{scheme_code}")
        if payload and payload.get("data"):
            p.write_text(json.dumps(payload))
            self.live = True
            return payload
        if p.exists():                      # network down -> serve stale cache
            LOGGER.warning("mfapi unreachable; serving cached %s", scheme_code)
            return json.loads(p.read_text())
        return None

    def search(self, query: str) -> List[Dict[str, Any]]:
        res = _get_json(f"{MFAPI_BASE}/mf/search?q={query.replace(' ', '%20')}")
        return res or []

    def nav_series(self, scheme_code: str, benchmark: Optional[pd.Series] = None
                   ) -> Tuple[Optional[pd.Series], Dict[str, Any], Optional[CleaningReport]]:
        payload = self.fetch_raw(scheme_code)
        if not payload:
            return None, {}, None
        meta = payload["meta"]
        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        s = df.dropna(subset=["nav"]).set_index("date")["nav"].sort_index()
        s = s[s > 0]
        s.name = str(scheme_code)
        nm = meta.get("scheme_name", "").upper()
        is_idcw = ("IDCW" in nm) or ("DIVIDEND" in nm)
        cleaned, rep = self.cleaner.clean(s, is_idcw_plan=is_idcw, benchmark=benchmark)
        # Growth-plan preference: warn loudly if this is an IDCW plan, because a
        # raw IDCW NAV understates true return and silently poisons every metric.
        if is_idcw:
            LOGGER.warning("Scheme %s is an IDCW/dividend plan. Total-return "
                           "reconstruction applied (%d payout steps). Prefer the "
                           "GROWTH plan code for clean analytics.",
                           scheme_code, rep.payout_steps_detected)
        return cleaned, meta, rep

    def health(self) -> bool:
        return self.live or any(CACHE.glob("mfapi_*.json"))


class AMFIRegistryLive:
    """AMFI NAVAll.txt -> scheme master (code, ISINs, name, category, AMC)."""

    LINE = re.compile(r"^(?P<code>\d{4,8});(?P<isin1>[^;]*);(?P<isin2>[^;]*);"
                      r"(?P<name>[^;]+);(?P<nav>[0-9.]+|N\.A\.);(?P<date>[^;]+)\s*$")

    # A category header is "<Family> Schemes(<Asset> - <Sub>)". The literal
    # Scheme/Schemes token before the parenthesis is what separates it from an AMC
    # banner — testing only for "(" and ")" mis-reads any AMC whose COMPANY NAME
    # contains parentheses as a category, which then persists to every scheme after
    # it until the next real header. Verified on the live file: 75 genuine headers
    # vs exactly 1 paren-bearing AMC banner, "IL&FS Mutual Fund (IDF)".
    #
    # That banner sits inside the "Close Ended Schemes(Income)" block as an ordinary
    # AMC, so its 12 IDF schemes correctly inherit that legacy debt label and remain
    # unscoreable. Deliberately NOT special-cased any further: blanking the category
    # at a paren-bearing banner would also strip the correct
    # "Close Ended Schemes(Income)" from the 2,532 schemes of the 7 AMCs listed
    # after it in the same block.
    CATEGORY_HEADER = re.compile(r"^.*Schemes?\s*\(.*\)\s*$", re.IGNORECASE)

    def __init__(self) -> None:
        self.live = False

    def master(self, max_age_hours: int = 20) -> pd.DataFrame:
        p = CACHE / "amfi_master.parquet"
        if p.exists() and (time.time() - p.stat().st_mtime) < max_age_hours * 3600:
            return pd.read_parquet(p)
        txt = _get_text(AMFI_NAVALL)
        if txt is None:
            if p.exists():
                LOGGER.warning("AMFI unreachable; serving cached master")
                return pd.read_parquet(p)
            LOGGER.error("AMFI unreachable and no cache. Run bootstrap.py with network.")
            return pd.DataFrame(columns=["amfi_code", "isin", "scheme_name",
                                         "category_raw", "amc", "nav", "date"])
        rows, category, amc = [], None, None
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("Scheme Code"):
                continue
            m = self.LINE.match(line)
            if m:
                if m.group("nav") == "N.A.":
                    continue
                rows.append(dict(amfi_code=m.group("code"),
                                 isin=(m.group("isin1") or "").strip() or None,
                                 scheme_name=m.group("name").strip(),
                                 category_raw=category, amc=amc,
                                 nav=float(m.group("nav")), date=m.group("date")))
            elif self.CATEGORY_HEADER.match(line):  # "Open Ended Schemes(Equity Scheme - Large Cap Fund)"
                category = line
            elif ";" not in line:                   # AMC banner line
                amc = line
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(p)
            self.live = True
        return df

    def resolve(self, query: str) -> Optional[pd.Series]:
        """Accepts a full scheme name, a scheme name fragment, OR an ISIN.
        Prefers DIRECT GROWTH plans.

        The token fallback below is a SUBSET test — every query token must appear
        somewhere in the candidate name — so a full name can match a *longer*
        scheme whose extra words it doesn't contain: "Tata Mid Cap Fund" is a
        token-subset of "Tata Large & MId Cap Fund". Hence two guards, in order:
        an exact-name branch before the fallback (a caller who already knows the
        full name must never be fuzzy-matched at all), and a tightest-match
        tie-break after it so a fragment resolves to the most specific candidate
        rather than to whichever row happened to come first in AMFI's file.
        """
        m = self.master()
        if m.empty:
            return None
        q = query.strip().upper()
        names = m["scheme_name"].str.upper()
        hit = m[m["isin"].fillna("").str.upper() == q]
        if hit.empty:
            hit = m[names == q]
        if hit.empty:
            toks = [t for t in re.split(r"\s+", q) if t]
            mask = pd.Series(True, index=m.index)
            for t in toks:
                mask &= names.str.contains(re.escape(t), na=False)
            hit = m[mask]
        if hit.empty:
            return None
        # rank: DIRECT + GROWTH first (the only plan worth analysing), then the
        # tightest match — fewest characters beyond the query — so an equal-ranked
        # tie is broken by specificity instead of by file order.
        name = hit["scheme_name"].str.upper()
        score = (name.str.contains("DIRECT").astype(int) * 2
                 + name.str.contains("GROWTH").astype(int) * 2
                 - name.str.contains("IDCW|DIVIDEND").astype(int) * 3)
        ranked = hit.assign(_s=score, _len=name.str.len()).sort_values(
            ["_s", "_len"], ascending=[False, True])
        if len(ranked) > 1:
            top = ranked.iloc[0]
            rest = ranked[(ranked["_s"] == top["_s"]) & (ranked["_len"] == top["_len"])]
            if len(rest) > 1:
                LOGGER.warning("AMFI resolve(%r) is ambiguous — %d equally-specific "
                               "schemes (%s); taking %s (%s).", query, len(rest),
                               ", ".join(rest["amfi_code"].astype(str).head(4)),
                               top["scheme_name"], top["amfi_code"])
        return ranked.iloc[0]

    def health(self) -> bool:
        return self.live or (CACHE / "amfi_master.parquet").exists()


class CapBandAdapter:
    """
    AMFI half-yearly large/mid/small classification. This file is what makes the
    SEBI 80%/65% true-to-label checks real; without it they are decorative.
    Auto-download is best-effort (AMFI serves it as an XLSX behind a page), so a
    manual drop-in path is supported and the pipeline says so explicitly.
    """

    LOCAL = CACHE / "amfi_cap_classification.csv"
    XLSX_BASE = "https://www.amfiindia.com/Themes/Theme1/downloads/"
    XLSX_BASE_ALT = "https://portal.amfiindia.com/spages/"
    LINK_RE = re.compile(r'href="([^"]*AverageMarketCapitalization(\d{1,2}[A-Za-z]{3}\d{4})\.xlsx)"', re.I)
    ALLOWED_HOSTS = frozenset({"amfiindia.com", "www.amfiindia.com", "portal.amfiindia.com"})

    @classmethod
    def _allowed(cls, url: str) -> bool:
        """A scraped href host is untrusted input — only fetch/parse AMFI over https."""
        p = urlparse(url)
        return p.scheme == "https" and p.netloc.lower() in cls.ALLOWED_HOSTS

    def _discover_url(self) -> Optional[str]:
        """Scrape AMFI_CAPLIST_PAGE for the most recently dated xlsx link (full href, so a
        host change on AMFI's side — seen in practice — doesn't break discovery)."""
        html = _get_text(AMFI_CAPLIST_PAGE, timeout=30)
        if not html:
            return None
        dated = []
        for href, datestr in self.LINK_RE.findall(html):
            try:
                dated.append((datetime.strptime(datestr, "%d%b%Y").date(),
                              urljoin(AMFI_CAPLIST_PAGE, href)))
            except ValueError:
                continue
        if not dated:
            return None
        dated.sort(key=lambda x: x[0], reverse=True)
        return dated[0][1]

    def _fallback_urls(self) -> List[str]:
        """Construct candidate URLs from the most recent completed half-year end
        (30-Jun/31-Dec), skipping a period too recent to plausibly be published yet
        (AMFI publish lag), tried against both known download hosts."""
        today = date.today()
        candidates = sorted({c for c in (date(today.year, 12, 31), date(today.year, 6, 30),
                                         date(today.year - 1, 12, 31), date(today.year - 1, 6, 30))
                             if c <= today}, reverse=True)
        period = next((c for c in candidates if (today - c).days >= 45), candidates[-1])
        fname = f"AverageMarketCapitalization{period.day}{period.strftime('%b')}{period.year}.xlsx"
        return [self.XLSX_BASE + fname, self.XLSX_BASE_ALT + fname]

    def fetch(self) -> bool:
        """Download the latest AMFI cap-list xlsx and write a clean symbol,company,cap_band
        CSV to self.LOCAL (raw columns are NOT dumped — the loader's "cap" substring sniff
        would match the Market-Cap NUMBER columns first and break)."""
        discovered = self._discover_url()
        urls = ([discovered] if discovered else []) + self._fallback_urls()
        url, raw = None, None
        for candidate in urls:
            if not candidate or not self._allowed(candidate):
                if candidate:
                    LOGGER.warning("Skipping non-AMFI cap-list URL: %s", candidate)
                continue
            raw = _get_bytes(candidate, referer=AMFI_CAPLIST_PAGE)
            if raw is not None:
                url = candidate
                break
        if raw is None:
            return False
        try:
            (CACHE / Path(url).name).write_bytes(raw)
            df = pd.read_excel(CACHE / Path(url).name, header=1, engine="openpyxl")
            cat_col = next(c for c in df.columns if "categor" in c.lower())

            def clean_sym(col: str) -> pd.Series:
                s = df[col].astype(str).str.strip()
                return s.mask(s.str.lower().isin(["", "-", "nan", "none"]))

            sym = clean_sym("NSE Symbol").fillna(clean_sym("BSE Symbol"))
            band = (df[cat_col].astype(str).str.extract(r"(?i)(large|mid|small)", expand=False)
                    .str.lower().str.capitalize() + " Cap")
            company = df["Company name"].astype(str).str.strip()
            out = pd.DataFrame({"symbol": sym, "company": company, "cap_band": band})
            # Keep rows lacking an exchange symbol when the company name is usable —
            # band_lookup() keys on company too, so dropping them loses ~180 small-caps.
            good_company = ~company.str.lower().isin(["", "nan", "none"])
            out = out[good_company & out["cap_band"].isin(["Large Cap", "Mid Cap", "Small Cap"])]
            out.to_csv(self.LOCAL, index=False)
            LOGGER.info("Cap-band list fetched: %d symbols from %s", len(out), url)
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Cap-band xlsx fetch/parse failed (%s): %s", url, exc)
            return False

    def ensure(self) -> None:
        """Self-healing: fetch-and-build if the local CSV is absent. Never raises — a
        failure here must still let the pipeline run offline / accept a manual drop-in."""
        if self.LOCAL.exists():
            return
        try:
            if not self.fetch():
                LOGGER.warning("Cap-band auto-fetch failed; manual drop-in required "
                               "(see bootstrap.py [5/5]).")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Cap-band auto-fetch raised %s; degrading gracefully", exc)

    def load(self) -> Optional[pd.DataFrame]:
        self.ensure()
        if self.LOCAL.exists():
            df = pd.read_csv(self.LOCAL)
            cols = {c.lower().strip(): c for c in df.columns}
            sym = next((cols[c] for c in cols if "symbol" in c or "ticker" in c), None)
            nm = next((cols[c] for c in cols if "company" in c or "name" in c), None)
            cat = next((cols[c] for c in cols if "cap" in c or "categor" in c or "class" in c), None)
            if cat is None:
                LOGGER.error("cap list present but no category column found")
                return None
            out = pd.DataFrame({
                "symbol": df[sym].astype(str).str.strip() if sym else None,
                "company": df[nm].astype(str).str.strip() if nm else None,
                "cap_band": df[cat].astype(str).str.lower().str.extract(
                    r"(large|mid|small)", expand=False)})
            return out.dropna(subset=["cap_band"])
        return None

    def band_lookup(self) -> Dict[str, str]:
        df = self.load()
        if df is None:
            return {}
        out: Dict[str, str] = {}
        for _, r in df.iterrows():
            if isinstance(r.get("symbol"), str) and r["symbol"] not in ("nan", "None"):
                out[r["symbol"].upper()] = r["cap_band"]
            if isinstance(r.get("company"), str):
                out[r["company"].upper()] = r["cap_band"]
        return out

    def health(self) -> bool:
        return self.LOCAL.exists()


class YFinanceAdapter:
    """NSE stock history for Agent B's underlying-holdings backtest."""

    def __init__(self) -> None:
        self.live = False

    def prices(self, tickers: Sequence[str], start: str) -> Optional[pd.DataFrame]:
        key = CACHE / f"px_{abs(hash((tuple(sorted(tickers)), start)))}.parquet"
        if key.exists():
            return pd.read_parquet(key)
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            LOGGER.warning("yfinance not installed: pip install yfinance")
            return None
        try:
            syms = [t if t.endswith((".NS", ".BO")) else f"{t}.NS" for t in tickers]
            df = yf.download(syms, start=start, progress=False, auto_adjust=True)
            px = df["Close"] if "Close" in df else df
            if isinstance(px, pd.Series):
                px = px.to_frame()
            px = px.dropna(how="all")
            if px.empty:
                return None
            px.to_parquet(key)
            self.live = True
            return px
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("yfinance fetch failed: %s", exc)
            return None

    def health(self) -> bool:
        return self.live or any(CACHE.glob("px_*.parquet"))


class AnthropicNewsAgent:
    """
    AGENT D, live. Calls the Messages API with the server-side web_search tool
    and forces a strict JSON envelope so the output is parseable, not prose.
    Requires ANTHROPIC_API_KEY. Falls back to neutral (no invented news) if absent
    — a sentiment score must never be fabricated.
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.live = False

    def research(self, amc: str, manager: str, sectors: Sequence[str]) -> Dict[str, Any]:
        if not self.api_key:
            LOGGER.warning("ANTHROPIC_API_KEY unset — Agent D returns NEUTRAL "
                           "(no news is better than invented news)")
            return dict(net_sentiment=0.0, n_items=0, red_flags=[], items=[],
                        live=False, note="no API key")
        r = _requests()
        if r is None:
            return dict(net_sentiment=0.0, n_items=0, red_flags=[], items=[], live=False)
        prompt = (
            f"Search for news from the last 90 days about the Indian mutual fund "
            f"AMC '{amc}', fund manager '{manager}', and regulatory/sector news for "
            f"these sectors: {', '.join(sectors)}. Focus on SEBI actions, manager "
            f"exits, AMC governance, and sector stress.\n\n"
            "Respond with ONLY a JSON object, no preamble, no markdown fences:\n"
            '{"items":[{"headline":str,"entity":str,"date":"YYYY-MM-DD",'
            '"tone":-1|0|1,"regulatory_flag":bool}],"net_sentiment":float,'
            '"summary":str}\n'
            "net_sentiment in [-1,1]. If you find nothing, return empty items and "
            "net_sentiment 0. Do not invent headlines."
        )
        try:
            resp = r.post(ANTHROPIC_MESSAGES, timeout=90, headers={
                "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"},
                json={"model": self.MODEL, "max_tokens": 2000,
                      "messages": [{"role": "user", "content": prompt}],
                      "tools": [{"type": "web_search_20250305", "name": "web_search"}]})
            resp.raise_for_status()
            blocks = resp.json().get("content", [])
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else {}
            items = data.get("items", [])
            self.live = True
            return dict(net_sentiment=float(data.get("net_sentiment", 0.0)),
                        n_items=len(items),
                        red_flags=[i["headline"] for i in items if i.get("regulatory_flag")],
                        items=items, summary=data.get("summary", ""), live=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Agent D live call failed (%s) — returning NEUTRAL", exc)
            return dict(net_sentiment=0.0, n_items=0, red_flags=[], items=[], live=False)

    def health(self) -> bool:
        return bool(self.api_key)


# ==============================================================================
# DATA READINESS REPORT — drives the orchestrator's honest confidence score
# ==============================================================================
def readiness() -> pd.DataFrame:
    checks = [
        ("AMFI scheme master", AMFIRegistryLive().health(), "auto (bootstrap.py)", "Agent A"),
        ("NAV history (mfapi)", MFAPIAdapter().health(), "auto (bootstrap.py)", "Agents A/C"),
        ("Cap-band list", CapBandAdapter().health(), "auto, else 1 manual download", "SEBI true-to-label"),
        ("Stock prices (yfinance)", YFinanceAdapter().health(), "auto (bootstrap.py)", "Agent B"),
        ("News (ANTHROPIC_API_KEY)", AnthropicNewsAgent().health(), "paste key in .env", "Agent D"),
        ("Historical disclosures", (CACHE / "disclosures").exists(), "MANUAL — see plan", "Agent B (core)"),
    ]
    df = pd.DataFrame(checks, columns=["source", "ready", "how", "consumer"])
    df["status"] = np.where(df["ready"], "READY", "MISSING")
    return df[["source", "status", "how", "consumer"]]


# ==============================================================================
# SELFTEST — NAVAll header/banner discrimination (pure, no network)
# ==============================================================================
def _selftest_navall() -> None:
    """Regression test for the AMFI header-vs-AMC-banner split.

    The original rule ("any line with '(' and ')' is a category header") mis-read
    `IL&FS Mutual Fund (IDF)` — an AMC whose COMPANY NAME contains parentheses — as
    a category. That bogus category then persisted to every scheme after it, and the
    AMC column silently inherited whichever firm was listed before it. On the live
    file that mislabelled 2,544 rows and misattributed 12 IL&FS schemes to ICICI
    Prudential.
    """
    sample = "\n".join([
        "Scheme Code;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date",
        "",
        "Open Ended Schemes(Equity Scheme - Large Cap Fund)",
        "",
        "Alpha Mutual Fund",
        "100001;INF000000001;-;Alpha Large Cap Fund - Direct - Growth;123.45;01-Jul-2026",
        "",
        "Close Ended Schemes(Income)",
        "",
        "Beta Mutual Fund",
        "100002;INF000000002;-;Beta Income Fund - Growth;10.11;01-Jul-2026",
        "",
        "IL&FS Mutual Fund (IDF)",                       # AMC banner, NOT a category
        "100003;INF000000003;-;IL&FS Infrastructure Debt Fund Series 1A - Growth;9.87;01-Jul-2026",
        "",
        "Gamma Mutual Fund",                              # follows the IDF block
        "100004;INF000000004;-;Gamma Income Fund - Growth;12.34;01-Jul-2026",
    ])

    rows, category, amc = [], None, None
    for line in sample.splitlines():
        line = line.strip()
        if not line or line.startswith("Scheme Code"):
            continue
        m = AMFIRegistryLive.LINE.match(line)
        if m:
            rows.append(dict(code=m.group("code"), name=m.group("name").strip(),
                             category_raw=category, amc=amc))
        elif AMFIRegistryLive.CATEGORY_HEADER.match(line):
            category = line
        elif ";" not in line:
            amc = line
    by = {r["code"]: r for r in rows}

    assert by["100001"]["category_raw"] == "Open Ended Schemes(Equity Scheme - Large Cap Fund)"
    assert by["100001"]["amc"] == "Alpha Mutual Fund"

    # The paren-bearing AMC must be read as an AMC, never as a category...
    assert by["100003"]["amc"] == "IL&FS Mutual Fund (IDF)", by["100003"]
    assert "IL&FS" not in str(by["100003"]["category_raw"]), by["100003"]
    # ...and its schemes keep the enclosing (legacy, debt) header — which mf_universe
    # then refuses, so they can never reach the equity model.
    assert by["100003"]["category_raw"] == "Close Ended Schemes(Income)", by["100003"]

    # Schemes of the NEXT AMC must retain that same real category. Suppressing the
    # category at a paren-bearing banner would strip it from these — on the live file
    # that would wrongly blank 2,532 legitimate close-ended income schemes.
    assert by["100004"]["category_raw"] == "Close Ended Schemes(Income)", by["100004"]
    assert by["100004"]["amc"] == "Gamma Mutual Fund"

    # No AMC banner may ever be mistaken for a category.
    assert not AMFIRegistryLive.CATEGORY_HEADER.match("IL&FS Mutual Fund (IDF)")
    assert not AMFIRegistryLive.CATEGORY_HEADER.match("Alpha Mutual Fund")
    assert AMFIRegistryLive.CATEGORY_HEADER.match("Open Ended Schemes(Equity Scheme - ELSS)")
    assert AMFIRegistryLive.CATEGORY_HEADER.match(
        "Open Ended Schemes(Exchange Traded Funds (ETFs) - Equity ETF)")
    print("[selftest] NAVAll: paren-bearing AMC banner read as AMC (not category); "
          "following AMCs keep their real category — PASS")
    print("[selftest] PASS — mf_datasources NAVAll parsing")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="mf_datasources checks")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest_navall()
