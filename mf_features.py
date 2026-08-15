"""
====================================================================================
mf_features.py — Phase B as-of feature engine (docs/phase_b_design.md §2)
====================================================================================
Every feature is a function of NAV data at dates <= the anchor t ONLY (the label
may look at t..t+3y; features may not). Enforcement is mechanical: every
computation here slices its input series to `.loc[:t]` (or an equivalent
searchsorted bound) before touching it, and `--selftest` perturbs all post-t
NAVs by +20% across the whole universe and asserts bit-identical features
(design §7.4).

Feature groups (design §2.1-2.4):
  own-NAV        cagr_1y/3y/5y, vol_1y/3y, sharpe_3y, sortino_3y, max_dd_3y,
                 current_dd, mom_6m, mom_12m_ex1m, rr3y_neg_share, age_years
  peer-relative  excess_1y/3y, rank_3y, beta_3y, te_3y, ir_3y, upcap_3y,
                 downcap_3y, excess_persist — all vs the SAME leave-one-out
                 peer pool the label benchmark uses (PeerProxyResolver), but
                 TRAILING (composite of peer NAV returns <= t), never forward
  sector context sector_mom_12m, sector_vol_1y, sector_rel_strength,
                 is_thematic (also the missingness gate) — thematic funds only
  cohort z       within-anchor cross-sectional z-scores of the level-type
                 features (defense against regime memorization, §2.4)
  missingness    has_1y/3y/5y, has_peer, has_persist indicators (young-fund
                 signal instead of silent imputation)

Strictness rule for windowed features: a *_1y/_3y/_5y feature is NaN unless the
history actually covers the window start under the same 10-day-staleness as-of
rule the label uses (design §2: "NaN + a paired missingness indicator").

Exclusions per §2.5: no current holdings, no TER/AUM, no fund-identity one-hots.

Output: mf_cache/phase_b/features.parquet, keyed (amfi_code, anchor).
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from mf_benchmarks import PeerProxyResolver, _asof
from mf_datasources import CACHE
from mf_labels import LABELS_PATH, load_manifest, load_nav_panel

OUT_DIR = CACHE / "phase_b"
FEATURES_PATH = OUT_DIR / "features.parquet"

TRADING_DAYS = 252
# Mirrors mf_pipeline.RISK_FREE_RATE_ANNUAL (not imported — heavy module).
RF_ANNUAL = 0.0675
RF_DAILY = (1.0 + RF_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

# Level-type features that additionally get a within-anchor z-score (§2.4).
LEVEL_FEATURES = [
    "cagr_1y",
    "cagr_3y",
    "cagr_5y",
    "mom_6m",
    "mom_12m_ex1m",
    "excess_1y",
    "excess_3y",
    "sector_mom_12m",
    "sector_rel_strength",
]


# ==============================================================================
# Windowed as-of primitives (all take the FULL series but slice to <= t first)
# ==============================================================================
def _trailing_cagr(series: pd.Series, t: pd.Timestamp, years: int) -> float:
    """Annualized trailing return over [t-years, t], date-based endpoints with
    the 10-day staleness rule on both ends; NaN if history doesn't cover it."""
    d1, v1 = _asof(series, t)
    if d1 is None or v1 is None or v1 <= 0:
        return np.nan
    d0, v0 = _asof(series, t - pd.DateOffset(years=years))
    if d0 is None or v0 is None or v0 <= 0:
        return np.nan
    yrs = (d1 - d0).days / 365.25
    if yrs < years - 0.1:
        return np.nan
    return float((v1 / v0) ** (1.0 / yrs) - 1.0)


def _simple_return(series: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Simple (non-annualized) return between as-of(t0) and as-of(t1)."""
    d1, v1 = _asof(series, t1)
    d0, v0 = _asof(series, t0)
    if d0 is None or d1 is None or v0 is None or v0 <= 0 or d0 >= d1:
        return np.nan
    return float(v1 / v0 - 1.0)


def _window_slice(series: pd.Series, t: pd.Timestamp, years: int) -> pd.Series | None:
    """Levels over [as-of(t-years), t]; None unless history covers the window
    start under the 10-day staleness rule."""
    d0, _ = _asof(series, t - pd.DateOffset(years=years))
    if d0 is None:
        return None
    win = series.loc[d0:t]
    return win if len(win) >= 3 else None


def _ann_vol(rets: np.ndarray) -> float:
    if rets.size < 3:
        return np.nan
    return float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS))


def _sharpe(rets: np.ndarray) -> float:
    if rets.size < 3:
        return np.nan
    excess = rets - RF_DAILY
    sd = np.std(excess, ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(excess.mean() / sd * math.sqrt(TRADING_DAYS))


def _sortino(rets: np.ndarray) -> float:
    if rets.size < 3:
        return np.nan
    excess = rets - RF_DAILY
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float(np.mean(np.square(downside))))
    if dd < 1e-12:
        return np.nan
    return float(excess.mean() / dd * math.sqrt(TRADING_DAYS))


def _ann_return(rets: np.ndarray) -> float:
    if rets.size == 0:
        return np.nan
    gross = float(np.prod(1.0 + rets))
    if gross <= 0:
        return np.nan
    return gross ** (TRADING_DAYS / rets.size) - 1.0


# ==============================================================================
# Composites (leave-one-out peer, per-sector, all-thematic)
# ==============================================================================
def _returns_frame(codes: list[str], nav_panel: dict[str, pd.Series]) -> pd.DataFrame:
    cols = {c: nav_panel[c].pct_change() for c in codes if c in nav_panel}
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1, sort=True)


def _compound(rets: pd.Series) -> pd.Series:
    rets = rets.dropna()
    return (1.0 + rets).cumprod()


class FeatureEngine:
    """Precomputes per-fund LOO peer composites, sector composites and aligned
    fund-vs-composite return frames from a NAV panel; `features(code, t)` then
    reads only data <= t out of them (composites are built causally from daily
    returns, so slicing the precomputed series at t is identical to building
    them from a panel truncated at t — verified by --selftest)."""

    MIN_LOO_PEERS = 2  # same floor as the label benchmark (§1.3)
    MIN_JOINT_OBS = 60  # QuantEngine convention for beta/TE
    MIN_JOINT_YEARS = 2.75  # same coverage floor as forward_return()
    MIN_PERSIST_CHECKPOINTS = 8

    def __init__(self, manifest: pd.DataFrame, nav_panel: dict[str, pd.Series]):
        self.manifest = manifest.set_index("amfi_code")
        self.nav_panel = nav_panel
        self.resolver = PeerProxyResolver(manifest)

        # ---- LOO peer composites (trailing twin of the label benchmark) -----
        self.comp_level: dict[str, pd.Series] = {}
        self.joint: dict[str, pd.DataFrame] = {}  # fund vs composite daily returns
        for gkey, codes in self.resolver.groups.items():
            rets = _returns_frame(codes, nav_panel)
            if rets.empty:
                continue
            total = rets.sum(axis=1)
            count = rets.notna().sum(axis=1)
            for code in rets.columns:
                own = rets[code]
                loo_n = count - own.notna().astype(int)
                loo = (total - own.fillna(0.0)) / loo_n.where(loo_n >= self.MIN_LOO_PEERS)
                loo = loo.dropna()
                if loo.empty:
                    continue
                self.comp_level[code] = _compound(loo)
                joint = pd.concat({"fund": own, "comp": loo}, axis=1, join="inner").dropna()
                self.joint[code] = joint

        # ---- sector / all-thematic composites (§2.3) -------------------------
        them = manifest[manifest["category"] == "Sectoral/Thematic"]
        self.sector_level: dict[str, pd.Series] = {}
        for sector, grp in them.groupby("sector"):
            rets = _returns_frame(list(grp["amfi_code"]), nav_panel)
            if rets.empty:
                continue
            self.sector_level[sector] = _compound(rets.mean(axis=1))
        all_them = _returns_frame(list(them["amfi_code"]), nav_panel)
        self.thematic_level = _compound(all_them.mean(axis=1)) if not all_them.empty else None

    # --------------------------------------------------------------------------
    def group_key(self, code: str) -> tuple:
        return self.resolver._group_key(code)

    def _relative_stats(self, code: str, t: pd.Timestamp) -> dict[str, float]:
        out = dict(beta_3y=np.nan, te_3y=np.nan, ir_3y=np.nan, upcap_3y=np.nan, downcap_3y=np.nan)
        joint = self.joint.get(code)
        if joint is None:
            return out
        d_lo = t - pd.DateOffset(years=3)
        win = joint.loc[d_lo:t]
        if len(win) < self.MIN_JOINT_OBS:
            return out
        if (t - win.index[0]).days < self.MIN_JOINT_YEARS * 365.25:
            return out
        f = win["fund"].to_numpy()
        b = win["comp"].to_numpy()
        var_b = np.var(b, ddof=1)
        if var_b > 1e-12:
            out["beta_3y"] = float(np.cov(f, b, ddof=1)[0, 1] / var_b)
        active = f - b
        te = float(np.std(active, ddof=1) * math.sqrt(TRADING_DAYS))
        out["te_3y"] = te
        if te > 1e-12:
            out["ir_3y"] = float((_ann_return(f) - _ann_return(b)) / te)
        up, down = b > 0, b < 0

        # up/down-capture share the same geo-mean(fund)/geo-mean(bench) form over
        # the mask'd periods; factor it out so the two only differ by the mask.
        # Guards are preserved exactly: <=10 obs or |den|<=1e-12 => NaN (the seed).
        def _capture(mask) -> float:
            if mask.sum() <= 10:
                return np.nan
            num = float(np.prod(1.0 + f[mask]) ** (1.0 / mask.sum()) - 1.0)
            den = float(np.prod(1.0 + b[mask]) ** (1.0 / mask.sum()) - 1.0)
            return num / den if abs(den) > 1e-12 else np.nan

        out["upcap_3y"] = _capture(up)
        out["downcap_3y"] = _capture(down)
        return out

    def _excess_persist(self, series: pd.Series, comp: pd.Series | None, t: pd.Timestamp) -> float:
        """Share of trailing 12 quarterly checkpoints with trailing-1y excess > 0."""
        if comp is None:
            return np.nan
        hits = []
        for k in range(12):
            q = t - pd.DateOffset(months=3 * k)
            f = _trailing_cagr(series, q, 1)
            c = _trailing_cagr(comp, q, 1)
            if np.isfinite(f) and np.isfinite(c):
                hits.append(f - c > 0)
        if len(hits) < self.MIN_PERSIST_CHECKPOINTS:
            return np.nan
        return float(np.mean(hits))

    # --------------------------------------------------------------------------
    def features(self, code: str, t: pd.Timestamp) -> dict[str, float]:
        series = self.nav_panel[code].loc[:t]
        row = self.manifest.loc[code]
        f: dict[str, float] = {}

        # ---- own-NAV (§2.1) ---------------------------------------------------
        f["cagr_1y"] = _trailing_cagr(series, t, 1)
        f["cagr_3y"] = _trailing_cagr(series, t, 3)
        f["cagr_5y"] = _trailing_cagr(series, t, 5)

        win1 = _window_slice(series, t, 1)
        win3 = _window_slice(series, t, 3)
        r1 = win1.pct_change().to_numpy()[1:] if win1 is not None else np.array([])
        r3 = win3.pct_change().to_numpy()[1:] if win3 is not None else np.array([])
        f["vol_1y"] = _ann_vol(r1)
        f["vol_3y"] = _ann_vol(r3)
        f["sharpe_3y"] = _sharpe(r3)
        f["sortino_3y"] = _sortino(r3)
        if win3 is not None:
            dd = win3 / win3.cummax() - 1.0
            f["max_dd_3y"] = float(dd.min())
            f["current_dd"] = float(dd.iloc[-1])
        else:
            f["max_dd_3y"] = np.nan
            f["current_dd"] = np.nan

        f["mom_6m"] = _simple_return(series, t - pd.DateOffset(months=6), t)
        f["mom_12m_ex1m"] = _simple_return(series, t - pd.DateOffset(months=12), t - pd.DateOffset(months=1))

        vals = series.to_numpy()
        w = 3 * TRADING_DAYS
        if len(vals) > w:
            ann = (vals[w:] / vals[:-w]) ** (1.0 / 3.0) - 1.0
            f["rr3y_neg_share"] = float((ann < 0).mean())
        else:
            f["rr3y_neg_share"] = np.nan

        f["age_years"] = float((t - series.index[0]).days / 365.25) if len(series) else np.nan

        # ---- peer-relative (§2.2) ----------------------------------------------
        comp_full = self.comp_level.get(code)
        comp = comp_full.loc[:t] if comp_full is not None else None
        c1 = _trailing_cagr(comp, t, 1) if comp is not None else np.nan
        c3 = _trailing_cagr(comp, t, 3) if comp is not None else np.nan
        f["excess_1y"] = f["cagr_1y"] - c1
        f["excess_3y"] = f["cagr_3y"] - c3
        f.update(self._relative_stats(code, t))
        f["excess_persist"] = self._excess_persist(series, comp, t)

        # ---- sector context (§2.3) ----------------------------------------------
        thematic = row["category"] == "Sectoral/Thematic"
        f["is_thematic"] = float(thematic)
        f["sector_mom_12m"] = np.nan
        f["sector_vol_1y"] = np.nan
        f["sector_rel_strength"] = np.nan
        if thematic:
            lvl_full = self.sector_level.get(row["sector"])
            if lvl_full is not None:
                lvl = lvl_full.loc[:t]
                f["sector_mom_12m"] = _simple_return(lvl, t - pd.DateOffset(months=12), t)
                swin = _window_slice(lvl, t, 1)
                if swin is not None:
                    f["sector_vol_1y"] = _ann_vol(swin.pct_change().to_numpy()[1:])
                if self.thematic_level is not None:
                    thm = self.thematic_level.loc[:t]
                    thm_mom = _simple_return(thm, t - pd.DateOffset(months=12), t)
                    f["sector_rel_strength"] = f["sector_mom_12m"] - thm_mom

        # ---- missingness indicators (§2) ----------------------------------------
        f["has_1y"] = float(np.isfinite(f["cagr_1y"]))
        f["has_3y"] = float(np.isfinite(f["cagr_3y"]))
        f["has_5y"] = float(np.isfinite(f["cagr_5y"]))
        f["has_peer"] = float(np.isfinite(f["excess_1y"]))
        f["has_persist"] = float(np.isfinite(f["excess_persist"]))
        return f


# ==============================================================================
# Panel assembly (adds cross-sectional rank_3y + within-anchor z-scores, §2.4)
# ==============================================================================
def assemble_panel(engine: FeatureEngine, pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs: DataFrame with amfi_code + anchor. Returns the full feature panel."""
    rows = []
    for code, t in pairs[["amfi_code", "anchor"]].itertuples(index=False):
        t = pd.Timestamp(t)
        rec = dict(amfi_code=code, anchor=t)
        rec.update(engine.features(code, t))
        rows.append(rec)
    panel = pd.DataFrame(rows)

    # rank of trailing cagr_3y within (anchor, peer pool) — same pool as the label
    gkeys = {c: str(engine.group_key(c)) for c in panel["amfi_code"].unique()}
    panel["_gkey"] = panel["amfi_code"].map(gkeys)
    pool = panel.groupby(["_gkey", "anchor"])["cagr_3y"]
    panel["rank_3y"] = pool.rank(pct=True).where(pool.transform("count") >= 3)
    panel = panel.drop(columns="_gkey")

    # within-anchor cross-sectional z-scores across ALL funds at that anchor
    for col in LEVEL_FEATURES:
        grp = panel.groupby("anchor")[col]
        mean = grp.transform("mean")
        std = grp.transform("std")
        cnt = grp.transform("count")
        z = (panel[col] - mean) / std.where((std > 1e-12) & (cnt >= 5))
        panel[f"z_{col}"] = z
    return panel


def build_features(labels_path=LABELS_PATH) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    pairs = labels[["amfi_code", "anchor"]].drop_duplicates()
    manifest = load_manifest()
    nav_panel = load_nav_panel(manifest)
    engine = FeatureEngine(manifest, nav_panel)
    panel = assemble_panel(engine, pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_PATH, index=False)
    return panel


# ==============================================================================
# Anti-lookahead selftest (design §7.4): perturb ALL post-t NAVs by +20% and
# -20% across the whole universe; every feature at anchor t must be identical.
# ==============================================================================
def run_asof_selftest(anchors=("2016-03-31", "2018-06-29", "2021-12-31")) -> bool:
    manifest = load_manifest()
    nav_panel = load_nav_panel(manifest)
    ok = True
    for anchor in anchors:
        t = pd.Timestamp(anchor)
        codes = [c for c in manifest["amfi_code"] if c in nav_panel and len(nav_panel[c].loc[:t]) >= 5]
        pairs = pd.DataFrame({"amfi_code": codes, "anchor": t})
        base = assemble_panel(FeatureEngine(manifest, nav_panel), pairs)
        for bump in (1.2, 0.8):
            perturbed = {}
            for code, s in nav_panel.items():
                s2 = s.copy()
                s2[s2.index > t] *= bump
                perturbed[code] = s2
            pert = assemble_panel(FeatureEngine(manifest, perturbed), pairs)
            same = (
                base.drop(columns=["amfi_code", "anchor"])
                .round(12)
                .equals(pert.drop(columns=["amfi_code", "anchor"]).round(12))
            )
            print(f"anchor {anchor} bump {bump}: {'IDENTICAL — no lookahead' if same else 'MISMATCH — LOOKAHEAD BUG'}")
            if not same:
                bad = [
                    c
                    for c in base.columns
                    if c not in ("amfi_code", "anchor") and not base[c].round(12).equals(pert[c].round(12))
                ]
                print("  differing columns:", bad)
                ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selftest", action="store_true", help="anti-lookahead test: perturb post-t NAVs, assert identical features"
    )
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if run_asof_selftest() else 1)
    panel = build_features()
    n_num = panel.drop(columns=["amfi_code", "anchor"]).shape[1]
    print(f"features: {panel.shape[0]} rows x {n_num} feature columns -> {FEATURES_PATH}")
    print(panel.drop(columns=["amfi_code", "anchor"]).notna().mean().round(3).to_string())
