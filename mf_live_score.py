"""
====================================================================================
mf_live_score.py — one-anchor live scoring of the cohort_q1 model (to-do #4)
====================================================================================
Training scores every fund at all 144 monthly anchors (mf_labels.ANCHOR_GRID) to
build a labeled dataset. Live scoring needs only ONE fund, TODAY — recomputing
the full 144-anchor grid for that is ~100x more work than necessary. This module
reuses mf_features.assemble_panel VERBATIM with a one-anchor `pairs` frame
instead: same feature math, same cross-sectional rank_3y/z_* recipe, just one
anchor touched instead of 144.

WHY THE PANEL IS UNIVERSE-WIDE, NOT COHORT-RESTRICTED: rank_3y is cohort-local
(grouped by (cohort_key, anchor)) but z_* is grouped by anchor ACROSS THE WHOLE
MANIFEST (mf_features.py assemble_panel). Restricting the live panel to only the
target fund's cohort peers would compute z_* over ~5-20 funds instead of the
~136-fund training universe — a silent distribution shift the model was never
calibrated for. So build_live_panel always spans the full manifest at one anchor;
the ~100x saving is 1 anchor vs 144, not a narrower universe.

HONESTY GATES (never fabricate a probability):
  NOT_IN_UNIVERSE — fund isn't in the trained universe manifest, or has no cached
                    NAV at all. No trained cohort membership is defined for it.
  STALE_NAV       — the fund's latest cached NAV is more than STALE_MAX_DAYS
                    behind `today`. Scoring off month-old data and presenting it
                    as current would fabricate freshness.
  THIN_COHORT     — fewer than mf_labels.COHORT_MIN_SIZE cohort peers have a
                    valid as-of NAV at the scoring anchor. "Top quartile of a
                    3-fund cohort" is not a well-posed question (mirrors the
                    exact gate mf_labels.add_cohort_targets used at training
                    time to mark a row ineligible rather than guess a target).
  OK              — probability + cohort_percentile, always alongside
                    signal_context (holdout AUC ~0.578, lift ~1.76x — weak) and
                    imputed_features (which inputs fell back to a training
                    median), never presented as a standalone verdict.

ANCHOR CHOICE: `t` = the target fund's own latest cached NAV date, capped at
`today` — NOT a per-fund-independent anchor for every cohort member. Cross-
sectional features compare funds AT THE SAME t by construction (that is what
every training anchor did); using each peer's own latest date would compare
funds through different endpoints, a comparison training never made. A stale
peer's features go NaN under the existing 10-day _asof tolerance and drop out
of the rank/z pools exactly as a dead fund would have at a 2019 training anchor
— no new tolerance logic needed here.

KNOWN, DELIBERATE TRAIN/SERVE DELTA: training's cross-sectional pool at any
anchor T only contained funds that survived >= 2.75y past T (mf_labels.
forward_return's requirement for a valid label). The live pool at t=today
cannot filter on a future that hasn't happened — it includes funds that would
have been excluded from a comparable training anchor. This is irreducible
(survivorship is unknowable), the effect on a ~136-fund pool's mean/std is
small, and it is documented here rather than patched with a fake filter.
====================================================================================
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mf_benchmarks import _asof
from mf_features import FeatureEngine, assemble_panel
from mf_infer import CohortInferencer, DEFAULT_TARGET, load_default
from mf_labels import COHORT_MIN_SIZE, load_manifest, load_nav_panel

STALE_MAX_DAYS = 30

_STATUSES = ("OK", "THIN_COHORT", "STALE_NAV", "NOT_IN_UNIVERSE")


def _gap_result(status: str, target: str, **extra: Any) -> Dict[str, Any]:
    """Same-shaped dict for every non-OK status — callers can read `probability`
    without a KeyError regardless of which gate fired."""
    assert status in _STATUSES
    out = dict(status=status, target=target, probability=None, cohort_percentile=None,
              cohort_key=None, cohort_n=0, universe_n=0, anchor=None, staleness_days=None,
              signal_context=None, imputed_features=[], cohort_codes=[], features=None,
              note=None)
    out.update(extra)
    return out


def build_live_panel(manifest: pd.DataFrame, nav_panel: Dict[str, pd.Series],
                     t: pd.Timestamp, engine: Optional[FeatureEngine] = None,
                     codes: Optional[List[str]] = None
                     ) -> tuple[FeatureEngine, pd.DataFrame]:
    """One-anchor feature panel over the full universe (see module docstring for
    why it must stay universe-wide). `codes` overrides membership — used ONLY by
    --selftest to replay a historical training pool exactly; production always
    passes codes=None (fresh as-of-t membership: any fund with a valid NAV at t)."""
    if codes is None:
        codes = [c for c in manifest["amfi_code"]
                 if c in nav_panel and _asof(nav_panel[c], t)[0] is not None]
    if engine is None:
        engine = FeatureEngine(manifest, nav_panel)
    pairs = pd.DataFrame({"amfi_code": codes, "anchor": t})
    return engine, assemble_panel(engine, pairs)


def score_live(code: str, *,
              manifest: Optional[pd.DataFrame] = None,
              nav_panel: Optional[Dict[str, pd.Series]] = None,
              inferencer: Optional[CohortInferencer] = None,
              engine: Optional[FeatureEngine] = None,
              today: Optional[pd.Timestamp] = None,
              target: str = DEFAULT_TARGET) -> Dict[str, Any]:
    """Score one fund against the cohort model at the latest anchor its own NAV
    data supports. See module docstring for the honesty gates and anchor choice."""
    manifest = manifest if manifest is not None else load_manifest()
    nav_panel = nav_panel if nav_panel is not None else load_nav_panel(manifest)
    inferencer = inferencer if inferencer is not None else load_default()
    # No module-level engine cache: a caller (e.g. MasterOrchestrator) that scores
    # many funds should build ONE FeatureEngine and pass it in explicitly (a plain
    # strong reference, tied to that caller's own lifetime) rather than rely on an
    # id()-keyed global here, which risks silently serving a stale engine if
    # CPython reuses a freed manifest/nav_panel's id() across short-lived callers.
    engine = engine if engine is not None else FeatureEngine(manifest, nav_panel)
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now().normalize()

    manifest_idx = manifest.set_index("amfi_code")
    series = nav_panel.get(code)
    if code not in manifest_idx.index or series is None or series.dropna().empty:
        return _gap_result("NOT_IN_UNIVERSE", target,
                           note="Fund not in the trained universe manifest, or no cached NAV.")

    last_dt = series.dropna().index.max()
    t = min(last_dt, today)
    staleness_days = int((today - t).days)
    if staleness_days > STALE_MAX_DAYS:
        return _gap_result("STALE_NAV", target, anchor=str(t.date()), staleness_days=staleness_days,
                           note=f"Latest cached NAV is {staleness_days}d behind today "
                                f"(> {STALE_MAX_DAYS}d tolerance) — refresh the cache before scoring.")

    engine, panel = build_live_panel(manifest, nav_panel, t, engine=engine)
    panel_codes = set(panel["amfi_code"])
    if code not in panel_codes:
        return _gap_result("STALE_NAV", target, anchor=str(t.date()), staleness_days=staleness_days,
                           note="Fund's own as-of NAV lookup failed building the live panel.")

    cohort_key = engine.group_key(code)
    cohort_present = [c for c in engine.resolver.groups[cohort_key] if c in panel_codes]
    if len(cohort_present) < COHORT_MIN_SIZE:
        return _gap_result("THIN_COHORT", target, cohort_key=str(cohort_key),
                           cohort_n=len(cohort_present), universe_n=len(panel),
                           anchor=str(t.date()), staleness_days=staleness_days,
                           note=f"Only {len(cohort_present)} cohort peers have a valid NAV at this "
                                f"anchor (< {COHORT_MIN_SIZE}) — 'top quartile of cohort' is not "
                                f"well-posed for a cohort this thin.")

    meta_cols = ["amfi_code", "anchor"]
    cohort_panel = panel[panel["amfi_code"].isin(cohort_present)].set_index("amfi_code")
    probs = {c: inferencer.predict(row.drop(meta_cols[1:]).to_dict(), target)
            for c, row in cohort_panel.iterrows()}
    cohort_percentile = float(pd.Series(probs).rank(pct=True)[code])

    target_features = cohort_panel.loc[code].drop(meta_cols[1:]).to_dict()
    imputed = [c for c in inferencer.features
              if not (isinstance(target_features.get(c), (int, float))
                      and np.isfinite(target_features.get(c)))]

    return dict(status="OK", target=target, probability=probs[code],
               cohort_percentile=cohort_percentile, cohort_key=str(cohort_key),
               cohort_n=len(cohort_present), universe_n=len(panel),
               anchor=str(t.date()), staleness_days=staleness_days,
               signal_context=inferencer.signal_context(target),
               imputed_features=imputed,
               # Frozen for the prediction ledger (mf_ledger.py): cohorts reshuffle
               # over time (manifest edits, closures), so realizing this prediction
               # years from now must rank against THIS exact peer set, not whichever
               # peers happen to still be in the cohort at realization time — else
               # the realized outcome would silently be a different, redefined label.
               cohort_codes=sorted(cohort_present),
               # Frozen exact feature vector — the only snapshot of this live anchor
               # that will ever exist (unlike training anchors, never written to
               # features.parquet); NaN -> None to stay JSON-safe for the ledger.
               features={c: (float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None)
                        for c, v in target_features.items()},
               note="weak validated signal (AUC ~0.578, lift ~1.76x) — "
                    "supporting datapoint, never a verdict")


# ==============================================================================
# SELFTEST — feature parity, prediction parity, anti-lookahead, honesty gates.
# All three replay against artifacts already on disk; no retrain needed.
# ==============================================================================
def _selftest() -> None:
    manifest = load_manifest()
    nav_panel = load_nav_panel(manifest)
    feats = pd.read_parquet("mf_cache/phase_b/features.parquet")
    labels = pd.read_parquet("mf_cache/phase_b/labels.parquet")

    T = feats["anchor"].max()
    codes_T = sorted(labels.loc[labels["anchor"] == T, "amfi_code"].unique())
    assert len(codes_T) >= COHORT_MIN_SIZE, "test anchor too thin — pick another"

    # ---- Test A: feature parity vs the stored training panel -----------------
    _, live_panel = build_live_panel(manifest, nav_panel, T, codes=codes_T)
    stored = feats[feats["anchor"] == T]
    live_sorted = live_panel.sort_values("amfi_code").reset_index(drop=True)
    stored_sorted = stored[live_sorted.columns].sort_values("amfi_code").reset_index(drop=True)
    same = live_sorted.drop(columns=["amfi_code", "anchor"]).round(12).equals(
        stored_sorted.drop(columns=["amfi_code", "anchor"]).round(12))
    if not same:
        bad = [c for c in live_sorted.columns if c not in ("amfi_code", "anchor")
              and not live_sorted[c].round(12).equals(stored_sorted[c].round(12))]
        raise AssertionError(f"FAIL: live panel diverges from features.parquet at {bad}")
    print(f"[selftest] Test A (feature parity @ {T.date()}, n={len(codes_T)}): PASS")

    # ---- Test B: prediction parity through the shipped artifact --------------
    inf = load_default()
    meta = ["amfi_code", "anchor"]
    live_rows = live_sorted.set_index("amfi_code")
    stored_rows = stored_sorted.set_index("amfi_code")
    max_err = 0.0
    for target in inf.targets:
        for code in codes_T:
            p_live = inf.predict(live_rows.loc[code].drop(meta[1:]).to_dict(), target)
            p_stored = inf.predict(stored_rows.loc[code].drop(meta[1:]).to_dict(), target)
            max_err = max(max_err, abs(p_live - p_stored))
    assert max_err < 1e-9, f"FAIL: live-vs-stored prediction diverges (max {max_err:.2e})"
    print(f"[selftest] Test B (prediction parity, {len(inf.targets)} targets): "
          f"max|Δp|={max_err:.2e} — PASS")

    # ---- Test C: anti-lookahead on the live code path specifically ----------
    truncated = {c: s.loc[:T] for c, s in nav_panel.items()}
    _, trunc_panel = build_live_panel(manifest, truncated, T, codes=codes_T)
    trunc_sorted = trunc_panel.sort_values("amfi_code").reset_index(drop=True)
    same_c = live_sorted.drop(columns=["amfi_code", "anchor"]).round(12).equals(
        trunc_sorted.drop(columns=["amfi_code", "anchor"]).round(12))
    assert same_c, "FAIL: live panel changes when post-anchor NAV data is removed — lookahead bug"
    print("[selftest] Test C (hard-truncate post-anchor NAV, same features): PASS")

    # ---- Gate tests: THIN_COHORT / STALE_NAV / NOT_IN_UNIVERSE ---------------
    target_code = codes_T[0]
    engine = FeatureEngine(manifest, nav_panel)
    cohort_key = engine.group_key(target_code)
    real_peers = set(engine.resolver.groups[cohort_key])

    thin_manifest = manifest[~manifest["amfi_code"].isin(
        [c for c in real_peers if c != target_code][2:])].reset_index(drop=True)
    res = score_live(target_code, manifest=thin_manifest, nav_panel=nav_panel,
                    inferencer=inf, today=T + pd.Timedelta(days=1))
    assert res["status"] == "THIN_COHORT", f"FAIL: expected THIN_COHORT, got {res['status']}"
    print(f"[selftest] THIN_COHORT gate (cohort_n={res['cohort_n']} < {COHORT_MIN_SIZE}): PASS")

    # nav_panel's real cache runs well past T, so use `truncated` (frozen at T)
    # to actually exercise staleness rather than always capping at `today`.
    res = score_live(target_code, manifest=manifest, nav_panel=truncated, inferencer=inf,
                    today=T + pd.Timedelta(days=45))
    assert res["status"] == "STALE_NAV", f"FAIL: expected STALE_NAV, got {res['status']}"
    print(f"[selftest] STALE_NAV gate (staleness={res['staleness_days']}d > {STALE_MAX_DAYS}d): PASS")

    res = score_live("NOT-A-REAL-CODE", manifest=manifest, nav_panel=nav_panel,
                    inferencer=inf, today=T)
    assert res["status"] == "NOT_IN_UNIVERSE"
    print("[selftest] NOT_IN_UNIVERSE gate (unknown code): PASS")

    # ---- End-to-end sanity: a real fund scores OK with a stable-shaped dict --
    res = score_live(target_code, manifest=manifest, nav_panel=nav_panel, inferencer=inf,
                    today=T)
    assert res["status"] == "OK" and 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["cohort_percentile"] <= 1.0
    assert target_code in res["cohort_codes"] and len(res["cohort_codes"]) == res["cohort_n"]
    assert res["features"] and set(res["features"]) == set(inf.features)
    print(f"[selftest] end-to-end score_live({target_code}): status=OK "
          f"p={res['probability']:.3f} percentile={res['cohort_percentile']:.2f} "
          f"cohort_n={res['cohort_n']} universe_n={res['universe_n']} — PASS")

    print("[selftest] PASS — live scoring reproduces the training panel exactly, "
          "predictions match the shipped artifact, and honesty gates fire correctly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="feature/prediction parity vs stored artifacts + anti-lookahead + gates")
    ap.add_argument("--code", type=str, default=None, help="score a single AMFI code")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.code:
        print(score_live(args.code))
    else:
        ap.print_help()
