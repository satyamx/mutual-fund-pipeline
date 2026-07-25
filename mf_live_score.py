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
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from mf_benchmarks import _asof
from mf_features import FeatureEngine, LEVEL_FEATURES, assemble_panel
from mf_infer import CohortInferencer, DEFAULT_TARGET, load_default
from mf_labels import COHORT_MIN_SIZE, load_manifest, load_nav_panel
import mf_universe
import mf_overrides

LOGGER = logging.getLogger("MFLiveScore")

STALE_MAX_DAYS = 30

# OUT_OF_TRAINING_UNIVERSE is a PERMANENT refusal — the cohort_q1 model was fitted
# and holdout-validated on 10 equity categories only, so no amount of fetching makes
# a liquid fund or an index fund scoreable by it. It is deliberately distinct from
# NOT_IN_UNIVERSE (a fixable data gap: right category, just no cached NAV / not in
# the trained sample), because the two demand opposite responses from the caller —
# one is "never ask again", the other is "refresh and retry".
_STATUSES = ("OK", "THIN_COHORT", "STALE_NAV", "NOT_IN_UNIVERSE",
             "OUT_OF_TRAINING_UNIVERSE", "SECTOR_UNRESOLVED")

# Categories whose cohort is keyed on SECTOR, not on the category alone:
# PeerProxyResolver._group_key returns ("sector", row["sector"]) for these. The
# manifest's 52 sector values are HAND-CURATED and AMFI's NAVAll carries no sector
# field whatsoever, so a candidate outside the trained sample has no honest cohort.
# Guessing one would fabricate the very peer set the label is defined against
# ("top quartile WITHIN cohort") — so these are refused, not guessed. This is the
# binding constraint on whole-market coverage: 256 of the 569 canonical scoreable
# schemes are Sectoral/Thematic.
SECTOR_KEYED_CATEGORIES = frozenset({"Sectoral/Thematic"})


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


def load_universe_index() -> Dict[str, str]:
    """`amfi_code -> AMFI category_raw` for the whole market, from the NAVAll master.

    Built once and injected into `score_live` so a batch loop doesn't re-read the
    master per fund. Returns {} if the master isn't cached — the gate then simply
    can't distinguish a permanent category refusal from a data gap, and falls back
    to the conservative NOT_IN_UNIVERSE, which is still a refusal.
    """
    try:
        import mf_datasources  # noqa: PLC0415 — optional/heavier dependency, loaded on demand
        df = mf_datasources.AMFIRegistryLive().master()
    except Exception as exc:  # noqa: BLE001 — a missing master must never break scoring
        LOGGER.warning("Universe index unavailable (%s) — category gate degrades to NOT_IN_UNIVERSE", exc)
        return {}
    if df is None or df.empty:
        return {}
    return {str(c): cat for c, cat in zip(df["amfi_code"], df["category_raw"])}


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


def _augmented_manifest(manifest: pd.DataFrame, code: str, category: str,
                        series: pd.Series, sector: Optional[str] = None) -> pd.DataFrame:
    """`manifest` + ONE synthetic row for a candidate outside the trained sample.

    Only `amfi_code`, `category` and `sector` are read by FeatureEngine /
    PeerProxyResolver; the remaining columns are filled for schema consistency only.
    `sector` is None unless a CURATED one was supplied via mf_overrides — it is never
    inferred, since a guessed sector fabricates the peer set the label is defined on.
    """
    nav = series.dropna()
    row = {c: None for c in manifest.columns}
    row.update(amfi_code=str(code), scheme_name=f"<candidate {code}>", amc=None,
               category=category, sector=sector,
               first_nav_date=nav.index.min(), last_nav_date=nav.index.max(),
               n_obs=int(len(nav)))
    return pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)


def _z_candidate_from_reference(panel: pd.DataFrame, reference_codes: set,
                                code: str) -> pd.DataFrame:
    """Fill ONLY the candidate row's z_* using the reference population's mean/std.

    Reference rows keep the z-scores they were built with and are never recomputed —
    that is what makes them provably identical to a no-candidate run. The candidate
    is then normalized against that same frozen distribution, which is the standard
    train/serve normalization contract.
    """
    ref_mask = panel["amfi_code"].astype(str).isin({str(c) for c in reference_codes})
    cand_mask = panel["amfi_code"].astype(str) == str(code)
    for col in LEVEL_FEATURES:
        ref = panel.loc[ref_mask, col]
        std, cnt = ref.std(), ref.count()
        # Same guard assemble_panel applies: std > 1e-12 and at least 5 observations.
        ok = pd.notna(std) and std > 1e-12 and cnt >= 5
        panel.loc[cand_mask, f"z_{col}"] = (
            (panel.loc[cand_mask, col] - ref.mean()) / std if ok else np.nan)
    return panel


def build_candidate_panel(manifest: pd.DataFrame, nav_panel: Dict[str, pd.Series],
                          code: str, category: str, t: pd.Timestamp,
                          sector: Optional[str] = None
                          ) -> tuple[FeatureEngine, pd.DataFrame, set]:
    """One-anchor panel: the UNTOUCHED trained reference universe, plus one candidate.

    The reference rows are taken from a panel built as if the candidate did not
    exist, and are then reused verbatim. That is not a stylistic choice — it is
    required for correctness, and the naive "assemble one panel over reference +
    candidate" approach is measurably wrong:

      `excess_1y` / `excess_3y` are benchmark-relative, and where no real index is
      available the benchmark falls back to `PeerProxyResolver.loo_median`, the
      leave-one-out median over COHORT PEERS. Adding the candidate to the manifest
      puts it inside that median, which shifts the benchmark — and therefore the
      excess_* features — of every reference fund sharing its cohort. Measured on
      real cached data at anchor 2023-06-30 this moved z_excess_1y by up to 4.7e-2
      and z_excess_3y by 3.2e-2, i.e. thirteen orders of magnitude above float
      noise. Freezing only the z-STATISTICS does not help, because the underlying
      feature VALUES have already moved.

    The candidate's own row is taken from the augmented panel, where its cohort rank
    and its own peer-proxy benchmark are computed correctly — `loo_median` excludes
    self, so the candidate's benchmark is the median over reference peers, exactly
    as it would have been had it been in the trained sample.

    Restricting this path to non-sector-keyed categories additionally leaves
    `PeerProxyResolver.small_sectors` untouched (it is derived solely from
    Sectoral/Thematic rows), so no existing fund's cohort is redefined either.
    """
    # (1) The reference universe, exactly as a no-candidate run would compute it.
    _, ref_panel = build_live_panel(manifest, nav_panel, t)
    ref_codes = [str(c) for c in ref_panel["amfi_code"]]

    # (2) The candidate's own row, computed against those same peers.
    aug = _augmented_manifest(manifest, code, category, nav_panel[code], sector=sector)
    engine = FeatureEngine(aug, nav_panel)

    # `small_sectors` is derived from the Sectoral/Thematic rows with fewer than 3
    # funds, so a SECTOR-keyed candidate can tip a sector over that threshold and
    # silently re-cohort funds the model was calibrated on — moving them between
    # ("sector", X) and ("pooled_thematic",). Pin it to the base manifest's value and
    # rebuild `groups` so the candidate is placed by the SAME rule the training run
    # used. Category-keyed candidates can't affect it, but pinning unconditionally
    # keeps the invariant true regardless of which path is taken.
    base_resolver = FeatureEngine(manifest, nav_panel).resolver
    engine.resolver.small_sectors = base_resolver.small_sectors
    engine.resolver.groups = {}
    for _c in engine.resolver.manifest.index:
        engine.resolver.groups.setdefault(engine.resolver._group_key(_c), []).append(_c)
    pairs = pd.DataFrame({"amfi_code": ref_codes + [str(code)], "anchor": t})
    aug_panel = assemble_panel(engine, pairs)
    cand_row = aug_panel[aug_panel["amfi_code"].astype(str) == str(code)]

    panel = pd.concat([ref_panel, cand_row], ignore_index=True)
    return engine, _z_candidate_from_reference(panel, set(ref_codes), code), set(ref_codes)


def score_live(code: str, *,
              manifest: Optional[pd.DataFrame] = None,
              nav_panel: Optional[Dict[str, pd.Series]] = None,
              inferencer: Optional[CohortInferencer] = None,
              engine: Optional[FeatureEngine] = None,
              today: Optional[pd.Timestamp] = None,
              universe_index: Optional[Dict[str, str]] = None,
              nav_loader: Optional[Callable[[str], Optional[pd.Series]]] = None,
              overrides: Optional[Dict[str, Any]] = None,
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
    has_nav = series is not None and not series.dropna().empty
    # Set only when the fund is OUTSIDE the trained manifest but its AMFI category IS
    # one the model trained on — it is then scored by insertion into the frozen
    # reference panel rather than refused. Stays None for in-manifest funds.
    candidate_category: Optional[str] = None
    candidate_sector: Optional[str] = None
    if code not in manifest_idx.index or not has_nav:
        # Two very different refusals hide behind "not in the manifest", and the
        # caller must be able to tell them apart: a Liquid Fund is permanently
        # unscoreable by an equity-only model, whereas a Flexi Cap fund that simply
        # wasn't sampled into the 136 becomes scoreable the moment its NAV is cached.
        # Collapsing both into one status would tell the app to keep retrying the
        # former forever, and to give up on the latter.
        if universe_index:
            verdict = mf_universe.classify(universe_index.get(str(code)),
                                           mf_universe.trained_categories(manifest))
            if verdict.status == mf_universe.STATUS_OUT_OF_UNIVERSE:
                return _gap_result("OUT_OF_TRAINING_UNIVERSE", target,
                                   cohort_key=verdict.category,
                                   note=f"{verdict.reason}. The cohort_q1 model was fitted and "
                                        "holdout-validated on equity categories only, so no "
                                        "probability is emitted here — this is a permanent "
                                        "refusal, not a missing-data flag.")
            if verdict.status == mf_universe.STATUS_TRAINED:
                if verdict.category in SECTOR_KEYED_CATEGORIES:
                    # A CURATED sector (mf_overrides, git-tracked + validated) is the
                    # only way past this — never an inferred one.
                    candidate_sector = (overrides or {}).get(str(code))
                    candidate_sector = getattr(candidate_sector, "sector", None)
                    if not candidate_sector:
                        return _gap_result(
                            "SECTOR_UNRESOLVED", target, cohort_key=verdict.category,
                            note=f"{verdict.category} cohorts are keyed on SECTOR, and no free "
                                 "source publishes a scheme's sector (the manifest's are "
                                 "hand-curated). Guessing one would fabricate the peer set the "
                                 "within-cohort label is defined against. Supply a sourced row in "
                                 f"{mf_overrides.OVERRIDES_PATH} to resolve it.")
                if not has_nav and nav_loader is not None:
                    # load_nav_panel() only populates MANIFEST funds, so a candidate
                    # arrives with no series. Fetch on demand — gated behind a
                    # confirmed trained, non-sector-keyed category so a batch run
                    # never fetches for funds it was going to refuse anyway.
                    try:
                        fetched = nav_loader(str(code))
                    except Exception as exc:  # noqa: BLE001 — a failed fetch is a data gap, not a crash
                        LOGGER.warning("NAV fetch failed for candidate %s: %s", code, exc)
                        fetched = None
                    if fetched is not None and not fetched.dropna().empty:
                        series, has_nav = fetched, True
                        nav_panel = {**nav_panel, str(code): fetched}
                if has_nav:
                    candidate_category = verdict.category
        if candidate_category is None:
            return _gap_result("NOT_IN_UNIVERSE", target,
                               note="Fund not in the trained universe manifest, or no cached NAV.")

    last_dt = series.dropna().index.max()
    t = min(last_dt, today)
    staleness_days = int((today - t).days)
    if staleness_days > STALE_MAX_DAYS:
        return _gap_result("STALE_NAV", target, anchor=str(t.date()), staleness_days=staleness_days,
                           note=f"Latest cached NAV is {staleness_days}d behind today "
                                f"(> {STALE_MAX_DAYS}d tolerance) — refresh the cache before scoring.")

    if candidate_category is not None:
        # Fresh engine on purpose: any `engine` passed in was built on the base
        # manifest and knows nothing about this candidate.
        engine, panel, _ref = build_candidate_panel(manifest, nav_panel, code,
                                                    candidate_category, t,
                                                    sector=candidate_sector)
    else:
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

    # ---- OUT_OF_TRAINING_UNIVERSE: a PERMANENT refusal, distinct from a data gap --
    # Driven by real AMFI categories, so this exercises the actual shipped mapping
    # rather than a hand-written fixture.
    uni = load_universe_index()
    if uni:
        trained = mf_universe.trained_categories(manifest)
        off = {"debt": None, "index": None, "equity_untrained": None}
        for c, raw in uni.items():
            if c in nav_panel or c in set(manifest["amfi_code"]):
                continue
            v = mf_universe.classify(raw, trained)
            if v.status != mf_universe.STATUS_OUT_OF_UNIVERSE:
                continue
            r = str(raw)
            if off["debt"] is None and "Debt Scheme" in r:
                off["debt"] = c
            elif off["index"] is None and "Index Fund" in r:
                off["index"] = c
            elif off["equity_untrained"] is None and "Dividend Yield" in r:
                off["equity_untrained"] = c

        for kind, c in off.items():
            if c is None:
                continue
            res = score_live(c, manifest=manifest, nav_panel=nav_panel, inferencer=inf,
                            today=T, universe_index=uni)
            assert res["status"] == "OUT_OF_TRAINING_UNIVERSE", (kind, c, res["status"])
            assert res["probability"] is None, "FAIL: a refused fund must carry no probability"
            assert "permanent refusal" in res["note"], res["note"]
        found = {k: v for k, v in off.items() if v}
        assert found, "FAIL: no out-of-universe fund found in the real AMFI master"

        # Same call WITHOUT the index must degrade to the conservative refusal —
        # never to a score. A missing master may cost precision, never safety.
        any_code = next(iter(found.values()))
        res = score_live(any_code, manifest=manifest, nav_panel=nav_panel, inferencer=inf, today=T)
        assert res["status"] == "NOT_IN_UNIVERSE" and res["probability"] is None
        print(f"[selftest] OUT_OF_TRAINING_UNIVERSE gate on real AMFI rows "
              f"({', '.join(found)}): permanent refusal, no probability; "
              f"degrades safely without the index: PASS")

    # ---- Candidate insertion (#8 part 3) ------------------------------------
    # Score a fund that is NOT in the trained manifest by inserting it into the
    # frozen reference panel. Built by REMOVING a real fund from the manifest and
    # re-scoring it as an outsider: its true in-manifest result is the ground truth.
    victim = None
    for c in codes_T:
        row = manifest.set_index("amfi_code").loc[c]
        if row["category"] not in SECTOR_KEYED_CATEGORIES:
            peers = [p for p in manifest[manifest["category"] == row["category"]]["amfi_code"]
                     if p in nav_panel]
            if len(peers) > COHORT_MIN_SIZE:      # still a valid cohort once removed
                victim = c
                break
    assert victim is not None, "FAIL: no non-sectoral fund available for the insertion test"
    vic_cat = manifest.set_index("amfi_code").loc[victim]["category"]
    reduced = manifest[manifest["amfi_code"] != victim].reset_index(drop=True)

    # (a) THE LOAD-BEARING PROPERTY: inserting a candidate must not move ANY
    #     reference fund's z-scores. If it did, every peer would be scored on a
    #     different scale than the model was calibrated on.
    _, base_panel = build_live_panel(reduced, nav_panel, T)
    _, cand_panel, ref_codes = build_candidate_panel(reduced, nav_panel, victim, vic_cat, T)
    zcols = [f"z_{c}" for c in LEVEL_FEATURES]
    b = base_panel[base_panel["amfi_code"].isin(ref_codes)].sort_values("amfi_code")
    a = cand_panel[cand_panel["amfi_code"].isin(ref_codes)].sort_values("amfi_code")
    assert len(a) == len(b) and len(b) > 0
    assert a[zcols].reset_index(drop=True).equals(b[zcols].reset_index(drop=True)), \
        "FAIL: candidate insertion shifted the reference universe's z-scores"
    assert victim not in set(base_panel["amfi_code"]) and victim in set(cand_panel["amfi_code"])
    print(f"[selftest] insertion leaves all {len(ref_codes)} reference z-scores bit-identical: PASS")

    # (a2) TEETH: the naive "one panel over reference + candidate" really is wrong,
    #      so the split construction above is load-bearing, not defensive styling.
    #      The candidate joins its cohort's leave-one-out peer-proxy benchmark, which
    #      shifts peers' excess_* features.
    naive_engine = FeatureEngine(
        _augmented_manifest(reduced, victim, vic_cat, nav_panel[victim]), nav_panel)
    naive = assemble_panel(naive_engine, pd.DataFrame(
        {"amfi_code": list(ref_codes) + [str(victim)], "anchor": T}))
    nv = naive[naive["amfi_code"].isin(ref_codes)].sort_values("amfi_code").reset_index(drop=True)
    drift = (nv[zcols] - b[zcols].reset_index(drop=True)).abs().max().max()
    assert drift > 1e-6, (f"FAIL: expected the naive panel to contaminate peers, but max "
                          f"drift was {drift:.2e} — the guard may no longer be needed")
    print(f"[selftest] naive single-panel build WOULD contaminate peers "
          f"(max |Δz|={drift:.2e}) — split construction is load-bearing: PASS")

    # (b) cohorts of existing funds must be unchanged (small_sectors untouched).
    eng_base = FeatureEngine(reduced, nav_panel)
    eng_cand = build_candidate_panel(reduced, nav_panel, victim, vic_cat, T)[0]
    assert eng_cand.resolver.small_sectors == eng_base.resolver.small_sectors, \
        "FAIL: inserting a candidate redefined the sector-pooling of existing funds"
    for c in list(reduced["amfi_code"])[:25]:
        assert eng_cand.group_key(c) == eng_base.group_key(c), f"FAIL: cohort of {c} changed"
    print("[selftest] insertion leaves existing cohorts + small_sectors unchanged: PASS")

    # (c) Outsiders actually score, and track their in-manifest truth. Run over
    #     SEVERAL funds and require a spread of probabilities, so this can't pass
    #     vacuously on a cluster of near-zero scores.
    mi = manifest.set_index("amfi_code")
    checked, deltas = [], []
    for c in codes_T:
        if len(checked) >= 6:
            break
        cat = mi.loc[c]["category"]
        if cat in SECTOR_KEYED_CATEGORIES:
            continue
        peers = [p for p in manifest[manifest["category"] == cat]["amfi_code"] if p in nav_panel]
        if len(peers) <= COHORT_MIN_SIZE:
            continue
        red = manifest[manifest["amfi_code"] != c].reset_index(drop=True)
        # Build the AMFI string from the SHIPPED mapping rather than hand-writing it:
        # the manifest's category is not always AMFI's sub-category name (e.g. the
        # manifest pools AMFI's "Value Fund"/"Contra Fund" into one "Value/Contra").
        sub = next(k for k, v in mf_universe._SUBCATEGORY_MAP.items() if v == cat)
        got = score_live(c, manifest=red, nav_panel=nav_panel, inferencer=inf, today=T,
                         universe_index={str(c): f"Open Ended Schemes(Equity Scheme - {sub})"})
        truth = score_live(c, manifest=manifest, nav_panel=nav_panel, inferencer=inf, today=T)
        assert got["status"] == "OK", f"FAIL: candidate {c} not scored: {got['status']} / {got['note']}"
        assert 0.0 <= got["probability"] <= 1.0
        assert got["cohort_key"] == truth["cohort_key"], \
            f"FAIL: {c} landed in cohort {got['cohort_key']} vs true {truth['cohort_key']}"
        if truth["status"] == "OK":
            # Not bit-identical by construction: the trained cohort had one more
            # member, so peer ranks and the leave-one-out benchmark differ slightly.
            deltas.append(abs(got["probability"] - truth["probability"]))
        checked.append((c, cat, got["probability"]))

    assert len(checked) >= 3, f"FAIL: only {len(checked)} insertion cases exercised"
    assert deltas and max(deltas) < 0.05, f"FAIL: insertion drifted from truth by {max(deltas):.4f}"
    spread = max(p for _, _, p in checked) - min(p for _, _, p in checked)
    assert spread > 1e-3, (f"FAIL: all {len(checked)} probabilities within {spread:.2e} — the "
                           "comparison against truth would pass vacuously")
    print(f"[selftest] {len(checked)} outsiders scored via insertion "
          f"(p spread {spread:.3f}, max |Δ| vs in-manifest truth {max(deltas):.4f}, "
          f"cohorts all matched): PASS")

    # (d) sector-keyed categories are REFUSED, never guessed.
    sect = next((c for c in codes_T
                 if manifest.set_index("amfi_code").loc[c]["category"] in SECTOR_KEYED_CATEGORIES), None)
    if sect is not None:
        red2 = manifest[manifest["amfi_code"] != sect].reset_index(drop=True)
        res = score_live(sect, manifest=red2, nav_panel=nav_panel, inferencer=inf, today=T,
                         universe_index={str(sect): "Open Ended Schemes(Equity Scheme - Sectoral/ Thematic)"})
        assert res["status"] == "SECTOR_UNRESOLVED" and res["probability"] is None, res["status"]
        print("[selftest] sector-keyed outsider refused (no free sector source): PASS")

        # ...and a CURATED sector unlocks exactly that fund, with the trained
        # universe's cohorts still untouched.
        true_sector = manifest.set_index("amfi_code").loc[sect]["sector"]
        ov = {str(sect): mf_overrides.Override(amfi_code=str(sect), sector=str(true_sector),
                                               source_url="https://amc.example/sid.pdf")}
        res2 = score_live(sect, manifest=red2, nav_panel=nav_panel, inferencer=inf, today=T,
                          universe_index={str(sect): "Open Ended Schemes(Equity Scheme - Sectoral/ Thematic)"},
                          overrides=ov)
        assert res2["status"] in ("OK", "THIN_COHORT"), res2["status"]
        if res2["status"] == "OK":
            truth_s = score_live(sect, manifest=manifest, nav_panel=nav_panel,
                                 inferencer=inf, today=T)
            assert res2["cohort_key"] == truth_s["cohort_key"], \
                f"FAIL: curated sector put {sect} in {res2['cohort_key']} vs true {truth_s['cohort_key']}"
        # small_sectors must be pinned to the BASE manifest even though the candidate
        # is Sectoral/Thematic — otherwise it could tip a sector over the <3 threshold
        # and silently re-cohort funds the model was calibrated on.
        eng_s = build_candidate_panel(red2, nav_panel, sect, "Sectoral/Thematic", T,
                                      sector=str(true_sector))[0]
        base_s = FeatureEngine(red2, nav_panel)
        assert eng_s.resolver.small_sectors == base_s.resolver.small_sectors, \
            "FAIL: a sector-keyed candidate shifted small_sectors"
        for c in list(red2["amfi_code"])[:25]:
            assert eng_s.group_key(c) == base_s.group_key(c), f"FAIL: cohort of {c} changed"
        print(f"[selftest] curated sector unlocks {sect} -> {res2['status']}; "
              f"small_sectors + existing cohorts unchanged: PASS")

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
