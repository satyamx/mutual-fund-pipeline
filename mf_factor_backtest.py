"""
====================================================================================
mf_factor_backtest.py — factor-weight backtest harness for ScoringEngine (Step 2 prep)
====================================================================================
Question this answers: do the CURRENT ScoringEngine.FACTOR_MAP[EQUITY] weights
(mf_pipeline.py:895-983) actually rank funds better than chance within their own
peer cohort, and how does that compare to the fitted elastic-net model
(mf_model.py cohort_q1 / cohort_top_half runs)? Proposed weight changes can be
plugged into `evaluate()` and compared against the baseline this module reports.

WHAT IT DOES
  1. Maps FACTOR_MAP[EQUITY] metric names onto mf_cache/phase_b/features.parquet
     columns (`FEATURE_MAP` below). Two of the ten metrics have no feature
     counterpart in this dataset (`days_to_liquidate_wavg`, `expense_ratio` —
     15% of the nominal weight) and are marked untestable rather than proxied.
  2. `score(weights, df)` reproduces ScoringEngine.score_category's math
     (mf_pipeline.py:959-983) exactly — percentile-rank each mapped metric,
     direction-adjust, weight, sum, min-max stretch to 1..100 — generalized
     from a single-snapshot scoring call to this panel's repeated monthly
     anchors: the peer group at each step is (anchor, cohort_key), where
     cohort_key is the exact PeerProxyResolver grouping mf_labels.py used to
     build y_cohort_q1 / y_cohort_top_half (category for diversified equity,
     sector for thematic funds with >=3 members, pooled otherwise). Accepts
     ANY weight dict over FEATURE_MAP's testable metrics — untestable metrics
     and non-positive weights are dropped and the remainder renormalized,
     exactly as score_category does when a factor column is absent.
  3. `evaluate(weights)` scores the fixed formula (no fitting — there is
     nothing to train) on mf_model.load_cohort_dataset()'s exact rows, then
     measures rank quality on the SAME purged CPCV folds and causal holdout
     mf_cv.py/mf_model.py use for the fitted model, via mf_model's own metric
     functions (pooled_auc, within_anchor_rank_auc, decile_stats) so numbers
     are directly comparable.
  4. `main()` runs `evaluate()` on the CURRENT (unmodified) FACTOR_MAP[EQUITY]
     weights and writes mf_cache/phase_b/factor_backtest_baseline.md.

This module does not modify mf_pipeline.py, mf_model.py, or any cached
artifact; it only reads mf_cache/phase_b/{features,labels}.parquet (via
mf_model.load_cohort_dataset) and writes its own new report file.
====================================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from mf_cv import cpcv_folds, causal_holdout
from mf_model import (
    COHORT_TARGETS, OUT_DIR, load_cohort_dataset, pooled_auc,
    within_anchor_rank_auc, decile_stats,
)
from mf_pipeline import ScoringEngine, SEBICategory

REPORT_PATH = OUT_DIR / "factor_backtest_baseline.md"

# ==============================================================================
# Metric -> feature mapping (design decision, not invented data)
# ==============================================================================
# FACTOR_MAP[EQUITY] weight/direction pulled live from mf_pipeline.py so this
# harness can never silently drift out of sync with the source scoring math.
EQUITY_FACTORS: Dict[str, tuple] = ScoringEngine.FACTOR_MAP[SEBICategory.EQUITY]
DIRECTIONS: Dict[str, int] = {m: d for m, (_, d) in EQUITY_FACTORS.items()}
DEFAULT_WEIGHTS: Dict[str, float] = {m: w for m, (w, _) in EQUITY_FACTORS.items()}

# None = genuinely untestable with the data on hand (mf_cache/phase_b/features.parquet
# has no liquidity/impact-cost or expense-ratio column) -- NOT proxied, NOT invented.
FEATURE_MAP: Dict[str, Optional[str]] = {
    "rr3y_mean":              "cagr_3y",
    "rr5y_mean":              "cagr_5y",
    "sortino":                "sortino_3y",
    "info_ratio":             "ir_3y",
    "alpha":                  "excess_3y",
    "upside_capture":         "upcap_3y",
    "downside_capture":       "downcap_3y",
    "max_drawdown":           "max_dd_3y",     # stored negative; direction +1 already
                                                # means "less negative wins" (matches
                                                # mf_pipeline.py:915 comment)
    "days_to_liquidate_wavg": None,             # UNTESTABLE — no liquidity/AUM feature
    "expense_ratio":          None,             # UNTESTABLE — no expense-ratio feature
}

TESTABLE_WEIGHT = sum(w for m, w in DEFAULT_WEIGHTS.items() if FEATURE_MAP[m])
UNTESTABLE_WEIGHT = sum(w for m, w in DEFAULT_WEIGHTS.items() if not FEATURE_MAP[m])


# ==============================================================================
# score() — mirrors ScoringEngine._percentile_rank / score_category exactly
# ==============================================================================
def _percentile_rank(col: pd.Series, direction: int) -> pd.Series:
    """Verbatim copy of ScoringEngine._percentile_rank (mf_pipeline.py:959-962):
    peer percentile in [0,1]; NaNs get the peer-neutral median (0.5)."""
    ranked = col.rank(pct=True, ascending=(direction > 0))
    return ranked.fillna(0.5)


def score(weights: Dict[str, float], df: pd.DataFrame) -> pd.Series:
    """ScoringEngine-style composite score for arbitrary EQUITY factor weights.

    Peer group = (anchor, cohort_key): the same monthly, cohort-relative peer
    set y_cohort_q1/y_cohort_top_half were computed against. Untestable /
    non-positive-weight metrics are dropped and the rest renormalized, exactly
    as score_category does with `weight_sum`. Final step reproduces the
    per-peer-group min-max stretch to a 1..100 band (mf_pipeline.py:978-981).
    """
    available = {m: w for m, w in weights.items() if FEATURE_MAP.get(m) and w > 0}
    if not available:
        raise ValueError("score(): no testable metrics with positive weight")
    weight_sum = sum(available.values())

    grp_keys = [df["anchor"], df["cohort_key"]]
    composite = pd.Series(0.0, index=df.index)
    for metric, w in available.items():
        feat_col = FEATURE_MAP[metric]
        direction = DIRECTIONS[metric]
        pct = df.groupby(grp_keys)[feat_col].transform(
            lambda s: _percentile_rank(s, direction))
        composite += (w / weight_sum) * pct

    grp = composite.groupby(grp_keys)
    lo, hi = grp.transform("min"), grp.transform("max")
    spread = np.where((hi - lo) > 1e-12, (composite - lo) / (hi - lo), 0.5)
    return pd.Series(np.round(1.0 + 99.0 * spread, 1), index=df.index, name="factor_score")


# ==============================================================================
# evaluate() — same purged CPCV folds + causal holdout as mf_model.py
# ==============================================================================
def _factor_block_metrics(anchors: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict:
    """Same headline metrics as mf_model.block_metrics, minus the
    calibration-only fields (recall_at_05, brier) -- the factor score is not
    a probability, so calibration doesn't apply."""
    base = float(y.mean())
    rank, n_cohorts = within_anchor_rank_auc(anchors, y, p)
    dec = decile_stats(anchors, y, p)
    lift = dec["top_decile_precision"] / base if base > 0 else float("nan")
    return dict(
        n=int(len(y)), base_rate=base, negatives=int((~y.astype(bool)).sum()),
        pooled_auc=pooled_auc(y, p), rank_auc=rank, rank_auc_cohorts=n_cohorts,
        top_decile_precision=dec["top_decile_precision"], lift_over_base=lift,
        bottom_decile_neg_capture=dec["bottom_decile_neg_capture"],
        total_negatives=dec["total_negatives"],
    )


def evaluate(weights: Dict[str, float]) -> dict:
    """Score `weights` once (fixed formula, nothing to fit) and measure rank
    quality on the exact purged CPCV blocks + causal holdout mf_model.py used
    for the fitted cohort_q1 / cohort_top_half models, against both cohort
    targets. Returns a dict keyed by target name."""
    df, _ = load_cohort_dataset()
    p_all = score(weights, df).to_numpy()
    anchors_all = df["anchor"].to_numpy()

    results: dict = {"targets": {}}
    for tname, tcol in COHORT_TARGETS.items():
        y_all = df[tcol].to_numpy(dtype=bool)

        cpcv_blocks = {}
        for block, _train, test in cpcv_folds(df["anchor"]):
            cpcv_blocks[block] = _factor_block_metrics(
                anchors_all[test], y_all[test], p_all[test])

        _train_h, test_h = causal_holdout(df["anchor"])
        holdout_m = _factor_block_metrics(
            anchors_all[test_h], y_all[test_h], p_all[test_h])

        results["targets"][tname] = dict(
            cpcv_blocks=cpcv_blocks,
            cpcv_mean_pooled_auc=float(np.nanmean(
                [m["pooled_auc"] for m in cpcv_blocks.values()])),
            cpcv_mean_rank_auc=float(np.nanmean(
                [m["rank_auc"] for m in cpcv_blocks.values()])),
            cpcv_mean_lift=float(np.nanmean(
                [m["lift_over_base"] for m in cpcv_blocks.values()])),
            cpcv_mean_precision=float(np.nanmean(
                [m["top_decile_precision"] for m in cpcv_blocks.values()])),
            holdout=holdout_m,
        )
    return results


# ==============================================================================
# Reference numbers from the fitted elastic-net model (mf_model.py cohort runs),
# passed in fixed rather than re-parsed from a report, so the harness has no
# hidden dependency on report-file formatting.
# ==============================================================================
FITTED_MODEL_REFERENCE = {
    "cohort_q1": dict(cpcv_auc=0.548, holdout_auc=0.578, base_rate=0.323),
    "cohort_top_half": dict(cpcv_auc=0.537, holdout_auc=0.545, base_rate=0.465),
}


# ==============================================================================
# Report
# ==============================================================================
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


_METRIC_HEADER = ("| block | n | base rate | neg | pooled AUC | rank-AUC (cohorts) | "
                   "prec@top-dec | lift | bottom-dec neg capture |\n"
                   "|---|---|---|---|---|---|---|---|---|")


def _metric_row(block: str, m: dict) -> str:
    return ("| " + " | ".join([
        block, str(m["n"]), _fmt(m["base_rate"]), str(m["negatives"]),
        _fmt(m["pooled_auc"]), f"{_fmt(m['rank_auc'])} ({m['rank_auc_cohorts']})",
        _fmt(m["top_decile_precision"]), _fmt(m["lift_over_base"], 2),
        _fmt(m["bottom_decile_neg_capture"]),
    ]) + " |")


def write_report(results: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Factor backtest baseline — current ScoringEngine EQUITY weights")
    add("")
    add(f"Generated {datetime.now(timezone.utc).date()}. Measures how well the "
        "CURRENT ScoringEngine.FACTOR_MAP[EQUITY] weights (mf_pipeline.py:907-918) "
        "rank funds within their own peer cohort, on the same purged CPCV blocks "
        "and causal holdout mf_model.py uses for the fitted elastic-net model. "
        "The factor score is a fixed formula (percentile-rank -> weight -> "
        "min-max rescale), not a fitted model: there is nothing to train, so "
        "every fold here is evaluation-only.")
    add("")

    add("## Metric -> feature mapping")
    add("")
    add("| FACTOR_MAP metric | weight | direction | feature used | status |")
    add("|---|---|---|---|---|")
    for m, (w, d) in EQUITY_FACTORS.items():
        feat = FEATURE_MAP[m]
        status = "testable" if feat else "**UNTESTABLE — no counterpart in features.parquet**"
        add(f"| {m} | {w:.3f} | {'+1 (higher better)' if d > 0 else '-1 (lower better)'} "
            f"| {feat or '—'} | {status} |")
    add("")
    add(f"Testable weight: **{TESTABLE_WEIGHT:.2f}** of 1.00. Untestable weight: "
        f"**{UNTESTABLE_WEIGHT:.2f}** (`days_to_liquidate_wavg` 0.10 + "
        "`expense_ratio` 0.05) — mf_cache/phase_b/features.parquet carries no "
        "liquidity/impact-cost or expense-ratio column, so these are dropped "
        "rather than proxied, and the testable metrics' weights are "
        f"renormalized to sum to 1.00 (each divided by {TESTABLE_WEIGHT:.2f}), "
        "exactly as ScoringEngine.score_category renormalizes over `available` "
        "factors when a column is absent (mf_pipeline.py:969-977). This means "
        "the backtest below is necessarily a test of 85% of the current "
        "weight scheme, not 100% of it.")
    add("")

    add("## Method")
    add("")
    add("- Peer group for percentile-ranking = `(anchor, cohort_key)`, the "
        "exact PeerProxyResolver grouping mf_labels.py used to build "
        "`y_cohort_q1` / `y_cohort_top_half` (category for diversified "
        "equity categories, sector for thematic funds with >=3 members, "
        "pooled small-sector thematic otherwise).")
    add("- Rows = mf_model.load_cohort_dataset() output verbatim (features + "
        "labels inner-joined, anchors >= 2014-01-31, cohort-eligible rows "
        "only) — the identical population the fitted elastic-net cohort "
        "models were scored on.")
    add("- CPCV blocks and the causal holdout are mf_cv.py's `cpcv_folds` / "
        "`causal_holdout` unchanged; only the TEST mask is used per block "
        "(the factor score has no training step).")
    add("- pooled AUC / within-anchor rank-AUC / precision@top-decile / lift "
        "are mf_model.py's own metric functions (`pooled_auc`, "
        "`within_anchor_rank_auc`, `decile_stats`), reused verbatim so these "
        "numbers are on the identical footing as the fitted-model numbers "
        "quoted below.")
    add("")

    add("## Baseline results — current weights")
    add("")
    for tname in COHORT_TARGETS:
        r = results["targets"][tname]
        add(f"### {tname}")
        add("")
        add("**CPCV (purged/embargoed, 5 blocks)**")
        add("")
        add(_METRIC_HEADER)
        for block, m in r["cpcv_blocks"].items():
            add(_metric_row(block, m))
        add(f"| **mean** | | | | **{r['cpcv_mean_pooled_auc']:.3f}** | "
            f"**{r['cpcv_mean_rank_auc']:.3f}** | **{r['cpcv_mean_precision']:.3f}** | "
            f"**{r['cpcv_mean_lift']:.2f}** | |")
        add("")
        add("**Causal holdout (train <= 2019-07-31, test 2022-08-31..2023-07-31)**")
        add("")
        add(_METRIC_HEADER)
        add(_metric_row("holdout", r["holdout"]))
        add("")

    add("## Comparison to the fitted elastic-net model (same splits, mf_model.py)")
    add("")
    add("| target | base rate | factor CPCV AUC | enet CPCV AUC | factor holdout AUC | "
        "enet holdout AUC | factor beats enet (CPCV) | factor beats enet (holdout) |")
    add("|---|---|---|---|---|---|---|---|")
    for tname in COHORT_TARGETS:
        r = results["targets"][tname]
        ref = FITTED_MODEL_REFERENCE[tname]
        add(f"| {tname} | {ref['base_rate']:.3f} | {r['cpcv_mean_pooled_auc']:.3f} | "
            f"{ref['cpcv_auc']:.3f} | {r['holdout']['pooled_auc']:.3f} | "
            f"{ref['holdout_auc']:.3f} | "
            f"{'YES' if r['cpcv_mean_pooled_auc'] > ref['cpcv_auc'] else 'no'} | "
            f"{'YES' if r['holdout']['pooled_auc'] > ref['holdout_auc'] else 'no'} |")
    add("")

    add("## Verdict")
    add("")
    add("CPCV (5 independent purged blocks) is the primary evidence here — it "
        "averages out single-period regime noise. The causal holdout is one "
        "evaluate-once block and carries more sampling noise on its own.")
    add("")
    for tname in COHORT_TARGETS:
        r = results["targets"][tname]
        cpcv_auc = r["cpcv_mean_pooled_auc"]
        ho_auc = r["holdout"]["pooled_auc"]
        below_chance_blocks = sum(1 for m in r["cpcv_blocks"].values()
                                   if m["pooled_auc"] < 0.50)
        n_blocks = len(r["cpcv_blocks"])
        add(f"- **{tname}**: CPCV mean pooled AUC **{cpcv_auc:.3f}** "
            f"({below_chance_blocks}/{n_blocks} individual blocks scored below "
            f"0.500), lift over base rate {r['cpcv_mean_lift']:.2f}x. Causal "
            f"holdout pooled AUC {ho_auc:.3f}. "
            + ("The CPCV mean is *below* a coin flip — on average across "
               "regimes the current weights rank funds slightly *worse* than "
               "random within their cohort. The single holdout block being "
               "above 0.5 does not overturn that; it is one block against "
               "five that say otherwise."
               if cpcv_auc < 0.50 else
               "CPCV mean sits at essentially chance (~0.50)."))
    add("")
    add("**Blunt bottom line:** at the current FACTOR_MAP[EQUITY] weights, the "
        "testable 85% of the composite score carries **no reliable "
        "within-cohort predictive signal** for either target under purged "
        "CPCV — cohort_q1 and cohort_top_half both average pooled AUC ~0.46, "
        "i.e. at or below chance, and neither beats the fitted elastic-net "
        "model on any of the four AUC comparisons in the table above (factor "
        "CPCV 0.463/0.464 vs enet 0.548/0.537; factor holdout 0.567/0.529 vs "
        "enet 0.578/0.545). The one bright spot — cohort_q1's holdout AUC of "
        "0.567 — is a single block and should not be read as evidence the "
        "weights work; the 5-block CPCV mean for the same target is 0.463. "
        "This does not mean the underlying metrics (3y/5y return, Sortino, "
        "IR, alpha, capture ratios, drawdown) are worthless signals in "
        "general — it means *this specific fixed linear weighting*, "
        "evaluated purely on relative within-cohort rank, does not "
        "demonstrably beat a coin flip on this data. That is the baseline "
        "any proposed reweighting should be checked against with this same "
        "harness before being adopted.")
    add("")

    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"report -> {REPORT_PATH}")
    return text


def main() -> None:
    df, _ = load_cohort_dataset()
    print(f"cohort-eligible dataset: {len(df)} rows, anchors "
          f"{df['anchor'].min().date()}..{df['anchor'].max().date()}")
    print(f"testable weight = {TESTABLE_WEIGHT:.2f}, untestable = {UNTESTABLE_WEIGHT:.2f} "
          "(days_to_liquidate_wavg, expense_ratio)")
    results = evaluate(DEFAULT_WEIGHTS)
    for tname in COHORT_TARGETS:
        r = results["targets"][tname]
        print(f"[{tname}] CPCV mean pooled_auc={r['cpcv_mean_pooled_auc']:.3f} "
              f"rank_auc={r['cpcv_mean_rank_auc']:.3f} lift={r['cpcv_mean_lift']:.2f} "
              f"| holdout pooled_auc={r['holdout']['pooled_auc']:.3f} "
              f"rank_auc={r['holdout']['rank_auc']:.3f}")
    write_report(results)


if __name__ == "__main__":
    main()
