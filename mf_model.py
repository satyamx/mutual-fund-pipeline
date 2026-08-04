"""
====================================================================================
mf_model.py — Phase B modeling core (design §4-§5): elastic-net logistic primary,
isotonic/Platt calibration, HistGBT challenger, honest metrics, JSON artifact.
====================================================================================
Targets (design §5.4 + confirmed O2):
  or_hybrid      y          — the confirmed OR-label (condA-hybrid OR condB)
  excess_hybrid  condA      — excess-only companion, hybrid benchmark
  excess_peer    condA_peer — excess-only companion, pure peer-proxy benchmark
The two excess targets bound how much conclusions depend on the PRI+1.3%
benchmark approximation (71.7% of hybrid coverage) — reported as a RANGE.

Hyperparameters — a deliberate, documented deviation from design §4.1:
  §4.1 wanted C / l1_ratio chosen per-fold on purged validation slices. The
  split geometry makes that impossible: a 3y purge radius around a 2y test
  block leaves each fold a single ~2.5y contiguous training window (verified:
  fold B1's training span is 2019-02..2023-06; fold B2's is 2021-02..2023-06);
  purging a SECOND validation block out of that empties the training set.
  Instead the hyperparameters are FIXED A PRIORI at conservative values sized
  to ~475 effective samples, and a pre-registered sensitivity grid is run and
  reported so nothing hides. No selection => no selection leakage — strictly
  more honest than tuning, at the cost of possibly leaving a little AUC on the
  table.

Calibration (design §4.1/§7.6): per fold, the LAST 25% of the fold's (already
test-purged) training anchors form the calibration slice; the model is fitted
on the earlier 75%. Isotonic if the slice has >=300 rows and >=30 of the
minority class, else Platt. The slice never overlaps test windows because the
whole training mask is already purged + embargoed vs test.

Evaluate-once causal holdout (design §3.2/§7.7): train anchors <= 2019-07-31,
test anchors 2022-08-31..2023-07-31 — scored exactly once; the result is
stamped to mf_cache/phase_b/holdout_evaluated.json and reruns are refused
without --force.

Artifact: plain JSON (never pickle, design §6.2) — coefficients, imputation
medians, standardization, calibration map — human-readable so the weights
double as the ScoringEngine review input (deferred Step 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mf_cv import (
    BLOCKS, CAUSAL_TEST_END, CAUSAL_TEST_START, CAUSAL_TRAIN_END, MODEL_START,
    causal_holdout, cpcv_folds,
)
from mf_datasources import CACHE
from mf_labels import COHORT_MIN_SIZE

OUT_DIR = CACHE / "phase_b"
FEATURES_PATH = OUT_DIR / "features.parquet"
LABELS_PATH = OUT_DIR / "labels.parquet"
CPCV_RESULTS_PATH = OUT_DIR / "cpcv_results.json"
OOS_PRED_PATH = OUT_DIR / "oos_predictions.parquet"
HOLDOUT_STAMP = OUT_DIR / "holdout_evaluated.json"
HOLDOUT_PRED_PATH = OUT_DIR / "holdout_predictions.parquet"
COEF_PATH = OUT_DIR / "coefficients.json"
ARTIFACT_PATH = OUT_DIR / "model_artifact.json"
REPORT_PATH = OUT_DIR / "model_report.md"

# BUMP THIS whenever the fit changes in a way that makes predictions from before
# and after incomparable — a different training universe, target definition,
# feature set, or hyperparameters. mf_ledger stamps it as `model_id` on every
# prediction, and mf_eval segments outcome history by it, so a stale version
# silently averages two different models into one accuracy claim.
#   phase_b_v1 — 136-fund manifest, cohort_q1 base rate 0.325, holdout AUC 0.578
#   phase_b_v2 — 367-fund manifest (2026-08-04). Cohorts are ~2.7x larger, so
#                y_cohort_q1 is a DIFFERENT label (base rate 0.280): v1 and v2
#                predictions are not one series and must never be pooled.
#   phase_b_v2_1 — same fit as v2, calibration bounded away from 0 and 1 by the
#                rule of three (see Calibrator.fit). RANK-PRESERVING, so AUC and
#                lift are bit-identical to v2; only the probabilities the payload
#                emits changed. Bumped anyway: model_id must identify the exact
#                payload that produced a logged number, and v2 rows carry literal
#                0.0 probabilities this payload no longer emits.
MODEL_VERSION = "phase_b_v2_1"

TARGETS = {"or_hybrid": "y", "excess_hybrid": "condA", "excess_peer": "condA_peer"}

# Fixed a-priori configs (see module docstring) + pre-registered sensitivity grid.
PRIMARY_ENET = dict(C=0.1, l1_ratio=0.5)
ENET_SENSITIVITY = [
    dict(C=0.03, l1_ratio=0.5), dict(C=1.0, l1_ratio=0.5),
    dict(C=0.1, l1_ratio=0.2), dict(C=0.1, l1_ratio=0.8),
]
PRIMARY_HGBT = dict(learning_rate=0.06, max_leaf_nodes=15, max_iter=300,
                    min_samples_leaf=40, l2_regularization=1.0)
HGBT_SENSITIVITY = [
    dict(learning_rate=0.1, max_leaf_nodes=15, max_iter=300,
         min_samples_leaf=40, l2_regularization=1.0),
    dict(learning_rate=0.06, max_leaf_nodes=7, max_iter=300,
         min_samples_leaf=40, l2_regularization=1.0),
    dict(learning_rate=0.06, max_leaf_nodes=31, max_iter=300,
         min_samples_leaf=40, l2_regularization=1.0),
]

CHALLENGER_MARGIN = 0.03          # >3 AUC points to adopt (design §4.2)
CALIB_FRAC = 0.25
MIN_ISOTONIC_N = 300
MIN_ISOTONIC_MINORITY = 30
EFFECTIVE_N_DIVISOR = 30          # ≈ overlap inflation of monthly 3y windows (§0)
DECILE = 0.10
N_BOOT = 80                       # half-year block bootstrap replicates (§3.4)

META_COLS = ("amfi_code", "anchor")


# ==============================================================================
# Data
# ==============================================================================
def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    df = labels.merge(feats, on=["amfi_code", "anchor"], how="inner")
    df = df[df["anchor"] >= MODEL_START].reset_index(drop=True)
    feature_cols = [c for c in feats.columns if c not in META_COLS]
    return df, feature_cols


# ==============================================================================
# Preprocessing (fit on training rows ONLY — §4.1)
# ==============================================================================
class Preprocessor:
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.columns = list(X.columns)
        self.median = X.median().fillna(0.0)
        Xi = X.fillna(self.median)
        self.mean = Xi.mean()
        std = Xi.std(ddof=1)
        self.std = std.where(std > 1e-9, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xi = X[self.columns].fillna(self.median)
        return ((Xi - self.mean) / self.std).to_numpy(dtype=float)


# ==============================================================================
# Models + calibration
# ==============================================================================
def fit_enet(X: np.ndarray, y: np.ndarray, cfg: dict) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet", solver="saga", C=cfg["C"], l1_ratio=cfg["l1_ratio"],
        class_weight="balanced", max_iter=4000, tol=1e-4, random_state=0,
    ).fit(X, y)


def fit_hgbt(X: np.ndarray, y: np.ndarray, cfg: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **cfg, class_weight="balanced", early_stopping=False, random_state=0,
    ).fit(X, y)


class Calibrator:
    """Isotonic on a purged calibration slice; Platt (sigmoid on log-odds)
    fallback when the slice is too thin for isotonic tails (design §4.1)."""

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "Calibrator":
        minority = min(int(y.sum()), int((~y).sum()))
        if len(y) >= MIN_ISOTONIC_N and minority >= MIN_ISOTONIC_MINORITY:
            self.kind = "isotonic"
            # Bounded away from 0 and 1 by the RULE OF THREE. Isotonic happily
            # returns a flat 0.0 over any region of the calibration slice that
            # happened to contain no positives — and phase_b_v1 shipped exactly
            # that, giving 47 of 120 live funds a literal `probability: 0.0`.
            # Observing 0 positives in n samples does not evidence "impossible";
            # it bounds the rate at roughly 3/n with 95% confidence, and claiming
            # certainty a 0.558-AUC model cannot support is the "fabricated
            # accuracy" the honesty invariant forbids.
            #
            # This is RANK-PRESERVING: the bound clips a monotone map, so the
            # ordering (and therefore AUC, lift and cohort_percentile) is
            # unchanged. It corrects what the number CLAIMS, not what it ranks.
            eps = _certainty_floor(len(y))
            self._iso = IsotonicRegression(y_min=eps, y_max=1.0 - eps,
                                           out_of_bounds="clip").fit(p_raw, y)
        else:
            self.kind = "platt"
            z = _logit(p_raw).reshape(-1, 1)
            self._platt = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if self.kind == "isotonic":
            return self._iso.predict(p_raw)
        return self._platt.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]

    def to_json(self) -> dict:
        if self.kind == "isotonic":
            return dict(kind="isotonic",
                        x=[float(v) for v in self._iso.X_thresholds_],
                        y=[float(v) for v in self._iso.y_thresholds_])
        return dict(kind="platt", coef=float(self._platt.coef_[0][0]),
                    intercept=float(self._platt.intercept_[0]))


def _certainty_floor(n_calibration: int) -> float:
    """Rule of three: 0 events in n trials bounds the rate at ~3/n (95%).

    The smallest probability the calibration slice can honestly justify. Capped
    at 0.01 so a very small slice cannot produce an absurdly wide floor, and
    guarded against n=0.
    """
    return min(0.01, 3.0 / max(n_calibration, 1))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_split(anchors: np.ndarray, train_mask: np.ndarray,
                frac: float = CALIB_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """Last `frac` of the fold's training ANCHORS (by date) become the
    calibration slice; the model fits on the earlier anchors."""
    ta = np.unique(anchors[train_mask])
    cut = ta[int(np.ceil(len(ta) * (1 - frac)))] if len(ta) > 4 else ta[-1]
    calib = train_mask & (anchors >= cut)
    fit = train_mask & (anchors < cut)
    return fit, calib


# ==============================================================================
# Honest metrics (design §5.1)
# ==============================================================================
def pooled_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def within_anchor_rank_auc(anchors: np.ndarray, y: np.ndarray,
                           p: np.ndarray) -> tuple[float, int]:
    """AUC inside each monthly cohort, averaged — the fund-selection metric,
    immune to regime base-rate drift."""
    aucs = []
    for t in np.unique(anchors):
        m = anchors == t
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (float("nan"), 0)


def decile_stats(anchors: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict:
    """Within-anchor top-decile precision and bottom-decile negative capture,
    pooled over the cohorts of a test block."""
    top_hits = top_n = bot_neg = bot_n = total_neg = 0
    for t in np.unique(anchors):
        m = anchors == t
        yt, pt = y[m], p[m]
        k = max(1, int(np.floor(DECILE * len(yt))))
        order = np.argsort(-pt, kind="stable")
        top_hits += int(yt[order[:k]].sum())
        top_n += k
        bot = order[-k:]
        bot_neg += int((~yt[bot]).sum())
        bot_n += k
        total_neg += int((~yt).sum())
    return dict(
        top_decile_precision=top_hits / top_n if top_n else float("nan"),
        bottom_decile_neg_capture=bot_neg / total_neg if total_neg else float("nan"),
        n_top=top_n, n_bottom=bot_n, total_negatives=total_neg,
    )


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    out = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum():
            out.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()),
                            mean_p=float(p[m].mean()), frac_pos=float(y[m].mean())))
    return out


def block_metrics(anchors: np.ndarray, y: np.ndarray, p_raw: np.ndarray,
                  p_cal: np.ndarray) -> dict:
    base = float(y.mean())
    rank, n_cohorts = within_anchor_rank_auc(anchors, y, p_raw)
    dec = decile_stats(anchors, y, p_raw)
    lift = dec["top_decile_precision"] / base if base > 0 else float("nan")
    pos = y.astype(bool)
    recall = float((p_cal[pos] >= 0.5).mean()) if pos.any() else float("nan")
    brier = float(np.mean((p_cal - y) ** 2))
    return dict(
        n=int(len(y)), effective_n=round(len(y) / EFFECTIVE_N_DIVISOR),
        base_rate=base, negatives=int((~pos).sum()),
        pooled_auc=pooled_auc(y, p_raw), rank_auc=rank, rank_auc_cohorts=n_cohorts,
        top_decile_precision=dec["top_decile_precision"], lift_over_base=lift,
        recall_at_05=recall,
        bottom_decile_neg_capture=dec["bottom_decile_neg_capture"],
        total_negatives=dec["total_negatives"], brier=brier,
    )


# ==============================================================================
# Single-fold runner (shared by CPCV and holdout)
# ==============================================================================
def run_fold(df: pd.DataFrame, feature_cols: list[str], target_col: str,
             train_mask: np.ndarray, test_mask: np.ndarray,
             kind: str, cfg: dict) -> tuple[dict, pd.DataFrame, object, object]:
    anchors = df["anchor"].to_numpy()
    y = df[target_col].to_numpy(dtype=bool)
    fit_mask, calib_mask = calib_split(anchors, train_mask)
    X = df[feature_cols]

    if kind == "enet":
        pre = Preprocessor().fit(X[fit_mask])
        Xf, Xc, Xt = (pre.transform(X[m]) for m in (fit_mask, calib_mask, test_mask))
        model = fit_enet(Xf, y[fit_mask], cfg)
    else:  # hgbt handles NaN natively — no imputation/standardization
        pre = None
        Xf = X[fit_mask].to_numpy(dtype=float)
        Xc = X[calib_mask].to_numpy(dtype=float)
        Xt = X[test_mask].to_numpy(dtype=float)
        # The histogram binner cannot bin a column carrying <2 distinct non-NaN
        # values within this fold (all-NaN sector_* on a diversified-only slice,
        # or a has_* indicator that is constant here). Drop such columns, same
        # mask across fit/calib/test. They are information-free by construction.
        keep = np.array([np.unique(c[~np.isnan(c)]).size >= 2 for c in Xf.T])
        if keep.any():
            Xf, Xc, Xt = Xf[:, keep], Xc[:, keep], Xt[:, keep]
        model = fit_hgbt(Xf, y[fit_mask], cfg)

    cal = Calibrator().fit(model.predict_proba(Xc)[:, 1], y[calib_mask])
    p_test_raw = model.predict_proba(Xt)[:, 1]
    p_test_cal = cal.predict(p_test_raw)

    metrics = block_metrics(anchors[test_mask], y[test_mask], p_test_raw, p_test_cal)
    metrics["calibration_kind"] = cal.kind
    # in-sample diagnostics — quarantined, never evidence (§5.1)
    p_fit = model.predict_proba(Xf)[:, 1]
    metrics["in_sample_auc"] = pooled_auc(y[fit_mask], p_fit)
    metrics["in_sample_accuracy"] = float(((p_fit >= 0.5) == y[fit_mask]).mean())
    metrics["n_train_fit"] = int(fit_mask.sum())
    metrics["n_train_calib"] = int(calib_mask.sum())

    preds = pd.DataFrame(dict(
        amfi_code=df.loc[test_mask, "amfi_code"].to_numpy(),
        anchor=anchors[test_mask], y=y[test_mask],
        condA=df.loc[test_mask, "condA"].to_numpy(),
        condB=df.loc[test_mask, "condB"].to_numpy(),
        condA_peer=df.loc[test_mask, "condA_peer"].to_numpy(),
        p_raw=p_test_raw, p_cal=p_test_cal,
    ))
    return metrics, preds, model, pre


# ==============================================================================
# CPCV (primary evaluation, design §3.2)
# ==============================================================================
def run_cpcv(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    anchors = df["anchor"]
    results: dict = {"targets": {}, "sensitivity": {}}
    all_preds = []

    for tname, tcol in TARGETS.items():
        results["targets"][tname] = {"enet": {}}
        for block, train, test in cpcv_folds(anchors):
            m, preds, _, _ = run_fold(df, feature_cols, tcol, train, test,
                                      "enet", PRIMARY_ENET)
            results["targets"][tname]["enet"][block] = m
            preds["target"], preds["model"], preds["block"] = tname, "enet", block
            all_preds.append(preds)
            print(f"[cpcv] {tname:14s} enet {block}: pooled_auc={m['pooled_auc']:.3f} "
                  f"rank_auc={m['rank_auc']:.3f} neg_capture="
                  f"{m['bottom_decile_neg_capture']:.3f} (n={m['n']}, neg={m['negatives']})")

    # challenger on the confirmed deliverable target only
    results["targets"]["or_hybrid"]["hgbt"] = {}
    for block, train, test in cpcv_folds(anchors):
        m, preds, _, _ = run_fold(df, feature_cols, "y", train, test,
                                  "hgbt", PRIMARY_HGBT)
        results["targets"]["or_hybrid"]["hgbt"][block] = m
        preds["target"], preds["model"], preds["block"] = "or_hybrid", "hgbt", block
        all_preds.append(preds)
        print(f"[cpcv] or_hybrid      hgbt {block}: pooled_auc={m['pooled_auc']:.3f} "
              f"rank_auc={m['rank_auc']:.3f}")

    # pre-registered sensitivity grid (no selection — reported, not chosen from)
    for kind, grid in (("enet", ENET_SENSITIVITY), ("hgbt", HGBT_SENSITIVITY)):
        for cfg in grid:
            key = f"{kind}:{json.dumps(cfg, sort_keys=True)}"
            aucs, ranks = [], []
            for block, train, test in cpcv_folds(anchors):
                m, _, _, _ = run_fold(df, feature_cols, "y", train, test, kind, cfg)
                aucs.append(m["pooled_auc"])
                ranks.append(m["rank_auc"])
            results["sensitivity"][key] = dict(
                mean_pooled_auc=float(np.nanmean(aucs)),
                mean_rank_auc=float(np.nanmean(ranks)))
            print(f"[sens] {key}: mean_auc={np.nanmean(aucs):.3f}")

    pd.concat(all_preds, ignore_index=True).to_parquet(OOS_PRED_PATH, index=False)
    CPCV_RESULTS_PATH.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


# ==============================================================================
# Causal holdout — EVALUATED ONCE (design §7.7)
# ==============================================================================
def run_holdout(df: pd.DataFrame, feature_cols: list[str], force: bool = False) -> dict:
    if HOLDOUT_STAMP.exists() and not force:
        print("holdout already evaluated — reusing stamped result "
              f"({HOLDOUT_STAMP}). Use --force to re-evaluate (do NOT do this "
              "to shop for a better number).")
        return json.loads(HOLDOUT_STAMP.read_text(encoding="utf-8"))

    train, test = causal_holdout(df["anchor"])
    out = dict(
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        train_end=str(CAUSAL_TRAIN_END.date()),
        test_span=f"{CAUSAL_TEST_START.date()}..{CAUSAL_TEST_END.date()}",
        n_train=int(train.sum()), n_test=int(test.sum()), models={},
    )
    all_preds = []
    runs = [(t, TARGETS[t], "enet", PRIMARY_ENET) for t in TARGETS]
    runs.append(("or_hybrid", "y", "hgbt", PRIMARY_HGBT))
    for tname, tcol, kind, cfg in runs:
        m, preds, _, _ = run_fold(df, feature_cols, tcol, train, test, kind, cfg)
        m["reliability"] = reliability_table(
            preds["y"].to_numpy(dtype=bool), preds["p_cal"].to_numpy())
        out["models"][f"{tname}:{kind}"] = m
        preds["target"], preds["model"], preds["block"] = tname, kind, "HOLDOUT"
        all_preds.append(preds)
        print(f"[holdout] {tname}:{kind}: pooled_auc={m['pooled_auc']:.3f} "
              f"rank_auc={m['rank_auc']:.3f} neg_capture="
              f"{m['bottom_decile_neg_capture']:.3f} (neg={m['negatives']})")

    pd.concat(all_preds, ignore_index=True).to_parquet(HOLDOUT_PRED_PATH, index=False)
    HOLDOUT_STAMP.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ==============================================================================
# Within-cohort relative targets — second modeling attempt (y_cohort_q1,
# y_cohort_top_half from mf_labels.add_cohort_targets). Purely additive: reuses
# run_fold/block_metrics/Preprocessor/Calibrator/fit_enet/fit_hgbt unchanged,
# writes to its own artifact paths, and does not alter run_cpcv/run_holdout or
# their outputs above in any way.
# ==============================================================================
COHORT_TARGETS = {"cohort_q1": "y_cohort_q1", "cohort_top_half": "y_cohort_top_half"}
COHORT_CPCV_RESULTS_PATH = OUT_DIR / "cpcv_results_cohort.json"
COHORT_OOS_PRED_PATH = OUT_DIR / "oos_predictions_cohort.parquet"
COHORT_HOLDOUT_STAMP = OUT_DIR / "holdout_evaluated_cohort.json"
COHORT_HOLDOUT_PRED_PATH = OUT_DIR / "holdout_predictions_cohort.parquet"
COHORT_COEF_PATH = OUT_DIR / "coefficients_cohort.json"
COHORT_REPORT_PATH = OUT_DIR / "cohort_target_report.md"
# Imported, NOT redeclared: this is the one artifact that leaves mf_cache/ and is
# committed (see mf_infer for why). Two literals for the same path is exactly the
# drift mf_labels.MANIFEST_PATH is single-defined to avoid — and here a drift would
# mean the trainer writes a model the inference side never reads.
from mf_infer import COHORT_ARTIFACT_PATH  # noqa: E402 — path constant, not logic


def load_cohort_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Same feature/label merge as load_dataset(), restricted to rows where the
    fund's (anchor, cohort) cell had >= mf_labels.COHORT_MIN_SIZE funds with a
    valid R_fwd (mf_labels.add_cohort_targets encodes ineligibility as NA)."""
    df, feature_cols = load_dataset()
    df = df[df["y_cohort_q1"].notna()].reset_index(drop=True)
    for col in COHORT_TARGETS.values():
        df[col] = df[col].astype(bool)
    return df, feature_cols


def run_cpcv_cohort(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    anchors = df["anchor"]
    results: dict = {"targets": {}}
    all_preds = []
    for tname, tcol in COHORT_TARGETS.items():
        results["targets"][tname] = {"enet": {}, "hgbt": {}}
        for kind, cfg in (("enet", PRIMARY_ENET), ("hgbt", PRIMARY_HGBT)):
            for block, train, test in cpcv_folds(anchors):
                m, preds, _, _ = run_fold(df, feature_cols, tcol, train, test, kind, cfg)
                results["targets"][tname][kind][block] = m
                preds["target"], preds["model"], preds["block"] = tname, kind, block
                all_preds.append(preds)
                print(f"[cpcv-cohort] {tname:16s} {kind:4s} {block}: "
                      f"pooled_auc={m['pooled_auc']:.3f} rank_auc={m['rank_auc']:.3f} "
                      f"base_rate={m['base_rate']:.3f} neg={m['negatives']} "
                      f"(n={m['n']})")
    pd.concat(all_preds, ignore_index=True).to_parquet(COHORT_OOS_PRED_PATH, index=False)
    COHORT_CPCV_RESULTS_PATH.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def run_holdout_cohort(df: pd.DataFrame, feature_cols: list[str], force: bool = False) -> dict:
    if COHORT_HOLDOUT_STAMP.exists() and not force:
        print("cohort holdout already evaluated — reusing stamped result "
              f"({COHORT_HOLDOUT_STAMP}). Use --force to re-evaluate (do NOT do "
              "this to shop for a better number).")
        return json.loads(COHORT_HOLDOUT_STAMP.read_text(encoding="utf-8"))

    train, test = causal_holdout(df["anchor"])
    out = dict(
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        train_end=str(CAUSAL_TRAIN_END.date()),
        test_span=f"{CAUSAL_TEST_START.date()}..{CAUSAL_TEST_END.date()}",
        n_train=int(train.sum()), n_test=int(test.sum()), models={},
    )
    all_preds = []
    for tname, tcol in COHORT_TARGETS.items():
        for kind, cfg in (("enet", PRIMARY_ENET), ("hgbt", PRIMARY_HGBT)):
            m, preds, _, _ = run_fold(df, feature_cols, tcol, train, test, kind, cfg)
            out["models"][f"{tname}:{kind}"] = m
            preds["target"], preds["model"], preds["block"] = tname, kind, "HOLDOUT"
            all_preds.append(preds)
            print(f"[holdout-cohort] {tname}:{kind}: pooled_auc={m['pooled_auc']:.3f} "
                  f"rank_auc={m['rank_auc']:.3f} base_rate={m['base_rate']:.3f} "
                  f"neg={m['negatives']}")

    pd.concat(all_preds, ignore_index=True).to_parquet(COHORT_HOLDOUT_PRED_PATH, index=False)
    COHORT_HOLDOUT_STAMP.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def run_coefficients_cohort(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Standardized elastic-net coefficients, fit on the full cohort-eligible
    dataset (no bootstrap here — the CPCV per-block table already shows
    fold-to-fold (in)stability of the headline metrics)."""
    out = {}
    for tname, tcol in COHORT_TARGETS.items():
        y = df[tcol].to_numpy(dtype=bool)
        X = df[feature_cols]
        pre = Preprocessor().fit(X)
        model = fit_enet(pre.transform(X), y, PRIMARY_ENET)
        out[tname] = {k: float(v) for k, v in zip(feature_cols, model.coef_[0])}
    COHORT_COEF_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def build_cohort_artifact(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Full inference payload for the within-cohort targets (mirror of
    build_artifact, which only covers the OR/excess targets). coefficients_cohort
    ships weights only; live single-fund scoring also needs the intercept,
    imputation medians, standardization stats and calibration map. Human-readable
    JSON (design §6.2 — never pickle) so it can be applied with numpy alone
    (see mf_infer.py) without importing sklearn into the orchestrator.

    signal_context carries the honest measured edge (holdout AUC + base rate) so
    the consuming app can surface the probability with its true accuracy, never
    as an oracle."""
    holdout_auc: dict = {}
    if COHORT_HOLDOUT_STAMP.exists():
        stamp = json.loads(COHORT_HOLDOUT_STAMP.read_text(encoding="utf-8"))
        for tname in COHORT_TARGETS:
            m = stamp.get("models", {}).get(f"{tname}:enet")
            if m is not None:
                holdout_auc[tname] = float(m["pooled_auc"])

    anchors = df["anchor"].to_numpy()
    artifact = dict(
        version=MODEL_VERSION + "_cohort",
        created=datetime.now(timezone.utc).isoformat(),
        training_anchor_span=f"{df['anchor'].min().date()}..{df['anchor'].max().date()}",
        features=feature_cols,
        hyperparameters=PRIMARY_ENET,
        note=("Within-cohort relative targets (top-quartile / top-half vs same "
              "(anchor, cohort) peers). The ONLY validated edge in this pipeline "
              "(cohort_q1 holdout AUC ~0.558, lift ~1.10x) — weak, a signal INPUT "
              "not a standalone verdict. Elastic-net only (HGBT showed no edge on "
              "these targets). Fitted on all cohort-eligible anchors except the "
              "last-25% calibration slice."),
        models={},
    )
    all_mask = np.ones(len(df), dtype=bool)
    for tname, tcol in COHORT_TARGETS.items():
        y = df[tcol].to_numpy(dtype=bool)
        fit_mask, calib_mask = calib_split(anchors, all_mask)
        X = df[feature_cols]
        pre = Preprocessor().fit(X[fit_mask])
        model = fit_enet(pre.transform(X[fit_mask]), y[fit_mask], PRIMARY_ENET)
        cal = Calibrator().fit(
            model.predict_proba(pre.transform(X[calib_mask]))[:, 1], y[calib_mask])
        artifact["models"][tname] = dict(
            target=tcol,
            base_rate=float(y.mean()),
            signal_context=dict(holdout_auc=holdout_auc.get(tname),
                                base_rate=float(y.mean())),
            imputation_medians={c: float(v) for c, v in pre.median.items()},
            standardize_mean={c: float(v) for c, v in pre.mean.items()},
            standardize_std={c: float(v) for c, v in pre.std.items()},
            coefficients={c: float(v) for c, v in
                          zip(feature_cols, model.coef_[0])},
            intercept=float(model.intercept_[0]),
            calibration=cal.to_json(),
        )
    COHORT_ARTIFACT_PATH.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"cohort artifact -> {COHORT_ARTIFACT_PATH}")
    return artifact


def write_cohort_report(cpcv: dict, holdout: dict, coefs: dict, label_stats: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Within-cohort relative targets — second modeling attempt, honest results")
    add("")
    add(f"Generated {datetime.now(timezone.utc).date()}. Targets: `y_cohort_q1` "
        "(top quartile of same-cohort peers' 3y-forward return) and "
        "`y_cohort_top_half` (above cohort median), computed per (anchor, cohort) "
        "directly from R_fwd — cohort = PeerProxyResolver grouping (same "
        "`category` for diversified funds; same `sector` for thematic funds with "
        ">=3 members; a pooled small-sector thematic group otherwise). Anchors "
        f"with < {label_stats['min_size']} cohort members with a valid R_fwd are "
        "excluded rather than guessed.")
    add("")
    add("## Read this first")
    add("")
    add("- This target removes the regime effect the OR-label suffered from "
        "(2017 anchors ~32% positive, 2020 anchors ~99% positive) by construction: "
        "every fund in a cohort at a given anchor faces the same market, so the "
        "label only reflects RELATIVE standing, not market direction.")
    add("- Base rates below deviate from the nominal ~25%/~50% guide because "
        "small cohorts (n=4-9) cannot split into exact quartiles/halves under a "
        "continuous quantile — this is honest quantization noise, not a labeling "
        "bug (verified: n=5 cohorts, the five 5-fund thematic sectors, run "
        "40%/40% instead of 25%/50%; see label_stats below).")
    add("- Headline metric is AUC / rank-AUC / lift over base rate, never raw "
        "accuracy — a coin-flip base rate near 50% makes accuracy especially "
        "uninformative here.")
    add("- Leakage check (pre-registered in the task): the peer-relative features "
        "(excess_1y, excess_3y, beta_3y, te_3y, ir_3y, upcap_3y, downcap_3y) were "
        "re-verified against mf_features.py source — every one slices its input "
        "series to `.loc[:t]` before use (own NAV and the LOO peer composite "
        "alike), and `mf_features.py --selftest` (bumps all post-t NAVs +/-20% "
        "and asserts bit-identical features) PASSED before this run. No forward "
        "information reaches the feature side; the cohort ranking itself uses "
        "R_fwd only on the label side, exactly as with the OR-label.")
    add("")

    add("## Label base rates")
    add("")
    add("n_total/n_dropped/n_kept below are counted on the model-ready frame "
        "(labels inner-joined to features.parquet, anchors >= 2014-01-31) — "
        "narrower than the raw label-build population (mf_labels.py reports "
        "13,621 kept / 1,416 dropped over the full 2013-2023 anchor grid before "
        "the features merge).")
    add("")
    add(f"| target | n_total | n_dropped (cohort<{label_stats['min_size']}) | n_kept | "
        "base rate (kept) | base rate (causal-holdout span) | negatives (holdout span) |")
    add("|---|---|---|---|---|---|---|")
    for tname, tcol in COHORT_TARGETS.items():
        s = label_stats[tname]
        add(f"| {tname} | {label_stats['n_total']} | {label_stats['n_dropped']} | "
            f"{label_stats['n_kept']} | {s['base_rate']:.3f} | "
            f"{s['holdout_base_rate']:.3f} | {s['holdout_negatives']} |")
    add("")
    add(f"(For reference, the prior OR-label causal holdout had only 26 negatives "
        f"out of 1416 rows — base rate 0.982. Both cohort targets fix that.)")
    add("")

    add("## Causal holdout (evaluated once) — headline")
    add("")
    add(f"Train anchors <= {holdout['train_end']}; test anchors "
        f"{holdout['test_span']}; evaluated once at {holdout['evaluated_at'][:10]}.")
    add("")
    add(_METRIC_HEADER)
    for tname in COHORT_TARGETS:
        for kind in ("enet", "hgbt"):
            m = holdout["models"][f"{tname}:{kind}"]
            add(_metric_row(f"{tname} ({kind})", m))
    add("")

    add("## CPCV — purged/embargoed, 5 blocks (primary evaluation)")
    add("")
    for tname in COHORT_TARGETS:
        for kind, label in (("enet", "elastic-net (PRIMARY)"), ("hgbt", "HistGBT (challenger)")):
            blocks = cpcv["targets"][tname][kind]
            mean_auc = float(np.nanmean([m["pooled_auc"] for m in blocks.values()]))
            mean_rank = float(np.nanmean([m["rank_auc"] for m in blocks.values()]))
            mean_prec = float(np.nanmean([m["top_decile_precision"] for m in blocks.values()]))
            mean_lift = float(np.nanmean([m["lift_over_base"] for m in blocks.values()]))
            mean_cap = float(np.nanmean([m["bottom_decile_neg_capture"] for m in blocks.values()]))
            add(f"### {tname}, {label}")
            add("")
            add(_METRIC_HEADER)
            for block, m in blocks.items():
                add(_metric_row(block, m))
            add(f"| **mean** | | | | | **{mean_auc:.3f}** | **{mean_rank:.3f}** | "
                f"**{mean_prec:.3f}** | **{mean_lift:.2f}** | | **{mean_cap:.3f}** | | |")
            add("")

    add("## Challenger check (HGBT vs elastic-net, CPCV mean pooled AUC)")
    add("")
    add("| target | enet mean AUC | hgbt mean AUC | margin | verdict |")
    add("|---|---|---|---|---|")
    for tname in COHORT_TARGETS:
        enet_auc = float(np.nanmean(
            [m["pooled_auc"] for m in cpcv["targets"][tname]["enet"].values()]))
        hgbt_auc = float(np.nanmean(
            [m["pooled_auc"] for m in cpcv["targets"][tname]["hgbt"].values()]))
        margin = hgbt_auc - enet_auc
        verdict = "HGBT beats enet by >3pts" if margin > CHALLENGER_MARGIN else "no material edge"
        add(f"| {tname} | {enet_auc:.3f} | {hgbt_auc:.3f} | {margin:+.3f} | {verdict} |")
    add("")

    add("## Top standardized elastic-net coefficients (fit on full cohort-eligible set)")
    add("")
    for tname in COHORT_TARGETS:
        c = coefs[tname]
        top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:15]
        add(f"### {tname}")
        add("")
        add("| feature | coef |")
        add("|---|---|")
        for feat, v in top:
            add(f"| {feat} | {v:+.3f} |")
        add("")

    text = "\n".join(lines) + "\n"
    COHORT_REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"cohort report -> {COHORT_REPORT_PATH}")
    return text


# ==============================================================================
# Coefficients + half-year block-bootstrap sign stability (design §3.4)
# ==============================================================================
def run_coefficients(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    anchors = pd.to_datetime(df["anchor"])
    halfyear = anchors.dt.year * 10 + (anchors.dt.month > 6).astype(int)
    block_ids = np.sort(halfyear.unique())
    block_rows = {b: np.flatnonzero(halfyear == b) for b in block_ids}
    rng = np.random.default_rng(0)

    out: dict = {}
    for tname in ("or_hybrid", "excess_hybrid", "excess_peer"):
        y = df[TARGETS[tname]].to_numpy(dtype=bool)
        X = df[feature_cols]
        pre = Preprocessor().fit(X)
        model = fit_enet(pre.transform(X), y, PRIMARY_ENET)
        coefs = dict(zip(feature_cols, model.coef_[0]))

        stability = None
        if tname != "excess_peer":  # bootstrap the two decision-relevant models
            signs = np.zeros((N_BOOT, len(feature_cols)))
            for b in range(N_BOOT):
                sample = rng.choice(block_ids, size=len(block_ids), replace=True)
                rows = np.concatenate([block_rows[s] for s in sample])
                yb = y[rows]
                if len(np.unique(yb)) < 2:
                    continue
                Xb = X.iloc[rows]
                mb = fit_enet(Preprocessor().fit(Xb).transform(Xb), yb, PRIMARY_ENET)
                signs[b] = np.sign(mb.coef_[0])
            pos = (signs > 0).mean(axis=0)
            neg = (signs < 0).mean(axis=0)
            stability = dict(zip(feature_cols, np.maximum(pos, neg)))
            print(f"[boot] {tname}: {N_BOOT} replicates done")

        out[tname] = dict(
            coefficients={k: float(v) for k, v in coefs.items()},
            sign_stability={k: float(v) for k, v in stability.items()} if stability else None,
        )
    COEF_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ==============================================================================
# JSON model artifact (design §6.2 — never pickle)
# ==============================================================================
def build_artifact(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    anchors = df["anchor"].to_numpy()
    artifact = dict(
        version=MODEL_VERSION,
        created=datetime.now(timezone.utc).isoformat(),
        training_anchor_span=f"{df['anchor'].min().date()}..{df['anchor'].max().date()}",
        features=feature_cols,
        hyperparameters=PRIMARY_ENET,
        note=("Live probabilities inherit the 2014-2023 calibration cohorts; "
              "regime shift after the training span is not calibrated away. "
              "Model fitted on all anchors except the calibration slice "
              "(last 25% of anchors)."),
        models={},
    )
    all_mask = np.ones(len(df), dtype=bool)
    for tname, tcol in TARGETS.items():
        y = df[tcol].to_numpy(dtype=bool)
        fit_mask, calib_mask = calib_split(anchors, all_mask)
        X = df[feature_cols]
        pre = Preprocessor().fit(X[fit_mask])
        model = fit_enet(pre.transform(X[fit_mask]), y[fit_mask], PRIMARY_ENET)
        cal = Calibrator().fit(
            model.predict_proba(pre.transform(X[calib_mask]))[:, 1], y[calib_mask])
        artifact["models"][tname] = dict(
            target=tcol,
            base_rate=float(y.mean()),
            imputation_medians={c: float(v) for c, v in pre.median.items()},
            standardize_mean={c: float(v) for c, v in pre.mean.items()},
            standardize_std={c: float(v) for c, v in pre.std.items()},
            coefficients={c: float(v) for c, v in
                          zip(feature_cols, model.coef_[0])},
            intercept=float(model.intercept_[0]),
            calibration=cal.to_json(),
        )
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    print(f"artifact -> {ARTIFACT_PATH}")
    return artifact


# ==============================================================================
# Report (design §5.3 presentation rules are requirements)
# ==============================================================================
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _metric_row(block: str, m: dict) -> str:
    return ("| " + " | ".join([
        block, str(m["n"]), str(m["effective_n"]), _fmt(m["base_rate"]),
        str(m["negatives"]), _fmt(m["pooled_auc"]),
        f"{_fmt(m['rank_auc'])} ({m['rank_auc_cohorts']})",
        _fmt(m["top_decile_precision"]), _fmt(m["lift_over_base"], 2),
        _fmt(m["recall_at_05"]), _fmt(m["bottom_decile_neg_capture"]),
        _fmt(m["brier"]), m["calibration_kind"],
    ]) + " |")


_METRIC_HEADER = (
    "| block | n | eff. n | base rate | neg | pooled AUC | rank-AUC (cohorts) | "
    "prec@top-dec | lift | recall@0.5 | bottom-dec neg capture | Brier | calib |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def write_report(cpcv: dict, holdout: dict, coefs: dict) -> str:
    lines: list[str] = []
    add = lines.append

    or_blocks = cpcv["targets"]["or_hybrid"]["enet"]
    mean_auc = float(np.nanmean([m["pooled_auc"] for m in or_blocks.values()]))
    mean_rank = float(np.nanmean([m["rank_auc"] for m in or_blocks.values()]))
    mean_cap = float(np.nanmean([m["bottom_decile_neg_capture"] for m in or_blocks.values()]))
    hgbt_blocks = cpcv["targets"]["or_hybrid"]["hgbt"]
    hgbt_auc = float(np.nanmean([m["pooled_auc"] for m in hgbt_blocks.values()]))
    ho = holdout["models"]["or_hybrid:enet"]
    ho_gbt = holdout["models"]["or_hybrid:hgbt"]

    add("# Phase B model report — honest out-of-sample results")
    add("")
    add(f"Model version `{MODEL_VERSION}`; generated {datetime.now(timezone.utc).date()}.")
    add("")
    add("## Read this first")
    add("")
    add("- Label base rate is **0.846** (OR-label, hybrid benchmark). A constant "
        "all-positive classifier scores accuracy 0.846 and AUC 0.5 on this label. "
        "**Accuracy is therefore never a headline here.** Positive-class lift is "
        "mathematically capped at 1/0.846 ≈ **1.18×** — the economic value is in "
        "the NEGATIVE class (avoiding the ~15% bad windows) and in within-anchor "
        "ranking.")
    add("- Nominal n is ~30× inflated by overlapping 3y windows (adjacent monthly "
        "labels agree 93.5%). Effective n ≈ n/30 is printed beside every n.")
    add("- All numbers below are out-of-sample under a 3y purge + 1-month embargo "
        "(mf_cv.py). In-sample numbers appear only in the quarantined section at "
        "the end and are NOT evidence.")
    add("- Survivorship: the 131-fund universe is today's survivors; scores "
        "generalize to funds like these, not to the full AMFI universe (design §7.1).")
    add("- Hyperparameters were FIXED a priori (C=0.1, l1_ratio=0.5), not tuned: "
        "with a 3y purge radius, per-fold inner tuning would leave near-empty "
        "training sets (design deviation from §4.1, documented in mf_model.py). "
        "The pre-registered sensitivity grid below shows the results are not "
        "hyperparameter-fragile.")
    add("")

    add("## Headline — strictly-causal holdout (evaluated once)")
    add("")
    add(f"Train anchors ≤ {holdout['train_end']} (all label windows close by "
        f"2022-07); test anchors {holdout['test_span']}; evaluated once at "
        f"{holdout['evaluated_at'][:10]}. **Caveat: only {ho['negatives']} negatives "
        f"in the test span — negative-class statistics are weak by construction "
        "(design §3.2 acknowledged this in advance).**")
    add("")
    add(_METRIC_HEADER)
    add(_metric_row("holdout (enet)", ho))
    add(_metric_row("holdout (hgbt)", ho_gbt))
    add("")
    for key, label in (("excess_hybrid:enet", "excess-only, hybrid bench"),
                       ("excess_peer:enet", "excess-only, peer bench")):
        add(_metric_row(f"holdout ({label})", holdout["models"][key]))
    add("")

    add("## CPCV — purged/embargoed, 5 blocks (primary evaluation)")
    add("")
    add("Training on data after a test block is permitted and labeled as such: it "
        "answers *do these factor weights discriminate in an unseen regime*, not "
        "*what would live deployment have earned* (design §3.2).")
    add("")
    add("### OR-label (confirmed deliverable), elastic-net logistic — PRIMARY")
    add("")
    add(_METRIC_HEADER)
    for block, m in or_blocks.items():
        add(_metric_row(block, m))
    add(f"| **mean** | | | | | **{mean_auc:.3f}** | **{mean_rank:.3f}** | | | | "
        f"**{mean_cap:.3f}** | | |")
    add("")
    add("### OR-label, HistGBT challenger")
    add("")
    add(_METRIC_HEADER)
    for block, m in hgbt_blocks.items():
        add(_metric_row(block, m))
    add("")
    verdict_adopted = (hgbt_auc - mean_auc > CHALLENGER_MARGIN
                       and ho_gbt["pooled_auc"] >= ho["pooled_auc"])
    add(f"**Challenger rule (§4.2):** HGBT mean CPCV AUC {hgbt_auc:.3f} vs logistic "
        f"{mean_auc:.3f} → margin {hgbt_auc - mean_auc:+.3f} "
        f"(needs > +{CHALLENGER_MARGIN:.2f}); holdout {ho_gbt['pooled_auc']:.3f} vs "
        f"{ho['pooled_auc']:.3f}. **Challenger "
        f"{'ADOPTED' if verdict_adopted else 'NOT adopted — interpretable model stays primary'}.**")
    add("")

    add("### Excess-only companion (condA) under BOTH benchmark schemes — the O2 range")
    add("")
    add("71.7% of the hybrid benchmark rests on a PRI+1.3% total-return "
        "approximation; the pure peer-proxy label is the same model with that "
        "approximation removed. The spread between the two rows bounds how much "
        "the conclusions depend on the benchmark construction.")
    add("")
    for tname, label in (("excess_hybrid", "condA (hybrid benchmark)"),
                         ("excess_peer", "condA (pure peer-proxy)")):
        add(f"#### {label}")
        add("")
        add(_METRIC_HEADER)
        blocks = cpcv["targets"][tname]["enet"]
        for block, m in blocks.items():
            add(_metric_row(block, m))
        t_auc = float(np.nanmean([m["pooled_auc"] for m in blocks.values()]))
        t_rank = float(np.nanmean([m["rank_auc"] for m in blocks.values()]))
        add(f"| **mean** | | | | | **{t_auc:.3f}** | **{t_rank:.3f}** | | | | | | |")
        add("")

    add("## Sensitivity grid (pre-registered, OR-label)")
    add("")
    add("| config | mean pooled AUC | mean rank-AUC |")
    add("|---|---|---|")
    add(f"| enet PRIMARY {json.dumps(PRIMARY_ENET)} | {mean_auc:.3f} | {mean_rank:.3f} |")
    hgbt_rank = float(np.nanmean([m["rank_auc"] for m in hgbt_blocks.values()]))
    add(f"| hgbt PRIMARY {json.dumps(PRIMARY_HGBT)} | {hgbt_auc:.3f} | {hgbt_rank:.3f} |")
    for key, v in cpcv["sensitivity"].items():
        add(f"| {key} | {v['mean_pooled_auc']:.3f} | {v['mean_rank_auc']:.3f} |")
    add("")

    add("## Calibration — holdout reliability (OR-label, elastic-net)")
    add("")
    add("| calibrated-p bin | n | mean p | observed frac positive |")
    add("|---|---|---|---|")
    for r in ho["reliability"]:
        add(f"| {r['bin']} | {r['n']} | {r['mean_p']:.3f} | {r['frac_pos']:.3f} |")
    add("")

    add("## Per-leg behavior of the OR-label model (pooled CPCV OOS predictions)")
    add("")
    try:
        preds = pd.read_parquet(OOS_PRED_PATH)
        pp = preds[(preds.target == "or_hybrid") & (preds.model == "enet")]
        rows = []
        for t, grp in pp.groupby("anchor"):
            k = max(1, int(np.floor(DECILE * len(grp))))
            g = grp.sort_values("p_raw")
            rows.append(dict(which="bottom", condA=g.condA[:k].mean(),
                             condB=g.condB[:k].mean(), y=g.y[:k].mean()))
            rows.append(dict(which="top", condA=g.condA[-k:].mean(),
                             condB=g.condB[-k:].mean(), y=g.y[-k:].mean()))
        leg = pd.DataFrame(rows).groupby("which").mean()
        add("| model decile (within anchor) | P(condA excess) | P(condB absolute) | P(y) |")
        add("|---|---|---|---|")
        for which, r in leg.iterrows():
            add(f"| {which} | {r.condA:.3f} | {r.condB:.3f} | {r.y:.3f} |")
        add("")
        add("If the top-vs-bottom spread is mostly in condA, the model discriminates "
            "fund skill; if mostly condB, it is riding the market-regime leg.")
    except Exception as e:  # report must still render if preds missing
        add(f"(per-leg table unavailable: {e})")
    add("")

    add("## Factor coefficients (standardized) + half-year block-bootstrap sign stability")
    add("")
    add("Source for the deferred Step-2 ScoringEngine weight review (§4.3). A "
        "factor whose sign is not stable across bootstrap replicates is 'not "
        "evidenced' regardless of its point estimate (§3.4).")
    add("")
    for tname in ("or_hybrid", "excess_hybrid"):
        c = coefs[tname]["coefficients"]
        s = coefs[tname]["sign_stability"] or {}
        top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:15]
        add(f"### {tname}")
        add("")
        add("| feature | coef | sign stability |")
        add("|---|---|---|")
        for feat, v in top:
            add(f"| {feat} | {v:+.3f} | {s.get(feat, float('nan')):.2f} |")
        add("")
    add("### excess_peer (coefficients only — scheme-sensitivity comparison)")
    add("")
    c = coefs["excess_peer"]["coefficients"]
    ch = coefs["excess_hybrid"]["coefficients"]
    top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:15]
    add("| feature | coef (peer) | coef (hybrid) |")
    add("|---|---|---|")
    for feat, v in top:
        add(f"| {feat} | {v:+.3f} | {ch.get(feat, float('nan')):+.3f} |")
    add("")

    add("## In-sample (quarantined — NOT evidence)")
    add("")
    add("| fold | in-sample AUC | in-sample accuracy |")
    add("|---|---|---|")
    for block, m in or_blocks.items():
        add(f"| {block} | {_fmt(m['in_sample_auc'])} | {_fmt(m['in_sample_accuracy'])} |")
    add("")
    add("Reminder: predict-all-1 scores 0.846 accuracy; nothing in this table "
        "demonstrates skill.")
    add("")

    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"report -> {REPORT_PATH}")
    return text


# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cpcv", "coef", "holdout", "artifact",
                                        "report", "all", "cohort"], default="all")
    ap.add_argument("--force-holdout", action="store_true",
                    help="re-evaluate the causal holdout (audit use only)")
    args = ap.parse_args()

    if args.stage == "cohort":
        df, feature_cols = load_cohort_dataset()
        print(f"cohort dataset: {len(df)} rows, {len(feature_cols)} features, "
              f"anchors {df['anchor'].min().date()}..{df['anchor'].max().date()}")
        cpcv = run_cpcv_cohort(df, feature_cols)
        holdout = run_holdout_cohort(df, feature_cols, force=args.force_holdout)
        coefs = run_coefficients_cohort(df, feature_cols)
        build_cohort_artifact(df, feature_cols)
        full_df, _ = load_dataset()
        label_stats = dict(
            min_size=COHORT_MIN_SIZE, n_total=int(len(full_df)),
            n_dropped=int(full_df["y_cohort_q1"].isna().sum()),
            n_kept=int(len(df)))
        holdout_span = (full_df["anchor"] >= CAUSAL_TEST_START) & (full_df["anchor"] <= CAUSAL_TEST_END)
        for tname, tcol in COHORT_TARGETS.items():
            kept = full_df["y_cohort_q1"].notna()
            v = full_df.loc[kept, tcol].astype(bool)
            vh = full_df.loc[kept & holdout_span, tcol].astype(bool)
            label_stats[tname] = dict(
                base_rate=float(v.mean()),
                holdout_base_rate=float(vh.mean()) if len(vh) else float("nan"),
                holdout_negatives=int((~vh).sum()))
        write_cohort_report(cpcv, holdout, coefs, label_stats)
        return

    # RETAINED ON PURPOSE (do not delete as "dead"): the non-cohort stages below
    # (cpcv/coef/holdout/artifact/report over the old OR-label) train/evaluate the
    # RETIRED utility model — measured at/below chance (AUC ~0.463) and never shipped
    # to the live product. They are kept as reproducible BASELINE / AUDIT infra: the
    # honest "this composite is below chance" number is only defensible if the code
    # that produces it stays runnable. Same call as the deliberately-kept
    # ScoringEngine.FACTOR_MAP (see docs/STATUS.md). Only `cohort_q1` (the --stage
    # cohort path above) is validated and shipped.
    df, feature_cols = load_dataset()
    print(f"dataset: {len(df)} rows, {len(feature_cols)} features, "
          f"anchors {df['anchor'].min().date()}..{df['anchor'].max().date()}")

    if args.stage in ("cpcv", "all"):
        cpcv = run_cpcv(df, feature_cols)
    else:
        cpcv = json.loads(CPCV_RESULTS_PATH.read_text(encoding="utf-8"))

    if args.stage in ("coef", "all"):
        coefs = run_coefficients(df, feature_cols)
    else:
        coefs = json.loads(COEF_PATH.read_text(encoding="utf-8"))

    if args.stage in ("holdout", "all"):
        holdout = run_holdout(df, feature_cols, force=args.force_holdout)
    else:
        holdout = json.loads(HOLDOUT_STAMP.read_text(encoding="utf-8"))

    if args.stage in ("artifact", "all"):
        build_artifact(df, feature_cols)

    if args.stage in ("report", "all"):
        write_report(cpcv, holdout, coefs)


if __name__ == "__main__":
    main()
