"""
mf_infer — pure-numpy live inference for the within-cohort model.

The Phase-B model artifact is deliberately plain JSON (design §6.2, "never
pickle"): coefficients + intercept + imputation medians + standardization stats
+ calibration map. That means a single live fund can be scored with numpy alone,
WITHOUT importing sklearn into the orchestrator. This module does exactly that
for the cohort artifact written by mf_model.build_cohort_artifact.

The cohort_q1 probability is the only validated predictive signal in the
pipeline (phase_b_v2 holdout AUC ~0.558, lift ~1.10x) — a weak signal INPUT, never an
oracle. `signal_context()` exposes its measured accuracy so callers surface it
honestly.

Self-test: `python mf_infer.py --selftest` re-fits the sklearn pipeline on the
same cohort data and asserts this numpy path reproduces its probabilities
bit-for-bit (atol 1e-9), proving the JSON round-trip is exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# The SHIPPED model, git-tracked at top-level model/ (2026-08-03) — deliberately
# separate from mf_cache/phase_b/, which holds the regenerable research outputs
# (features/labels parquets, CPCV results, reports) and stays gitignored.
#
# This one file is different in kind: it is the frozen, holdout-validated payload
# that actually scores funds, and mf_ledger stamps its `version` as the `model_id`
# on every prediction so an outcome realized in ~2029 can be tied back to the exact
# model that made it. If it can be evicted from an actions/cache and silently
# rebuilt, that model_id stops meaning anything — and rebuilding it nightly would
# emit predictions from a model whose holdout AUC (~0.558) was never measured.
# Frozen and versioned is the only honest option; regenerate deliberately with
# `python mf_model.py --stage cohort` and commit the result as a model change.
#
# SINGLE DEFINITION on purpose — mf_model.py imports this rather than keeping a
# second literal that could drift (same rule as mf_labels.MANIFEST_PATH).
MODEL_DIR = Path(__file__).parent / "model"
COHORT_ARTIFACT_PATH = MODEL_DIR / "model_artifact_cohort.json"
DEFAULT_TARGET = "cohort_q1"


def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    # clip the logit before exp so extreme predictors saturate to 0/1 instead of
    # raising an overflow warning (sklearn guards this internally; we do it here).
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _apply_calibration(p_raw: np.ndarray, cal: Dict[str, Any]) -> np.ndarray:
    """Apply the shipped calibration map via np.interp (which already clamps to
    the endpoint y-values, matching out_of_bounds='clip'), or the Platt sigmoid.

    `kind="isotonic"` is the map's provenance, not its exact shape: since
    phase_b_v3 the knots are PAVA blocks with thin ones merged (mf_model.
    _merge_thin_blocks), so the map is monotone piecewise-linear rather than
    sklearn's raw step function. That is deliberate — the step version collapsed
    23,646 scored rows onto 53 distinct probabilities, discarding the model's own
    ordering inside each step, and the merged map preserves it (23,058 distinct)
    while guaranteeing every knot is backed by >= MIN_CAL_BLOCK observations."""
    if cal["kind"] == "isotonic":
        return np.interp(p_raw, np.asarray(cal["x"], dtype=float),
                         np.asarray(cal["y"], dtype=float))
    z = _logit(p_raw)
    return _sigmoid(cal["coef"] * z + cal["intercept"])


class CohortInferencer:
    """Loads a cohort model artifact and scores feature dicts. Stateless after
    construction; safe to build once and reuse."""

    def __init__(self, artifact_path: Path = COHORT_ARTIFACT_PATH) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"cohort artifact not found: {artifact_path} — run "
                "`python mf_model.py --stage cohort` to build it.")
        self.artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.features: list[str] = self.artifact["features"]
        self.models: Dict[str, Any] = self.artifact["models"]

    @property
    def targets(self) -> list[str]:
        return list(self.models)

    def signal_context(self, target: str = DEFAULT_TARGET) -> Dict[str, Any]:
        """Measured edge of this target (holdout AUC + base rate) — surface the
        probability WITH this, never as a standalone verdict."""
        return dict(self.models[target].get("signal_context", {}))

    def _vectorize(self, features: Dict[str, float], m: Dict[str, Any]) -> np.ndarray:
        med, mean, std = m["imputation_medians"], m["standardize_mean"], m["standardize_std"]
        out = np.empty(len(self.features), dtype=float)
        for i, c in enumerate(self.features):
            v = features.get(c, None)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                v = med[c]                      # impute exactly as Preprocessor.fit did
            out[i] = (float(v) - mean[c]) / std[c]
        return out

    def predict(self, features: Dict[str, float], target: str = DEFAULT_TARGET) -> float:
        """Calibrated P(fund is in cohort's top quartile / half over 3y-forward)."""
        m = self.models[target]
        z = self._vectorize(features, m)
        coef = np.array([m["coefficients"][c] for c in self.features], dtype=float)
        p_raw = _sigmoid(m["intercept"] + float(coef @ z))
        return float(_apply_calibration(np.array([p_raw]), m["calibration"])[0])

    def predict_all(self, features: Dict[str, float]) -> Dict[str, float]:
        return {t: self.predict(features, t) for t in self.targets}


_DEFAULT: Optional[CohortInferencer] = None


def load_default() -> CohortInferencer:
    """Process-wide singleton so the orchestrator loads the artifact once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CohortInferencer()
    return _DEFAULT


# ------------------------------------------------------------------------------
def _selftest() -> None:
    """Re-fit the sklearn pipeline on the same cohort data and assert this
    numpy path reproduces its calibrated probabilities exactly."""
    import mf_model as M  # sklearn imported only inside the test, never at runtime

    inf = CohortInferencer()
    df, feature_cols = M.load_cohort_dataset()
    assert feature_cols == inf.features, "feature order drift between artifact and dataset"
    anchors = df["anchor"].to_numpy()
    all_mask = np.ones(len(df), dtype=bool)
    max_err = 0.0
    for tname, tcol in M.COHORT_TARGETS.items():
        y = df[tcol].to_numpy(dtype=bool)
        fit_mask, calib_mask = M.calib_split(anchors, all_mask)
        X = df[feature_cols]
        pre = M.Preprocessor().fit(X[fit_mask])
        model = M.fit_enet(pre.transform(X[fit_mask]), y[fit_mask], M.PRIMARY_ENET)
        cal = M.Calibrator().fit(
            model.predict_proba(pre.transform(X[calib_mask]))[:, 1], y[calib_mask])
        # sklearn probabilities on every cohort row
        p_sklearn = cal.predict(model.predict_proba(pre.transform(X))[:, 1])
        # numpy-path probabilities from the shipped JSON artifact
        rows = X.to_dict(orient="records")
        p_numpy = np.array([inf.predict(r, tname) for r in rows])
        err = float(np.max(np.abs(p_sklearn - p_numpy)))
        max_err = max(max_err, err)
        print(f"[selftest] {tname:16s} n={len(rows)} max|Δp|={err:.2e} "
              f"context={inf.signal_context(tname)}")
        assert err < 1e-9, f"{tname}: numpy inference diverges from sklearn (max {err})"

    # ---- the SHIPPED payload may never claim certainty ----------------------
    # Structural guard on the artifact itself, not on one fit: phase_b_v1 shipped
    # an isotonic map with a flat 0.0 region and gave 47 of 120 live funds a
    # literal `probability: 0.0`. Any future regeneration that reintroduces a 0
    # or 1 endpoint fails here rather than reaching the app.
    checked = 0
    worst_min, worst_support = 1.0, None
    for tname in inf.targets:
        cal = inf.artifact["models"][tname]["calibration"]
        if cal.get("kind") != "isotonic":
            continue          # Platt is asymptotic — it cannot emit 0 or 1
        ys = [float(v) for v in cal["y"]]
        assert min(ys) > 0.0 and max(ys) < 1.0, (
            f"FAIL: {tname} calibration claims certainty (min={min(ys)}, max={max(ys)}) — "
            "a 0 or 1 probability asserts more than this model can support")
        # Removing the 0 is not enough — the number that replaces it has to be
        # backed by real observations, which is what `support` records.
        support = cal.get("support")
        assert support, f"FAIL: {tname} calibration ships no per-knot support counts"
        assert min(support) >= M.MIN_CAL_BLOCK, (
            f"FAIL: {tname} has a calibration block of {min(support)} observations "
            f"(< MIN_CAL_BLOCK={M.MIN_CAL_BLOCK}) — its probability claims more "
            "than its own support allows")
        checked += 1
        if min(ys) < worst_min:
            worst_min, worst_support = min(ys), min(support)
    # Every isotonic target must have been reachable; a Platt-only artifact is
    # legitimate, so report that rather than indexing a 'y' key it does not have.
    if checked:
        print(f"[selftest] shipped calibration bounded away from 0 and 1 "
              f"(min={worst_min:.6f}, every block >= {M.MIN_CAL_BLOCK} obs, "
              f"thinnest={worst_support}) — PASS")
    else:
        print("[selftest] shipped calibration is Platt on every target — "
              "asymptotic, cannot emit 0 or 1 — PASS")
    print(f"[selftest] PASS — numpy inference reproduces sklearn (max|Δp|={max_err:.2e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="verify numpy inference == sklearn pipeline on cohort data")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        inf = CohortInferencer()
        print(f"loaded cohort artifact: targets={inf.targets}")
        for t in inf.targets:
            print(f"  {t}: signal_context={inf.signal_context(t)}")
