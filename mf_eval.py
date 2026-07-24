"""
====================================================================================
mf_eval.py — MODEL-HEALTH EVALUATION FRAMEWORK (thresholded, app-ready)
====================================================================================
WHY THIS EXISTS
---------------
`mf_ledger.py` already COMPUTES the honest monitoring signals (PSI drift,
rank_stability, realized_ic) and `mf_artifact.py` already carries them to the
app inside `monitoring{}`. What was missing is the layer that answers the actual
question — **"when do I need to tweak the model?"** — by turning those raw
signals into a thresholded GREEN / AMBER / RED verdict per metric, an overall
model-health status, and an app-renderable panel. That decision layer is this
module. It adds NO new model and touches NO scoring — it reads what the ledger
already produced and grades it against DOCUMENTED thresholds (the constants
below), so "tweak it" becomes a rule you can read, not a hunch.

THE HONESTY LINE (read CLAUDE.md first)
---------------------------------------
- **Model health is NOT model accuracy.** Early in this model's life the ONLY
  measurable signals are leading indicators — input drift, run-to-run stability,
  pipeline coverage. Whether the model's *predictions are right* (outcome skill)
  cannot be known until ~3y of logged predictions mature (~2029 earliest; see
  mf_ledger). So `outcome_skill` reports PENDING, never a fabricated number, and
  a PENDING outcome NEVER drags the overall status to RED — "not yet measurable"
  is honestly different from "bad".
- **A drift signal is an INVESTIGATE flag, not a decay verdict.** PSI SIGNIFICANT
  caps at AMBER, never RED — the model was designed expecting market-regime drift
  (2025-26 vs the pooled 2013-2023 training era); it means "look", not "retrain"
  (mf_ledger design note). Only a realized-outcome collapse on MATURED predictions
  is a true RED "tweak it now".
- Every threshold here is a documented, tunable constant — no magic numbers buried
  in logic.

TWO SEPARATE SCORECARDS
-----------------------
This module grades the `cohort_q1` STATISTICAL MODEL. The tri-colour BUY/HOLD/SELL
verdict is a DETERMINISTIC RULE, not a learned model — its "evaluation" is verdict
distribution / stability, a different concern, deliberately not conflated here (that
would relaunder the composite the project retired).
====================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import mf_ledger

LOGGER = logging.getLogger("MFEval")

# --- Status vocabulary --------------------------------------------------------
GREEN, AMBER, RED, PENDING, UNAVAILABLE = "GREEN", "AMBER", "RED", "PENDING", "UNAVAILABLE"
# Only these three participate in the overall roll-up; PENDING/UNAVAILABLE are
# reported honestly but never colour the headline (see honesty line above).
_ACTIONABLE = {GREEN, AMBER, RED}
_WORSE = {GREEN: 0, AMBER: 1, RED: 2}

# --- Documented, tunable thresholds ------------------------------------------
# rank_stability (Spearman of this run's predicted probs vs the previous run's,
# on the common funds): high is good. Below RED the pipeline is behaving
# erratically run-to-run — a real "something changed under me" signal.
STABILITY_GREEN, STABILITY_RED = 0.90, 0.70

# pipeline error rate (funds that failed to evaluate / total attempted): low is good.
ERROR_RATE_GREEN, ERROR_RATE_RED = 0.05, 0.20

# realized outcome LIFT on matured predictions = precision in the model's top
# tier / the cohort base rate. >1 beats chance; the training holdout showed
# ~1.76x. Below 1.0 the model is worse than a coin at its own game -> RED "tweak".
LIFT_GREEN, LIFT_RED = 1.20, 1.00
# top tier = predicted cohort_percentile at/above this (the cohort_q1 target is
# top-quartile membership, so the natural decision cut is the 75th pct).
TOP_TIER_PCTL = 0.75


@dataclass
class MetricHealth:
    """One graded signal. `status` in {GREEN, AMBER, RED, PENDING, UNAVAILABLE};
    `value`/`threshold`/`detail` are the evidence the app shows under the dot."""
    name: str
    status: str
    label: str                       # short human sentence, app-renderable as-is
    value: Optional[float] = None
    threshold: Optional[str] = None
    detail: str = ""


def _worst(statuses: List[str]) -> str:
    actionable = [s for s in statuses if s in _ACTIONABLE]
    if not actionable:
        return PENDING
    return max(actionable, key=lambda s: _WORSE[s])


# ==============================================================================
# Per-signal grading (pure: dict in -> MetricHealth out)
# ==============================================================================
def grade_input_drift(psi: Dict[str, Any]) -> MetricHealth:
    st = psi.get("status")
    worst = psi.get("worst") or []
    worst_str = ", ".join(f"{w['feature']}({w['psi']})" for w in worst[:3])
    thr = f"PSI: OK<{mf_ledger.PSI_MODERATE}, shift>{mf_ledger.PSI_SIGNIFICANT}"
    if st == "OK":
        return MetricHealth("input_drift", GREEN,
                            "Live feature distribution matches the training population.",
                            psi.get("psi_max"), thr)
    if st in ("MODERATE_SHIFT", "SIGNIFICANT_SHIFT"):
        # Capped at AMBER by design — drift is INVESTIGATE, never an auto-retrain/RED.
        return MetricHealth("input_drift", AMBER,
                            "Feature drift vs the training era — investigate, not an "
                            "auto-retrain trigger.",
                            psi.get("psi_max"), thr,
                            detail=f"worst: {worst_str}" if worst_str else st)
    if st in ("REFERENCE_MISSING", "REFERENCE_STALE"):
        return MetricHealth("input_drift", UNAVAILABLE,
                            "Drift reference missing or stale — rebuild psi_reference "
                            "(python mf_ledger.py --build-reference).",
                            detail=st)
    return MetricHealth("input_drift", UNAVAILABLE,
                        "Not enough live funds to measure drift.", detail=str(st))


def grade_stability(rs: Dict[str, Any]) -> MetricHealth:
    st, sp = rs.get("status"), rs.get("spearman")
    thr = f"Spearman: green>={STABILITY_GREEN}, red<{STABILITY_RED}"
    if st == "OK" and sp is not None:
        status = GREEN if sp >= STABILITY_GREEN else (RED if sp < STABILITY_RED else AMBER)
        return MetricHealth("prediction_stability", status,
                            "Predicted ranks are consistent run-to-run."
                            if status == GREEN else
                            "Predicted ranks are shifting between runs — inspect inputs.",
                            sp, thr, detail=f"n_common={rs.get('n_common')}")
    if st == "FIRST_RUN":
        return MetricHealth("prediction_stability", PENDING,
                            "First logged run — stability needs a prior run to compare.",
                            threshold=thr)
    return MetricHealth("prediction_stability", PENDING,
                        "Too few funds overlap the previous run to grade stability.",
                        threshold=thr, detail=f"n_common={rs.get('n_common')}")


def grade_pipeline(coverage: Dict[str, Any]) -> MetricHealth:
    n_total = coverage.get("n_total", 0) or 0
    n_err = coverage.get("n_errors", 0) or 0
    thr = f"error rate: green<{ERROR_RATE_GREEN:.0%}, red>={ERROR_RATE_RED:.0%}"
    if n_total == 0:
        return MetricHealth("pipeline_coverage", UNAVAILABLE, "No funds attempted.", threshold=thr)
    rate = n_err / n_total
    status = GREEN if rate < ERROR_RATE_GREEN else (RED if rate >= ERROR_RATE_RED else AMBER)
    return MetricHealth("pipeline_coverage", status,
                        f"{n_total - n_err}/{n_total} funds evaluated cleanly.",
                        round(rate, 4), thr,
                        detail=f"thin_cohort={coverage.get('n_thin_cohort', 0)}, "
                               f"stale_nav={coverage.get('n_stale_nav', 0)}")


def grade_outcome(outcome: Dict[str, Any]) -> MetricHealth:
    """The REAL 'tweak it' signal — realized skill on MATURED predictions. PENDING
    for years by construction; when it lands, LIFT<1.0 is the honest RED."""
    st = outcome.get("status")
    thr = f"lift: green>={LIFT_GREEN}, red<{LIFT_RED} (base rate = 1x)"
    if st in ("PENDING_MATURITY", "INSUFFICIENT_MATURED"):
        em = outcome.get("earliest_maturity")
        return MetricHealth("outcome_skill", PENDING,
                            "Outcome accuracy is not yet measurable — predictions need "
                            f"~3y to mature{f' (earliest ~{em})' if em else ''}.",
                            threshold=thr,
                            detail=f"n_realized={outcome.get('n_realized', 0)} / "
                                   f"need {mf_ledger.MIN_MATURED_FOR_IC}")
    lift = outcome.get("lift")
    if st == "OK" and lift is not None:
        status = GREEN if lift >= LIFT_GREEN else (RED if lift < LIFT_RED else AMBER)
        return MetricHealth("outcome_skill", status,
                            f"Realized lift {lift:.2f}x over the cohort base rate on "
                            f"{outcome.get('n_realized')} matured predictions.",
                            lift, thr,
                            detail=f"realized_ic={outcome.get('realized_ic')}, "
                                   f"effective_n={outcome.get('effective_n')}")
    return MetricHealth("outcome_skill", UNAVAILABLE,
                        "Outcome metrics unavailable.", threshold=thr, detail=str(st))


# ==============================================================================
# Outcome metrics on the MATURED realizations (extends ledger.realized_summary
# with lift/precision/calibration — all honestly PENDING until maturity)
# ==============================================================================
def outcome_metrics(realizations_path: Path = mf_ledger.REALIZATIONS_PATH,
                    predictions_path: Path = mf_ledger.PREDICTIONS_PATH) -> Dict[str, Any]:
    base = mf_ledger.realized_summary(realizations_path, predictions_path)
    out: Dict[str, Any] = dict(
        status=base["status"], realized_ic=base["value"],
        n_matured=base["n_matured"], n_realized=base["n_realized"],
        effective_n=base["effective_n"], earliest_maturity=base["earliest_maturity"],
        lift=None, precision_top=None, base_rate=None, calibration_gap=None)
    if base["status"] != "OK":
        return out  # PENDING/INSUFFICIENT — no fabricated lift

    rows = [r for r in mf_ledger.read_jsonl(realizations_path) if r["status"] == "REALIZED"]
    y = np.array([float(r["y_realized"]) for r in rows])
    pctl = np.array([float(r["cohort_percentile"]) for r in rows])
    prob = np.array([float(r["probability"]) for r in rows])
    base_rate = float(y.mean())
    top = pctl >= TOP_TIER_PCTL
    precision_top = float(y[top].mean()) if top.any() else None
    out.update(
        base_rate=round(base_rate, 4),
        precision_top=None if precision_top is None else round(precision_top, 4),
        lift=(None if not precision_top or base_rate == 0
              else round(precision_top / base_rate, 3)),
        # reliability: does the mean predicted probability match the realized rate?
        calibration_gap=round(float(abs(prob.mean() - base_rate)), 4))
    return out


# ==============================================================================
# The app-facing report — the single object mf_artifact.py embeds as evaluation{}
# ==============================================================================
@dataclass
class ModelHealthReport:
    status: str                              # overall roll-up over ACTIONABLE metrics
    headline: str
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = ""
    model_id: Optional[str] = None
    note: str = ""


_DISCLAIMER = (
    "This panel reflects data-drift and pipeline health — NOT fund-outcome accuracy. "
    "The cohort signal is a weak, within-cohort ranking aid (training holdout AUC "
    "~0.578, lift ~1.76x); its real-world accuracy cannot be measured until ~3 years "
    "of logged predictions mature. 'Pending' means not-yet-measurable, not bad.")


def build_report(monitoring: Dict[str, Any], coverage: Dict[str, Any],
                 outcome: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, Any]:
    """Pure: raw monitoring{} + coverage{} + outcome{} -> the app-renderable
    evaluation{} block. No orchestrator / NAV / sklearn dependency, so it is
    unit-testable on synthetic dicts."""
    metrics = [
        grade_input_drift(monitoring.get("psi", {})),
        grade_stability(monitoring.get("rank_stability", {})),
        grade_pipeline(coverage),
        grade_outcome(outcome),
    ]
    overall = _worst([m.status for m in metrics])
    reds = [m.name for m in metrics if m.status == RED]
    ambers = [m.name for m in metrics if m.status == AMBER]
    if overall == RED:
        headline = f"ACTION NEEDED — {', '.join(reds)} failing; review the model."
    elif overall == AMBER:
        headline = f"WATCH — {', '.join(ambers)} drifting; investigate, no retrain required yet."
    elif overall == GREEN:
        headline = "HEALTHY — leading indicators nominal (outcome accuracy still maturing)."
    else:
        headline = "MONITORING ONLY — nothing actionable is measurable yet."
    report = ModelHealthReport(
        status=overall, headline=headline,
        metrics=[asdict(m) for m in metrics], outcome=outcome,
        disclaimer=_DISCLAIMER, model_id=model_id,
        note=monitoring.get("note", ""))
    return asdict(report)


def build_report_from_ledger(monitoring: Dict[str, Any], coverage: Dict[str, Any],
                             model_id: Optional[str] = None,
                             realizations_path: Path = mf_ledger.REALIZATIONS_PATH,
                             predictions_path: Path = mf_ledger.PREDICTIONS_PATH) -> Dict[str, Any]:
    """Convenience wrapper mf_artifact.py calls: pulls the matured-outcome metrics
    from the ledger, then grades everything."""
    outcome = outcome_metrics(realizations_path, predictions_path)
    return build_report(monitoring, coverage, outcome, model_id=model_id)


def render(report: Dict[str, Any]) -> str:
    dot = {GREEN: "🟢", AMBER: "🟡", RED: "🔴", PENDING: "⏳", UNAVAILABLE: "⚪"}
    L = ["=" * 74,
         f"  MODEL HEALTH: {dot[report['status']]} {report['status']}  "
         f"(model {report.get('model_id') or 'n/a'})",
         f"  {report['headline']}",
         "=" * 74]
    for m in report["metrics"]:
        v = "" if m["value"] is None else f"  [{m['value']}]"
        L.append(f"  {dot[m['status']]} {m['name']:<20}{v}")
        L.append(f"       {m['label']}")
        if m["detail"]:
            L.append(f"       ({m['detail']})")
    L.append("-" * 74)
    L.append(f"  {report['disclaimer']}")
    return "\n".join(L)


# ==============================================================================
# SELFTEST — pure, data-free (synthetic monitoring/coverage/outcome dicts)
# ==============================================================================
def _selftest() -> None:
    # 1) Healthy leading indicators, outcome PENDING -> overall GREEN, NOT dragged
    #    down by the pending outcome.
    mon = dict(psi=dict(status="OK", psi_max=0.03, worst=[]),
               rank_stability=dict(status="OK", spearman=0.96, n_common=120),
               note="n")
    cov = dict(n_total=136, n_errors=2, n_thin_cohort=5, n_stale_nav=0)
    out = dict(status="PENDING_MATURITY", realized_ic=None, n_matured=0, n_realized=0,
               effective_n=0, earliest_maturity="2029-03-31", lift=None)
    rep = build_report(mon, cov, out, model_id="cohort_v1")
    assert rep["status"] == GREEN, rep["status"]
    names = {m["name"]: m["status"] for m in rep["metrics"]}
    assert names["outcome_skill"] == PENDING and names["input_drift"] == GREEN
    assert "maturing" in rep["headline"].lower()
    render(rep)  # must not raise
    print(f"[selftest] healthy + pending outcome -> {rep['status']} (pending never reddens) — PASS")

    # 2) Feature drift SIGNIFICANT caps at AMBER (investigate, never RED)
    mon2 = dict(mon, psi=dict(status="SIGNIFICANT_SHIFT", psi_max=0.41,
                              worst=[dict(feature="vol_1y", psi=0.41)]))
    rep2 = build_report(mon2, cov, out)
    drift = next(m for m in rep2["metrics"] if m["name"] == "input_drift")
    assert drift["status"] == AMBER and rep2["status"] == AMBER, (drift["status"], rep2["status"])
    print("[selftest] PSI SIGNIFICANT_SHIFT -> AMBER investigate (never RED) — PASS")

    # 3) Erratic run-to-run predictions + high error rate -> RED
    mon3 = dict(mon, rank_stability=dict(status="OK", spearman=0.55, n_common=90))
    cov3 = dict(n_total=100, n_errors=30, n_thin_cohort=0, n_stale_nav=0)
    rep3 = build_report(mon3, cov3, out)
    assert rep3["status"] == RED
    assert {m["name"] for m in rep3["metrics"] if m["status"] == RED} == \
        {"prediction_stability", "pipeline_coverage"}
    print("[selftest] unstable ranks + high error rate -> RED action-needed — PASS")

    # 4) A MATURED outcome that collapsed BELOW chance -> RED, and lift math is right.
    #    30 top-tier (pctl 0.9) funds mostly miss; 30 bottom (pctl 0.3) mostly hit ->
    #    the model ranked backwards. probability tracks pctl so the IC corr is defined.
    real = ([dict(status="REALIZED", y_realized=(i % 5 == 0), cohort_percentile=0.9,
                  probability=0.9) for i in range(30)]          # top-tier precision = 0.2
            + [dict(status="REALIZED", y_realized=True, cohort_percentile=0.3,
                    probability=0.3) for i in range(30)])       # base_rate = (6+30)/60 = 0.6
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    tmp.write_text("\n".join(json.dumps(r) for r in real), encoding="utf-8")
    # a matching predictions file so realized_summary counts >= MIN_MATURED_FOR_IC -> OK.
    preds = Path(tempfile.mktemp(suffix=".jsonl"))
    preds.write_text("\n".join(json.dumps(dict(amfi_code=f"F{i}", target="cohort_q1",
                     anchor="2020-01-31", probability=0.5, run_id="r")) for i in range(60)),
                     encoding="utf-8")
    om = outcome_metrics(realizations_path=tmp, predictions_path=preds)
    assert om["status"] == "OK", om["status"]
    assert abs(om["base_rate"] - 0.6) < 1e-9, om
    assert om["lift"] == round(0.2 / 0.6, 3) and om["lift"] < LIFT_RED, om
    rep4 = build_report(mon, cov, om)
    assert next(m for m in rep4["metrics"] if m["name"] == "outcome_skill")["status"] == RED
    assert rep4["status"] == RED
    print(f"[selftest] matured outcome below chance (lift={om['lift']}) -> outcome RED — PASS")

    # 5) UNAVAILABLE reference doesn't crash and doesn't redden
    mon5 = dict(mon, psi=dict(status="REFERENCE_STALE"))
    rep5 = build_report(mon5, cov, out)
    assert next(m for m in rep5["metrics"] if m["name"] == "input_drift")["status"] == UNAVAILABLE
    assert rep5["status"] == GREEN
    print("[selftest] stale drift reference -> UNAVAILABLE, overall stays GREEN — PASS")

    print("[selftest] PASS — mf_eval grades leading indicators honestly, keeps 'pending' "
          "separate from 'bad', caps drift at investigate, and only a matured-outcome "
          "collapse turns the model RED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Model-health evaluation over the ledger + artifact")
    ap.add_argument("--selftest", action="store_true", help="pure, data-free selftest")
    ap.add_argument("--report", metavar="ARTIFACT.json[.gz]",
                    help="render the model-health panel from an emitted artifact's "
                         "monitoring{}+coverage{}, plus fresh ledger outcome metrics")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.selftest:
        _selftest()
    elif args.report:
        import gzip
        p = Path(args.report)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:   # type: ignore[operator]
            art = json.load(fh)
        rep = art.get("evaluation") or build_report_from_ledger(
            art.get("monitoring", {}), art.get("coverage", {}), art.get("model_id"))
        print(render(rep))
    else:
        ap.print_help()
