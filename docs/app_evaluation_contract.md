# App contract — the `evaluation{}` model-health panel

**Producer:** `mf_eval.py` (grading) → embedded by `mf_artifact.py` into the batch
artifact top-level as `evaluation{}`.
**Consumer:** Hisaab Kitaab (Flutter). This doc is the batch-producer side of the
contract; mirror the UI decisions into that repo's `DECISIONS.md`.

## Why it exists
The artifact already carries raw monitoring signals in `monitoring{}` (PSI,
rank_stability, realized_ic). `evaluation{}` is those signals **graded against
documented thresholds** into a GREEN / AMBER / RED model-health verdict the app can
render directly — the answer to *"when should I distrust this model?"* — without the
app re-deriving any thresholds.

## The load-bearing honesty rule the UI MUST honour
**Model health is NOT model accuracy.** Early in this model's life the only
measurable things are leading indicators (drift, stability, coverage). Whether the
predictions are *right* (`outcome_skill`) is unmeasurable until ~3y of logged
predictions mature (~2029 earliest). So:

- The app **must show `evaluation.disclaimer` verbatim** (or an equivalent) wherever
  the panel appears. Never label this panel "accuracy", "win rate", or "returns".
- `PENDING` (⏳) means **not-yet-measurable, not bad** — render it distinctly from a
  RED failure. A `PENDING` outcome never colours the overall status.
- The cohort signal it monitors is itself a **weak** ranking aid (holdout AUC ~0.578).
  Do not upsell the panel into a confidence score for a BUY.

## Schema (`artifact.evaluation`)

```jsonc
{
  "status":   "GREEN|AMBER|RED|PENDING",   // overall roll-up over ACTIONABLE metrics only
  "headline": "WATCH — input_drift drifting; investigate, no retrain required yet.",
  "model_id": "cohort_q1_...",             // matches artifact.model_id
  "disclaimer": "This panel reflects data-drift and pipeline health — NOT fund-outcome accuracy. ...",
  "note": "...",                           // ledger's own honest note
  "metrics": [                             // render as a list of dots + labels
    {
      "name":      "input_drift",          // stable id (see table below)
      "status":    "AMBER",                // GREEN|AMBER|RED|PENDING|UNAVAILABLE
      "label":     "Feature drift vs the training era — investigate, not an auto-retrain trigger.",
      "value":     0.34,                   // may be null
      "threshold": "PSI: OK<0.1, shift>0.25",
      "detail":    "worst: mom_12m_ex1m(0.34), cagr_1y(0.29)"
    }
    // ... prediction_stability, pipeline_coverage, outcome_skill
  ],
  "outcome": {                             // the real scorecard — PENDING for years
    "status": "PENDING_MATURITY",          // or INSUFFICIENT_MATURED / OK
    "realized_ic": null, "lift": null, "precision_top": null, "base_rate": null,
    "calibration_gap": null, "n_matured": 0, "n_realized": 0,
    "effective_n": 0, "earliest_maturity": "2029-03-31"
  }
}
```

## The four metrics

| `name` | what it measures | GREEN | AMBER | RED | when PENDING/UNAVAILABLE |
|---|---|---|---|---|---|
| `input_drift` | live feature distribution vs training (PSI) | matches | **any** shift (investigate) | *never* — drift is not a decay verdict | reference missing/stale, or too few funds |
| `prediction_stability` | run-to-run Spearman of predicted ranks | ≥0.90 | 0.70–0.90 | <0.70 (erratic) | first run / <20 common funds |
| `pipeline_coverage` | share of funds that evaluated cleanly | err <5% | 5–20% | ≥20% errors | no funds attempted |
| `outcome_skill` | realized lift over base rate on **matured** predictions | ≥1.20x | 1.0–1.20x | **<1.0x (worse than chance)** | until ~50 predictions mature (~2029) |

Overall `status` = worst of the **actionable** (GREEN/AMBER/RED) metrics. PENDING and
UNAVAILABLE metrics are shown but never worsen the headline. Thresholds are the tunable
constants at the top of `mf_eval.py` (`STABILITY_*`, `ERROR_RATE_*`, `LIFT_*`).

## Suggested UI
A compact "Model health" card: one status dot + `headline`, an expandable list of the
four metrics (dot + `label`, with `value`/`detail` on tap), and the `disclaimer` always
visible in the expanded state. Show `outcome.earliest_maturity` next to the PENDING
outcome row so the user sees *when* an accuracy read becomes possible. This card is
**global** (about the model), distinct from a fund's own verdict/colours.

## Refreshing it
`evaluation{}` is recomputed every batch run (drift/stability/coverage are live).
`outcome_skill` only advances when the offline realization job has run
(`python mf_ledger.py --realize`, monthly) and enough predictions have matured — so
early artifacts will show it PENDING indefinitely, which is correct.
