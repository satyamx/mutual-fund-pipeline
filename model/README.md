# model/ — the shipped model, versioned not generated

Two files, both git-tracked, both deliberately outside `mf_cache/phase_b/` (which
holds the regenerable research outputs and stays gitignored).

- **`model_artifact_cohort.json`** (17 KB) — the full inference payload for the
  `cohort_q1` elastic-net: coefficients, feature order, scaler statistics, imputation
  medians, calibration, and a `version` string. Since `phase_b_v3` the calibration map's knots are PAVA blocks with thin ones
merged, each carrying a `support` count of at least `mf_model.MIN_CAL_BLOCK`
observations — so no emitted probability claims more than its own evidence
allows, and `mf_infer --selftest` fails the artifact if one does.
Read by `mf_infer.CohortInferencer`
  with numpy alone, no sklearn. `mf_infer.COHORT_ARTIFACT_PATH` is the **single**
  definition of this path; `mf_model.py` imports it rather than declaring a second
  literal, because a drift there would mean the trainer writes a model the inference
  side never reads.
- **`psi_reference.json`** (13.5 KB) — the empirical training-time decile
  distribution the live PSI drift check compares against. Only meaningful paired with
  the exact model it was built from, which is why it lives here and not in the cache.

## Why versioned rather than regenerated

This is the one Phase-B output that is not research. It is the frozen,
holdout-validated payload that actually scores funds, and `mf_ledger` stamps its
`version` as the `model_id` on **every** prediction so an outcome realized in ~2029
can be tied back to the exact model that made the call.

Two failure modes that forces closed:

1. **Eviction.** It previously sat in `mf_cache/`, gitignored and restored from an
   `actions/cache` GitHub evicts on idle or size pressure. Every cold CI runner had
   no model, so the nightly emitted `cohort_status: null` for all 136 funds and
   appended zero ledger rows — a signal-less artifact that still looked like a
   successful run.
2. **Silent retraining.** Rebuilding it on a schedule would emit predictions from a
   model whose holdout AUC was never measured, while the ledger kept stamping a
   `model_id` that no longer identified anything. The reported ~0.558 belongs to one
   specific fit on one specific training window.

**Do not retrain on a schedule.** Regenerate deliberately:

```bash
./.venv/Scripts/python.exe mf_model.py --stage cohort
```

then rebuild the drift reference against the new model and commit both together as a
single, reviewable model change:

```bash
./.venv/Scripts/python.exe mf_ledger.py --build-reference
```

A new artifact means a new `version`, which means predictions before and after are
**not** comparable as one series — `mf_eval` reads `model_id` precisely so a model
change is visible rather than silently averaged into the outcome history.
