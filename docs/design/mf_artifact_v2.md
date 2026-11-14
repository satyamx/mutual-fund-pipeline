# `mf_artifact_v2` — the horizon menu in the batch artifact

Design, 2026-11-13. Companion to `docs/horizons.md` (the menu, recipe and bar). This file fixes the shape
of the artifact that carries the horizons to Hisaab Kitaab, the transition from v1, and what the app side
has to build. The app work is a separate session; the last section is its brief.

## 1. The constraint that shapes everything

Hisaab Kitaab's parser refuses any document whose `artifact_version` is not exactly `mf_artifact_v1`
(`lib/services/mf_artifact_parser.dart:19`), on purpose: a half-understood artifact is worse than none.
So an installed app cannot be handed v2, and v1 cannot quietly grow.

**Decision: v1 keeps publishing, unchanged, and v2 is a second file beside it.** v2 is a strict superset
of v1 — every v1 key is present with the same meaning — so one run builds the v1 document exactly as
today and derives v2 from it by adding keys. Nothing is renamed, moved or re-typed.

## 2. Top level

Everything in v1 (`generated_at`, `pipeline_sha`, `model_id`, `default_profile`, `coverage`, `monitoring`,
`evaluation`, `funds`) stays. `model_id` and `monitoring` remain the 3y model's, as in v1. Added:

```json
{
  "artifact_version": "mf_artifact_v2",
  "default_horizon": "3y",
  "horizons": [
    {
      "key": "1y",
      "months": 12,
      "target": "cohort_q1_1y",
      "model_id": "phase_b_1y_v1_cohort",
      "standing": "WEAK",
      "marginal": false,
      "signal_in_verdict": false,
      "holdout": {
        "auc": 0.0, "auc_lower_bound": 0.0, "rank_auc": 0.0, "top_decile_lift": 0.0,
        "n": 0, "effective_n": 0, "divisor": 23, "strict_divisor": 29,
        "cpcv_blocks_above_half": "4/6",
        "test_span": "2024-11-30..2025-10-31", "evaluated_at": "2026-11-13"
      },
      "ledger": {
        "rows_total": 0, "last_anchor": null, "first_maturity": null, "n_realized": 0,
        "month_end": {
          "anchor": "2026-10-31", "appended": 0, "skipped_funds": 0,
          "skipped_month": 0, "already_logged": false, "pending": false
        }
      }
    }
  ]
}
```

- `horizons` is ordered shortest first and is the **menu**: the app builds its picker from this list, not
  from a hardcoded one.
- `standing` is `PASS`, `WEAK` or `FAIL`, read from the horizon's git-tracked stamp
  (`model/holdout_stamp_<key>.json`). A horizon with no stamp is not listed at all.
- `signal_in_verdict` is true for `PASS`; a marginal PASS only when the owner has admitted that model id
  (`mf_horizon.MARGINAL_ADMITTED`). It is the single switch the rest of the document obeys.
- `holdout` is the one evaluation, verbatim from the stamp. The zeros above are placeholders in this
  design; a real artifact never carries a made-up number.
- `ledger` comes from the per-target ledger readers (`rows_total`, `last_anchor`; `first_maturity` and
  `n_realized` for this model id), plus `month_end` = this run's `mf_ledger.log_month_end` result for the
  target, 3y included (`null` when the run could not log). `pending: true` means the
  run fell in the first 3 days after the month-end, while NAVs settle: nothing written, not a skip.

## 3. Per fund

Every v1 fund key stays (`facts`, `signal_a`, `alerts_b`, …). Added: a `horizons` object with one entry
per menu key.

```json
"horizons": {
  "1m": {
    "facts": {
      "trailing_return": 0.012,
      "return_basis": "period",
      "consistency": 0.61,
      "worst_5pct_return": -0.071,
      "n_windows": 2410,
      "history_months": 131
    },
    "signal": {
      "status": "WITHHELD",
      "top_decile": null, "percentile": null, "probability": null,
      "cohort_n": null, "n_imputed": null, "imputed_fraction": null
    },
    "verdict": {
      "verdict": "HOLD", "color": "blue", "caveat": "…",
      "metric_colors": {}, "signal_used": false
    }
  }
}
```

**Facts** — measured from the fund's own NAV, for every horizon, whatever its standing:

| field | meaning | null when |
|---|---|---|
| `trailing_return` | return over the last `h`, as of the fund's last NAV | no NAV within the horizon's staleness tolerance of `t − h` |
| `return_basis` | `"period"` for `h < 12` months, `"annualized"` from 12 months up | never |
| `consistency` | share of rolling-`h` windows with a return ≥ 0 | under 36 months of NAV (`h < 36`), or no complete window |
| `worst_5pct_return` | 5th percentile of rolling-`h` returns, same basis | same as `consistency` |
| `n_windows` | rolling windows behind the two numbers above | never (0 when none) |
| `history_months` | NAV span | never |

A one-month return is **never annualised**: compounding it twelve times manufactures a number nobody
earned.

**Signal** — the horizon's cohort model output:

- When the horizon's `signal_in_verdict` is `false`, `status` is `"WITHHELD"` and every other field is
  null. The predictions still go to the ledger; they do not go to the artifact. "Never displayed" is then
  a property of the file, not a promise the app has to keep.
- When it is `true`, `status` is the usual vocabulary (`OK`, `INSUFFICIENT_HISTORY`, `THIN_COHORT`,
  `STALE_NAV`, …), or `NOT_SCORED` when the fund was not scored at all (never null); the numbers are
  present only for `OK`. `top_decile` is the one field the verdict
  reads.

**Verdict** — the horizon's rule from `docs/horizons.md` §6. `signal_used` says whether a sixth colour took
part. `metric_colors` carries the colours that were counted, so the verdict stays explainable.

**What stays null or absent, by design:** probabilities for any non-PASS horizon; any fact without enough
history; TER, AUM, holdings and manager fields (unchanged from v1 — no free source).

**Two things kept for compatibility, flagged rather than hidden:**

- `signal_a` still carries the 3y `cohort_q1_prob` exactly as v1 does, although 3y stands at WEAK. A v2
  reader takes signals from `funds[].horizons[...]` only and ignores the `signal_a` cohort fields.
- `funds[].horizons["3y"].verdict.verdict` equals `signal_a.verdict` for every fund. A selftest pins it.

## 4. Transition

1. The nightly emits both files from one run. v2 is best-effort: a v2 failure is logged and never blocks
   v1, its ledger commit or its Release assets (the v2 asset is then not replaced that night). Release assets: `mf_artifact_latest.json.gz` and
   `latest.json` stay v1; `mf_artifact_v2_latest.json.gz` is added.
2. Hisaab Kitaab ships a parser that accepts v2 (section 6).
3. Retiring v1 is the owner's call, and not before a build that reads v2 is the one installed.

## 5. Selftests that pin the shape (each seen failing first)

In `mf_artifact.py --selftest`, on a fixture:

1. **Superset.** v2 with `horizons`, `default_horizon` and each fund's `horizons` removed, and the version
   string swapped back, equals the v1 document.
2. **Menu.** `horizons[].key` is exactly the set of registry horizons that have a stamp, shortest first;
   each fund's `horizons` has exactly those keys, each with `facts`, `signal`, `verdict`.
3. **Withholding.** For a horizon with `signal_in_verdict: false`, every fund's signal is `WITHHELD` with
   all numbers null and `signal_used` is `false`. A fixture horizon forced to PASS shows numbers only on
   `OK` rows.
4. **3y parity.** `horizons["3y"].verdict.verdict == signal_a.verdict` for every fund.
5. **Basis.** `return_basis` is `period` under 12 months and `annualized` from 12; the document serialises
   with `allow_nan=False`.

## 6. Hisaab Kitaab — brief for the app session

Nothing here is built in this repo. Rules that already bind the app and still do: Drift migrations are
additive only (Hard Rule 10); `FundAnalysis` is keyed on `amfiCode` and is never re-keyed
(`lib/models/fund_analysis.dart:105`); `signalA` and the verdict are stored and never rendered (D6); the
app never renders BUY.

**Schema — two additive tables, one migration:**

- `FundHorizonAnalysis`, primary key (`amfiCode`, `horizonMonths`): `horizonKey`, `trailingReturn`,
  `returnBasis`, `consistency`, `worst5pctReturn`, `nWindows`, `historyMonths`, `signalStatus`,
  `signalTopDecile`, `signalPercentile`, `signalProbability`, `verdictJson` (stored, never rendered),
  `modelId`, `artifactGeneratedAt`, `pipelineSha`, `ingestedAt`. All measures nullable.
- `HorizonStanding`, primary key `horizonMonths`: `horizonKey`, `standing`, `marginal`, `signalInVerdict`,
  `auc`, `aucLowerBound`, `effectiveN`, `modelId`, `evaluatedAt`, `artifactGeneratedAt`.

Vocabulary columns (`standing`, `signalStatus`, `returnBasis`) are free text, as `eligibility` already is:
the pipeline owns the vocabulary, and a value it adds later must not fail an ingest.

**Parser.** Accept `mf_artifact_v1` and `mf_artifact_v2`; anything else is still refused whole. v2 runs the
v1 parse unchanged, then reads `horizons`. A v2 document whose `horizons` is missing or malformed is
refused whole, not ingested as v1. An unknown `standing` is treated as not PASS.

**Picker.** Built from the artifact's `horizons` list, default `default_horizon`. For the chosen horizon:

- the facts, always, with the basis stated ("over the last month", "per year over 3 years");
- the horizon's standing in plain words — for WEAK and FAIL, that the model has shown no reliable signal
  at this horizon;
- a signal line **only** when `signalInVerdict` is true and the fund's `signalStatus` is `OK`, phrased as
  where the fund sits in its peer group, never as a probability headline and never as BUY;
- a marginal horizon's signal with a visible "marginal" qualifier;
- no verdict, as today.

Fail closed: signal numbers on a horizon whose `signalInVerdict` is false are a pipeline bug and are not
rendered.
