# Horizons — the menu, the recipe, and the bar

**Committed 2026-11-13, before any non-3y model was trained.** This file is the pre-registration: the
geometry, the effective-n divisor and the pass bar of every horizon are fixed here, and each horizon is
then trained and scored against it **once**. Nothing in this file may be edited after a horizon's number
exists except to append that number to "Results" at the bottom.

The honest prior, stated before looking: **short-horizon NAV labels are mostly noise.** A fund's place in
its peer cohort over one month is dominated by which style the market paid that month. Expect WEAK or
FAIL at 1m/3m/6m, and treat a PASS there with more suspicion than a FAIL.

## 1. The menu

| key | months | target | model_id | status before this work |
|---|---|---|---|---|
| `1m` | 1 | `cohort_q1_1m` | `phase_b_1m_v1_cohort` | new |
| `3m` | 3 | `cohort_q1_3m` | `phase_b_3m_v1_cohort` | new |
| `6m` | 6 | `cohort_q1_6m` | `phase_b_6m_v1_cohort` | new |
| `1y` | 12 | `cohort_q1_1y` | `phase_b_1y_v1_cohort` | new (geometry settled 2026-11-10) |
| `3y` | 36 | `cohort_q1` | `phase_b_v5_cohort` | shipped, frozen, **the default** |

**5y is excluded.** Only 286 of 565 funds have six years of NAV; a 60-month label ends its anchors at
2021-10, and the causal-holdout rule below would leave training on 2014-01..2015-10 alone. Its first
forward check would land in 2031.

**1m stays in** despite the prior, for one reason that has nothing to do with skill: it is the only
horizon whose ledger matures in weeks, so it is the only one that can prove the forward-test machinery
works before 2027.

**Rejected: horizon as a model input, or one multi-output model.** One fit cannot honour five different
purge radii (`mf_cv.purge_train_mask` drops by a single label horizon), and one ledger key cannot carry
five maturities.

Every horizon uses the **same** label definition (`y_cohort_q1`: top quartile of the fund's peer cohort by
forward return, cohorts under 4 dropped), the **same** feature set, the **same** fixed elastic-net
(`C=0.1`, `l1_ratio=0.5`), and the **same** calibration rule (last 25% of training anchors). No
horizon-specific features or tuning: each one added would be a researcher degree of freedom spent after
the bar was set.

## 2. The rule of a horizon

Everything below is a function of the horizon `h` (in months) and a data cutoff `D`. The 3y and 1y values
that were already decided fall out of it; a selftest in `mf_horizon.py` pins that.

| quantity | rule |
|---|---|
| Label window | `[t, end(t)]`. `end(t) = t + MonthEnd(h)` — month-end steps, so adjacent 1m windows tile the calendar exactly. **3y keeps its legacy `t + DateOffset(years=3)`**, because its labels and ledger rows are frozen. |
| Purge | `h`. A training anchor is dropped when its window touches or overlaps any test window. |
| Embargo | 1 month, provided the pooled label autocorrelation one month past the window is under 0.10. Measured: 1m 0.04, 3m 0.05, 6m 0.07, 1y 0.07, 3y −0.05. So **1 month for every horizon**. |
| Shortfall | `max(h/12, 7 days)`. Minimum window `= h − shortfall`. |
| Staleness tolerance | `min(10, ceil(shortfall in days))` days, at both ends of the window. |
| Anchor grid | month-ends `2013-01-31 .. D − h`. Modelling starts 2014-01-31 (2013 anchors only feed peer benchmarks). |
| Data cutoff `D` | **2026-10-31** for 1m/3m/6m/1y. NAV is truncated at `D` when labels are built and the cache is read as it stands — no refetch. 3y stays on its frozen 2026-08-07 parquet (anchors to 2023-07-31). |
| Causal holdout | test = the **last 12 labelled anchors**; train = anchors `<= test_start − h − embargo`. |
| CPCV blocks | 2-year calendar blocks from 2014-01; the last block is cut at the anchor end; a final block shorter than 12 months merges into the one before it. |

**Why the shortfall has a 7-day floor.** 3y's `2.75/3` ratio is "one twelfth short is tolerable". At 1m one
twelfth is 2.5 days: a month-end on a weekend would fail the window for every fund at once and delete
the whole cross-section. Seven days covers a weekend plus a holiday run, and makes staleness — not the
window ratio — the binding rule where the window is short.

**Why month-end anchors survive at 1m.** With month-end steps the 1m windows do not overlap at all. A
denser grid would add rows and overlap in equal measure: more nominal n, a larger divisor, no more
information.

### The pinned values

| key | min window (y) | tol (d) | anchor end | holdout test | train `<=` | CPCV blocks |
|---|---|---|---|---|---|---|
| `1m` | 0.06417 | 7 | 2026-09-30 | 2025-10-31 .. 2026-09-30 | 2025-08-31 | B1–B5 as 1y, B6 2024-01 .. 2026-09 |
| `3m` | 0.22917 | 8 | 2026-07-31 | 2025-08-31 .. 2026-07-31 | 2025-04-30 | B1–B5 as 1y, B6 2024-01 .. 2026-07 |
| `6m` | 0.45833 | 10 | 2026-04-30 | 2025-05-31 .. 2026-04-30 | 2024-10-31 | B1–B5 as 1y, B6 2024-01 .. 2026-04 |
| `1y` | 0.91667 | 10 | 2025-10-31 | 2024-11-30 .. 2025-10-31 | 2023-10-31 | B1–B4 as today, B5 2022-01 .. 2023-12, B6 2024-01 .. 2025-10 |
| `3y` | 2.75 | 10 | 2023-07-31 | 2022-08-31 .. 2023-07-31 | 2019-07-31 | B1–B5 exactly as `mf_cv.BLOCKS` today |

Expected label counts (anchors from 2014-01), from an independent scratch implementation of the rule. The
real label build must land on these within 1% before any model is fitted; a larger gap is a bug in one of
the two and stops the run.

| key | funds | rows | anchors | base rate | holdout rows |
|---|---|---|---|---|---|
| `1m` | 556 | 46,601 | 153 | 0.275 | 6,447 |
| `3m` | 553 | 45,489 | 151 | 0.275 | 6,359 |
| `6m` | 550 | 43,837 | 148 | 0.276 | 6,205 |
| `1y` | 520 | 40,622 | 142 | 0.277 | 5,872 |
| `3y` | 371 | 28,434 | 115 | 0.279 | 4,132 |

## 3. The effective-n divisor

Nominal n overstates the evidence twice over: a fund's labels at neighbouring anchors repeat each other
(serial), and funds in one cohort at one anchor win or lose together (cross-fund).

**Serial part, measured on labels only.** Pooled within-fund autocorrelation `ρ_k` of the 0/1 label at
monthly lag `k`; the design effect of a 12-anchor block mean — the holdout is always 12 anchors — is
`DEFF12 = 1 + 2·Σ_{k=1..11} (1 − k/12)·ρ_k`. Take the larger of the all-anchor and the pre-holdout value.

**Cross-fund part.** 3y's long-standing divisor of 30 was never decomposed. Against its measured serial
DEFF12 of 8.76 it implies a cross-fund allowance `A = 30 / 8.76 = 3.42`. That allowance carries to another
horizon **only if cross-fund dependence there is no stronger**, and that is measured, not assumed: for
three fixed model-free scores (12m momentum ex last month, trailing 3y CAGR, minus 1y volatility), the
variance of the per-anchor AUC across non-holdout anchors divided by its Hanley–McNeil variance, median
over the three scores (`CS`).

```
κ(h)        = max(1, CS(h) / max(CS(1y), CS(3y)))        # can only raise the allowance, never lower it
divisor(h)  = ceil( max(DEFF12, DEFF12 pre-holdout) × A × κ(h) )
strict(h)   = ceil( max(DEFF12, DEFF12 pre-holdout) × max(A, CS(h)) )
```

| key | ρ₁ | DEFF12 | DEFF12 pre | CS | κ | **divisor** | strict |
|---|---|---|---|---|---|---|---|
| `1m` | 0.046 | 1.39 | 1.38 | 5.69 | 1.256 | **6** | 8 |
| `3m` | 0.466 | 2.65 | 2.66 | 4.55 | 1.004 | **10** | 13 |
| `6m` | 0.634 | 4.28 | 4.33 | 4.10 | 1 | **15** | 18 |
| `1y` | 0.733 | 6.24 | 6.38 | 4.53 | 1 | **23** | 29 |
| `3y` | 0.845 | 8.76 | 8.73 | 4.33 | 1 | **30** | 38 |

Notes that matter:

- The brief's suspicion holds at one month only. Cross-fund dependence is a quarter stronger at 1m than at
  1y/3y and indistinguishable from them at 3m and 6m.
- **1y is 23, not the 22 the formula gives today.** It was settled at 23 on 2026-11-10 from a pre-holdout
  DEFF12 of 6.49; the cache has moved and it now measures 6.38. The larger divisor is the stricter bar, so
  the earlier decision stands.
- **The `strict` column is an honest caveat about `A`.** Measured directly, cross-fund dependence at 1y
  and 3y is 4.3–4.5, above the 3.42 that 3y's 30 implies. The measurement is rough (the three scores range
  from 2.5 to 10), so it does not overturn the settled 23 and 30 — but it says they sit at the lenient end.
  Both bars are therefore fixed here, before training. See §4 for what happens between them.
- These are pinned constants. The measurement script is kept in the repo for reproduction; re-running it
  later does not move a divisor.

## 4. The bar

One formula, evaluated on the causal holdout's elastic-net metrics, with `n_pos` and `n_neg` each divided
by the horizon's divisor:

```
SE      = Hanley–McNeil standard error of the holdout AUC at (n_pos/divisor, n_neg/divisor)
(a) AUC − 1.645·SE > 0.50                      # pooled_auc
(b) within-anchor rank-AUC > 0.50              # rank_auc
(c) top-decile lift > 1                        # lift_over_base
(d) CPCV pooled AUC > 0.50 in at least ceil(2/3 × blocks) blocks   # 4 of 6; 4 of 5 for 3y

PASS  = (a) and (b) and (c) and (d)
WEAK  = (b) and (c) and (d), not (a)
FAIL  = anything else
```

What (a) asks for in practice, at the expected holdout sizes:

| key | effective n | AUC needed | at the strict divisor |
|---|---|---|---|
| `1m` | ~1,074 | ≥ 0.533 | ≥ 0.538 |
| `3m` | ~636 | ≥ 0.543 | ≥ 0.549 |
| `6m` | ~414 | ≥ 0.553 | ≥ 0.558 |
| `1y` | ~255 | ≥ 0.567 | ≥ 0.575 |
| `3y` | 138 | ≥ 0.592 | ≥ 0.603 |

Rules around the number:

- **One evaluation.** The holdout is scored once per `model_id`; the result is written to a git-tracked
  stamp and a second run is refused. There is no force flag for a horizon stamp. No refit after the number.
- **Marginal PASS.** A horizon that clears (a) at its divisor but not at its strict divisor is recorded as
  `PASS` with `marginal: true`, and **its signal stays out of the verdict** until the owner rules on it.
- **Four tests, one family.** Four new horizons at a one-sided 5% each is roughly a one-in-five chance of
  at least one PASS if none has any skill. A PASS here means "cleared a bar set in advance", not "proven";
  the ledger's forward test is the confirmation.
- **3y, scored against the same bar without refitting:** AUC 0.537, lower bound 0.445, rank-AUC 0.536, lift
  1.44, CPCV above 0.5 in 4 of 5 blocks → **WEAK**. Nothing about 3y changes.
- A bug in labels, geometry or data found after a number exists is fixed and re-run under a **new
  `model_id`**, with the old result kept in the stamp history and the reason written down. A disappointing
  number is not a bug.

## 5. Models, files and retraining

| what | 3y (untouched) | every other horizon |
|---|---|---|
| inference payload | `model/model_artifact_cohort.json` | `model/model_artifact_cohort_<key>.json` |
| PSI reference | `model/psi_reference.json` | `model/psi_reference_<key>.json` |
| holdout stamp + standing | `model/holdout_stamp_3y.json` (written from the existing frozen stamp; no rerun) | `model/holdout_stamp_<key>.json` |
| research outputs (gitignored) | `mf_cache/phase_b/` | `mf_cache/phase_b/<key>/` |

The stamp is git-tracked on purpose: the evaluate-once guard used to live only in gitignored `mf_cache/`,
which is exactly what a dead session loses. A WEAK or FAIL model is still committed — it is needed to log
its predictions.

**Retraining.** Never on a schedule (`model/README.md`). A horizon may be refitted at most once per 12
newly labelled anchors, so that its new holdout shares no anchor with the last one, under a new version
(`phase_b_<key>_v2`) and a new stamp against this same bar. Refitting until something passes is
manufacturing the target.

## 6. The verdict per horizon

Today's rule is five colours (full-history CAGR, excess vs benchmark, Sortino, 3y max drawdown,
consistency), a screen-score check and the compliance gates. The cohort signal has never been part of it.
Per horizon, the rule keeps that shape, and changes only where a change needs no invented number:

- **Consistency becomes horizon-matched:** the share of rolling-`h` windows with a non-negative return,
  with the same 0.60 / 0.40 thresholds (a share is dimensionless, so the thresholds carry). For `h < 36`
  it needs at least 36 months of NAV; otherwise it is null, counts as neither green nor red, and raises a
  coverage flag.
- **CAGR, excess, Sortino and 3y max drawdown keep their windows.** A drawdown threshold has no principled
  one-month analogue, and −20% applied unscaled to a worst month would be green for almost every fund —
  missing judgement dressed as a pass. Horizon-matched severity is published as a **fact** (the
  5th-percentile rolling-`h` return) and is never coloured.
- **The signal (owner's option 1).** Only for a horizon standing at PASS and not marginal, the rule gains a
  sixth colour: **green** when the fund's signal status is OK and it sits in the model's top decile of its
  cohort (the set the bar certified with lift > 1), neutral otherwise. It is **never red** — the bar tests
  nothing about the bottom. BUY and SELL thresholds are unchanged, so the signal can add one green and
  nothing else. WEAK, FAIL and marginal horizons run the five-colour rule.
- **3y's per-horizon verdict must equal today's verdict for every fund** (rolling-36m consistency is
  today's consistency, and 3y is WEAK). A selftest pins it.

## 7. The ledger

- **One directory per target, one file per anchor month:** `ledger/<target>/<YYYY-MM>.jsonl`. Row schema,
  hash and dedupe key `(amfi_code, target, anchor, model_id)` are unchanged.
- **`ledger/predictions.jsonl` is never rewritten or moved.** It stays as the legacy 3y file (68.4 MiB on
  2026-11-13, about six weeks from GitHub's 100 MiB push limit). From the rotation commit on, 3y rows
  append to `ledger/cohort_q1/<YYYY-MM>.jsonl`. Readers take the legacy file plus the target's directory;
  dedupe looks across both.
- **Readers are per target.** `current_model_id`, the monitoring block and realization each take a target,
  so another horizon's rows cannot leak into the frozen `mf_artifact_v1` monitoring block.
- **Maturity is looked up per target** from the registry (window end, minimum window, tolerance). A row
  whose target is not in the registry is skipped and counted, never guessed.
- **New horizons log month-end anchors only.** The forward test should sample what the holdout sampled.
  On each nightly run, for the latest month-end `M` no later than the newest NAV date: if the run is
  within 10 days of `M` and the target has no rows at `M`, the panel is built as of `M` (NAV `<= M` only,
  each fund within the horizon's staleness tolerance) and one row per OK fund is appended with
  `anchor = M`. After 10 days the month is skipped and counted. Daily rows for four more targets would add
  several hundred megabytes a year to the repo for no extra independent evidence.
- **Every horizon is logged, PASS or not.** No backfill. First anchors 2026-11-30 if this ships before
  then; first maturities: 1m 2026-12-31, 3m 2027-02-28, 6m 2027-05-31, 1y 2027-11-30.
- **A row is realized only once the fund's NAV reaches the window end**, or the window end is more than
  the staleness tolerance in the past. A monthly job that runs on the 1st would otherwise close a 1m window
  on a NAV one or two days short of it, permanently.
- 3y keeps logging daily, as today. That is an existing cost (over 200 MB a year), and changing it would
  change v1 monitoring, so it is left for the owner.

## 8. The nightly

Features do not depend on the horizon, so one live panel per run serves all five models; scoring four more
is four more dot products per fund. The month-end ledger rows need one extra panel build on the first run
after a month-end. The emit step is bound by the throttled NAV fetch (33–45 of the 90 minutes). The added
work must measure under 5 minutes locally over all 565 funds; if it does not, it does not ship in the
nightly until it does.

## 9. Results

*Appended once per horizon, after its single evaluation. Nothing above this line changes.*

| key | model_id | evaluated | n | eff-n | AUC | lower bound | rank-AUC | lift | CPCV blocks > 0.5 | standing |
|---|---|---|---|---|---|---|---|---|---|---|
| `3y` | `phase_b_v5_cohort` | 2026-08-07 | 4,132 | 138 | 0.537 | 0.445 | 0.536 | 1.44 | 4 of 5 | **WEAK** |
| `1m` | `phase_b_1m_v1_cohort` | 2026-11-13 | 6,447 | 1,074 | 0.535 | 0.502 | 0.534 | 1.21 | 6 of 6 | **PASS, marginal** |
| `3m` | `phase_b_3m_v1_cohort` | 2026-11-13 | 6,359 | 636 | 0.548 | 0.505 | 0.546 | 1.32 | 5 of 6 | **PASS, marginal** |
| `6m` | `phase_b_6m_v1_cohort` | 2026-11-13 | 6,205 | 414 | 0.541 | 0.487 | 0.541 | 1.40 | 5 of 6 | **WEAK** |
| `1y` | `phase_b_1y_v1_cohort` | 2026-11-13 | 5,872 | 255 | 0.540 | 0.472 | 0.540 | 1.35 | 5 of 6 | **WEAK** |

Each new horizon was fitted and scored exactly once (`mf_model.py --stage cohort --horizon <key>`, offline,
on the labels built from the cache cut at 2026-10-31). The full numbers are in
`model/holdout_stamp_<key>.json`.

How to read it, written the same day and not revised:

- **No horizon is a clean PASS.** 1m and 3m clear the lower-bound test at their divisor (0.502, 0.505) and
  miss it at the strict divisor (0.497, 0.499). By the rule fixed in §4 they are `PASS` with
  `marginal: true`, and **their signals stay out of the verdict until the owner rules.**
- **The edge is the same size everywhere.** AUC runs 0.535–0.548 across the four new horizons, against
  3y's 0.537. The short horizons pass where the long ones do not because their windows overlap less and
  so carry more independent evidence, not because the model ranks better at one month.
- **This is exactly the case the prior warned about.** Two of four clearing a one-sided 5% bar by 0.002
  and 0.005 is within what four null tests can produce (§4, "four tests, one family"). The ledger's
  forward test — 1m from early 2027 — is what would turn "marginal" into anything stronger.
- The HistGBT challenger scored higher than the elastic-net on every new horizon's holdout (0.555–0.574).
  The elastic-net was the pre-registered model, so that is recorded and nothing is switched after the
  number.
