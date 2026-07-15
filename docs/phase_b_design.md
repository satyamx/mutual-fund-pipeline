# Phase B Design — Predicting "Good" Fund-Windows (3-Year Forward OR-Label)

**Status:** DESIGN ONLY — no implementation in this document's scope.
**Author:** Phase-B design pass (claude-fable-5), 2026-07-14.
**Grounding:** every number in this document was measured from the real cached data in
`mf_cache/` (136 funds, full mfapi NAV histories) using read-only probe scripts run in a
scratch directory. Nothing below is a guess unless explicitly marked as one.

---

## 0. Measured reality (probe results that anchor everything)

Probe method: monthly month-end anchors 2013-01 to 2023-07; per (fund, anchor), the
3-year-forward annualized return from real cleaned NAVs; peer benchmark = leave-one-out
median of the fund's peer group's forward returns (peer-group rules in §1.3); OR-label
evaluated exactly as confirmed.

| Fact | Measured value |
|---|---|
| Funds with usable NAV | **136 / 136** |
| Usable (fund, anchor) samples | **15,037** (14,784 with a valid LOO peer benchmark, 98.3%) |
| **Base rate P(y=1), OR-label** | **0.842** |
| P(absolute condition alone: fwd ≥ 10% p.a.) | 0.806 |
| P(excess condition alone: fwd − peer ≥ +2pp) | 0.295 |
| P(excess-only, i.e. what the OR adds beyond absolute) | 0.036 |
| Negatives available in total | 2,372 (15.8%) |
| Consecutive-month label agreement (window-overlap severity) | **0.935** |
| Anchors per fund | median 126; 120 funds ≥ 60; 113 funds ≥ 100 |
| Samples with ≥ 3y trailing history for features | 10,532 (70.0%); base rate there 0.835 |
| Samples with ≥ 1y trailing history | 89.8% |

Base rate by anchor year (the single most important honesty table):

| Anchor year | n | P(y=1) | P(fwd ≥ 10%) | P(excess ≥ +2pp) | negatives |
|---|---|---|---|---|---|
| 2013 | 1,335 | 0.957 | 0.954 | 0.339 | 58 |
| 2014 | 1,356 | 0.914 | 0.908 | 0.304 | 117 |
| 2015 | 1,370 | 0.704 | 0.689 | 0.269 | 405 |
| 2016 | 1,404 | 0.652 | 0.597 | 0.291 | 488 |
| 2017 | 1,413 | 0.405 | 0.134 | 0.343 | 841 |
| 2018 | 1,434 | 0.790 | 0.762 | 0.319 | 301 |
| 2019 | 1,446 | 0.959 | 0.957 | 0.306 | 60 |
| 2020 | 1,476 | 0.991 | 0.991 | 0.307 | 14 |
| 2021 | 1,501 | 0.985 | 0.985 | 0.271 | 22 |
| 2022 | 1,527 | 0.980 | 0.979 | 0.236 | 31 |
| 2023 | 775 | 0.955 | 0.954 | 0.252 | 35 |

**Three structural conclusions that drive the whole design:**

1. **The OR-label is dominated by its absolute leg, and the absolute leg is a market-regime
   variable, not a fund-skill variable.** In 8 of 11 anchor years, ≥ 90% of *all* funds
   clear 10% p.a. forward; in 2017 anchors only 13% do. A model fed fund-level NAV features
   cannot predict the market regime three years out, and must not pretend to. The excess
   condition, by contrast, is regime-stable (24–34% every single year) — that is where the
   genuine fund-discrimination signal lives.
2. **Accuracy is a nearly worthless headline at an 84% base rate.** "Predict 1 for
   everything" scores 84.2%. The user-floated 90–95% in-sample accuracy is trivially
   reachable and means almost nothing; it will be reported only as an in-sample diagnostic,
   never as evidence. The honest headline metrics are within-anchor ranking quality, lift
   over base rate, negative-class capture, and calibration (§5).
3. **The 15,037 samples are ~30× inflated by window overlap.** Consecutive monthly labels
   agree 93.5% of the time because their 3-year windows share 35 of 36 months. The number
   of *independent* observations is roughly (funds × non-overlapping 3y windows in a
   ~10.5-year anchor span) ≈ 136 × 3.5 ≈ **~475 effective samples**. Every choice below —
   interpretable-first model, purged CV, block bootstrap for coefficient stability — is
   sized to ~475 effective samples, not to 15,037 nominal rows.

---

## 1. Label construction

### 1.1 Anchor-date scheme

- Anchor grid: **calendar month-end dates** (`pd.date_range(..., freq="ME")`).
- Anchor value: last available NAV on or before the anchor date, **staleness tolerance
  10 calendar days** (an anchor with no NAV print in the prior 10 days yields no sample —
  guards against fund suspension periods and data gaps).
- Anchor span for the modeling table: **2014-01-31 through 2023-07-31** (last anchor whose
  3y forward window closes inside the cached history ending 2026-07). The 2013 anchors are
  usable for the label but have < 1y of trailing history for features (direct plans began
  2013-01), so they enter only as label-side peer-benchmark contributors, not as training
  rows.

### 1.2 Forward return (both OR-conditions share it)

For fund *i* at anchor *t*:

```
NAV0 = last NAV at date d0 ≤ t            (d0 within 10 days of t)
NAV1 = last NAV at date d1 ≤ t + 3 years  (d1 within 10 days of t+3y)
yrs  = (d1 − d0) / 365.25                 (require yrs ≥ 2.75, else no sample)
R_fwd(i,t) = (NAV1 / NAV0)^(1/yrs) − 1    (annualized, i.e. "yoy"/CAGR)
```

NAVs come from the cached `mfapi_<code>.json` files after cleaning with the **existing
`NAVCleaner` logic** (`mf_datasources.py`): pass-1 non-persistent-spike removal always on;
IDCW payout reconstruction is structurally unnecessary because bootstrap resolved every
fund to its Direct-Growth plan (verified: no IDCW plans in the manifest).

### 1.3 Peer/sector benchmark ("create our own benchmarks")

There is no historical index data on disk, and fetching index TRI histories is out of
scope; the universe itself is the benchmark material. Two-layer construction:

- **Diversified categories** (Flexi, Large, Mid, Small, Large&Mid, ELSS, Value/Contra,
  Multi, Focused): benchmark of fund *i* at anchor *t* is the **leave-one-out (LOO) median**
  of `R_fwd(j,t)` over all funds *j ≠ i* in the same manifest `category` with a valid
  forward return at *t*. Peer pools are 7–9 funds throughout (measured median 9).
- **Sectoral/Thematic funds**: if the fund's manifest `sector` has **≥ 3 funds**
  (Banking/Financial, IT, Pharma, FMCG, Infra, Auto, Energy, PSU, MNC, Manufacturing —
  measured pools of 101–630 samples), benchmark = LOO median of same-sector peers.
  Otherwise (Quant, Housing, Exports, Commodities, Business Cycle = 2 funds each;
  Defence, ESG = 1 fund) the fund falls back to the **pooled small-sector thematic
  median** (LOO), and the sample is tagged `benchmark_quality = "pooled_fallback"` so it
  can be excluded in sensitivity runs. For Defence/ESG the "sector benchmark" is honestly
  not a sector benchmark; this is disclosed, not hidden (§8).
- Median, not mean: robust to the single-fund blowups that a 7-fund pool cannot average
  away.
- Minimum pool: **≥ 2 peers after LOO exclusion**, else condition A is `NaN → False`
  (conservative: the fund can then only qualify via the absolute leg). Measured impact:
  only 1.7% of samples lack a valid benchmark.

**Why LOO matters:** without it, a sector with 3 funds would have each fund ~33%
benchmarked against itself, mechanically shrinking measured excess toward zero.

**Known bias, accepted and disclosed:** peers are today's survivors, so the peer median is
upward-biased relative to the true investable peer set at time *t*. This makes condition A
*harder* to pass (conservative direction for a "good fund" label), but it also means
measured historical excess understates true excess. See §7.1.

### 1.4 The OR-label, exactly

```
condA(i,t) = [ R_fwd(i,t) − Bench_fwd(i,t) ≥ +0.02 ]     (excess vs peers, +2pp p.a.)
condB(i,t) = [ R_fwd(i,t) ≥ 0.10 ]                       (absolute, 10% p.a.)
y(i,t)     = condA OR condB
```

Both legs are stored separately alongside `y` in the label table so that per-leg
diagnostics (and the optional excess-only companion model, §5.4) are free.

### 1.5 Point-in-time rules

- The **label** may use data through `t + 3y` — that is its definition. The **features**
  may use data through `t` only (strict as-of rule, §2).
- The benchmark inside the label uses peer forward returns over the *same* window as the
  fund's own forward return: contemporaneous with the label, not lookahead *within the
  labeling convention*. The lookahead sin to avoid is the peer *pool membership* being
  chosen with future knowledge — it unavoidably is (survivors), and §7 mitigates by
  disclosure and sensitivity analysis, because point-in-time universe reconstruction is
  not possible with the data on disk.
- Fund→peer mapping uses the manifest `category`/`sector` as of today. Category
  reclassifications (e.g., SEBI 2017/2026 recategorisation waves) are not reconstructible
  from NAV data; treated as a documented limitation, not silently ignored.

---

## 2. Feature set (all strictly as-of the anchor `t`)

Global as-of rule: every feature is a function of `NAV[fund, dates ≤ t]` and
`NAV[peer funds, dates ≤ t]` only. Enforcement is mechanical: the feature engine receives
a series **truncated at `t`** (`nav.loc[:t]`) and physically cannot see the future. A unit
test perturbs post-`t` NAVs and asserts bit-identical features (§7.4, task 3).

Trailing windows: 1y = 252 trading obs (or date-based `t−365d`), 3y, 5y. A feature whose
window extends before the fund's first NAV is `NaN` + a paired missingness indicator
(logistic handles this as an explicit "young fund" signal instead of silent imputation).

### 2.1 Own-NAV features

| # | Feature | Computation (all on NAV ≤ t) |
|---|---|---|
| 1–3 | `cagr_1y, cagr_3y, cagr_5y` | `(NAV_t/NAV_{t−w})^(1/w) − 1`, date-based endpoints, 10-day staleness rule |
| 4–5 | `vol_1y, vol_3y` | std of daily returns × √252 (`QuantEngine.annualised_vol`) |
| 6 | `sharpe_3y` | `QuantEngine.sharpe` on trailing 3y daily returns |
| 7 | `sortino_3y` | `QuantEngine.sortino`, MAR = risk-free |
| 8 | `max_dd_3y` | `QuantEngine.max_drawdown` on trailing 3y levels |
| 9 | `current_dd` | NAV_t / trailing-3y running max − 1 (distance from high-water mark) |
| 10 | `mom_6m` | 6-month simple return |
| 11 | `mom_12m_ex1m` | 12-month return excluding last month (classic momentum, avoids 1-month reversal) |
| 12 | `rr3y_neg_share` | share of trailing daily-stepped rolling-3y CAGRs < 0 (`QuantEngine.rolling_returns` on data ≤ t) — consistency |
| 13 | `age_years` | (t − first_nav_date)/365.25 |

### 2.2 Peer-relative features (vs the same peer pool as §1.3, trailing not forward)

The trailing peer composite is an **equal-weight daily-rebalanced index of peer funds'
NAV returns** (peers with data at that date; LOO). This is the point-in-time twin of the
label benchmark.

| # | Feature | Computation |
|---|---|---|
| 14–15 | `excess_1y, excess_3y` | trailing fund CAGR − trailing peer-composite CAGR |
| 16 | `rank_3y` | percentile rank of `cagr_3y` within (anchor, peer pool) |
| 17 | `beta_3y` | `QuantEngine.beta_alpha` vs peer composite, trailing 3y daily |
| 18 | `te_3y` | tracking error vs peer composite |
| 19 | `ir_3y` | information ratio vs peer composite |
| 20–21 | `upcap_3y, downcap_3y` | `QuantEngine.capture_ratios` vs peer composite |
| 22 | `excess_persist` | share of trailing 12 quarterly checkpoints where trailing-1y excess > 0 (rolling-CAGR persistence vs peers) |

### 2.3 Sector/segment context (thematic funds; NaN + indicator for diversified)

| # | Feature | Computation |
|---|---|---|
| 23 | `sector_mom_12m` | trailing 12m return of the fund's sector composite (the "real per-sector return proxy" built from sectoral-fund NAVs) |
| 24 | `sector_vol_1y` | trailing 1y vol of the sector composite |
| 25 | `sector_rel_strength` | sector composite 12m return − all-thematic composite 12m return |
| 26 | `is_thematic` | indicator (also the missingness gate for 23–25) |

### 2.4 Cohort normalization (design decision, not a feature)

Raw return levels encode the regime (a 25% `cagr_3y` in 2017 ≠ in 2021). Every level-type
feature (1–3, 10–11, 14–15, 23, 25) is additionally supplied as a **within-anchor
cross-sectional z-score** (across all 136 funds at that anchor). The model sees both the
raw and the cohort-normalized version; regularization decides which carries weight. This
is the main defense against the model degenerating into a regime-memorizer (§0,
conclusion 1).

### 2.5 Explicitly excluded from backtest features

- **Current portfolio holdings / cap-band mix** — CONFIRMED decision: current holdings are
  not historical point-in-time; using them for 2016 anchors is lookahead. They appear only
  in the live-scoring layer (§6.4).
- **TER, AUM** — no historical series on disk; excluded rather than proxied.
- **Fund-identity one-hots** — with ~475 effective samples and surviving funds only, a
  per-fund dummy is a leakage vector for the fund's full-history average outcome.

Feature count: ~26 raw + ~10 cohort-z + ~6 missingness indicators ≈ **40 columns**, which
against ~475 effective samples is already generous; the elastic-net penalty (§4) is the
control.

---

## 3. Train / validation / test design

### 3.1 The #1 leakage risk: overlapping 3-year label windows

Adjacent monthly anchors share 35/36 of their forward window (measured label
autocorrelation 0.935). A random split would put near-duplicates of test rows into
training and produce fraudulent 95%+ scores. Mandatory countermeasures:

- **Purging:** a training sample (i, t_train) is dropped if its label window
  `[t_train, t_train+3y]` overlaps any test anchor's window `[t_test, t_test+3y]` —
  i.e., drop training anchors within **3 years before** a test block and inside it.
- **Embargo:** an additional **1 month** after the test block's window end before training
  anchors resume (guards serial correlation of returns beyond the window itself).

### 3.2 Why strict walk-forward alone cannot evaluate this label — and what to do

The negatives are concentrated in 2015–2017 anchors (73% of all negatives). A strict
walk-forward scheme can only test 2015–2017 by training on 2013–2014-minus-purge — i.e.,
on almost nothing (3y purge before a 2016 test block leaves only 2013-01…2013-01 anchors
with ≥ 1y features). Conversely the testable late blocks (2019–2023) contain 36–66
negatives out of ~2,400+ samples — no statistical power on the negative class.

Design: **Combinatorial Purged Cross-Validation (CPCV, López-de-Prado-style) as the
primary evaluation, plus one strictly-causal final holdout as the headline.**

- **CPCV blocks (5):** B1 2014-01…2015-12, B2 2016-01…2017-12, B3 2018-01…2019-12,
  B4 2020-01…2021-12, B5 2022-01…2023-07 (measured sizes 2,300–4,100 samples; negatives
  580 / 1,329 / 361 / 36 / 66 respectively).
- Each fold: test = one block; train = all other blocks after purging every training
  anchor whose window overlaps the test block's windows (3y on each side) + 1-month
  embargo. Training on data *after* the test block is permitted **and clearly labeled**:
  it answers "do these factor weights discriminate in an unseen regime," which is the
  question an interpretable factor model must pass. It is not a simulation of live
  deployment.
- **Strictly-causal holdout (the headline number):** train on anchors ≤ 2019-07 (so all
  training windows close by 2022-07), test on anchors 2022-08…2023-07 — untouched during
  all model development, evaluated exactly once. Weak negative-class power (~35–60
  negatives) is acknowledged in the report next to the number.
- Fold count sanity: 5 CPCV folds + 1 causal holdout, on a median 13.5y history with a 3y
  horizon, is the honest maximum; more folds just re-slice the same ~3.5 independent
  windows per fund.

### 3.3 Class-balance handling

- `class_weight="balanced"` in the logistic loss (reweighting, **never** duplication/SMOTE —
  synthetic oversampling of overlapping windows manufactures leakage).
- All threshold-based metrics reported per test block (a pooled metric would let B4/B5's
  98% positive blocks drown B2).
- Calibration fitted per fold on a purged validation slice of the training side, never on
  test.

### 3.4 Statistical inference at ~475 effective samples

Coefficient stability via **block bootstrap over anchor half-years** (resample time blocks,
refit, report coefficient sign-stability %). A factor whose sign flips across bootstrap
replicates is reported as "not evidenced," regardless of its point estimate.

---

## 4. Model plan

### 4.1 Interpretable-first primary model

- **Elastic-net logistic regression** (`sklearn.linear_model.LogisticRegression`,
  `penalty="elasticnet"`, `solver="saga"`; C and l1_ratio chosen on purged validation
  slices only). Features standardized on training folds (means/stds computed on train
  only — a small but real leakage vector otherwise).
- Missingness indicators enter as features; missing numerics imputed with the training-fold
  median (never pooled).
- **Calibration:** isotonic regression (`CalibratedClassifierCV`-style, but fitted on a
  purged validation slice, not via naive CV which would violate purging). Fall back to
  Platt (sigmoid) if the validation slice for a fold has < ~300 samples or < 30 negatives
  — isotonic overfits sparse tails. The deliverable is a **calibrated P(y=1 | features)**.

### 4.2 Challenger and the ≤ 3-point rule (CONFIRMED)

- Challenger: `sklearn.ensemble.HistGradientBoostingClassifier` (no extra dependency
  beyond scikit-learn, handles NaN natively, monotonic-constraint support if wanted).
- Decision rule: the challenger replaces the logistic **only if** it beats it by **more
  than 3 points of mean out-of-sample AUC (in AUC percentage points, e.g. 0.74 → > 0.77)
  across the CPCV folds, and also does not lose on the causal holdout**. Expected outcome
  at ~475 effective samples: it will not clear the bar; the comparison is still run and
  reported.
- If the challenger wins, explainability is preserved via permutation importance +
  partial-dependence on the top factors; the ScoringEngine weight review (§4.3) then uses
  the logistic anyway (it remains the interpretable companion).

### 4.3 Factor weights → ScoringEngine (satisfies the deferred "Step 2" weight review)

The standardized logistic coefficients are a direct, data-derived review of
`ScoringEngine.FACTOR_MAP[SEBICategory.EQUITY]` (`mf_pipeline.py`, ~line 907):

1. Map model features to FACTOR_MAP metrics where they correspond
   (`sortino_3y → sortino`, `ir_3y → info_ratio`, `upcap_3y/downcap_3y →
   upside/downside_capture`, `max_dd_3y → max_drawdown`, `excess_3y ≈ alpha`,
   `cagr_3y/rr3y_neg_share ≈ rr3y_mean` family).
2. Proposed weight for metric m = `|coef_m| / Σ|coef|` (sign must agree with FACTOR_MAP
   direction; a sign disagreement with > 80% bootstrap stability is a *finding*, presented
   for user decision, not auto-applied).
3. Deliverable: a side-by-side table (current hand-set weight vs data-derived weight vs
   bootstrap sign-stability) + a proposed updated FACTOR_MAP. **No code change to
   ScoringEngine until the user signs off.**

---

## 5. Metrics & honest success criteria

### 5.1 Reported metrics (all out-of-sample, per CPCV block + causal holdout)

| Metric | Why it's here |
|---|---|
| ROC-AUC (pooled per block) | standard discrimination summary |
| **Within-anchor rank-AUC** | AUC computed inside each monthly cohort then averaged — immune to regime base-rate drift; *this is the fund-selection metric* |
| Precision@top-decile (within anchor cohorts) | "if you bought the model's top 10% picks" |
| **Lift over base rate** = precision@decile ÷ block base rate | the honesty normalizer |
| Recall (positive class) at the calibrated 0.5 threshold | completeness |
| **Bottom-decile negative capture** | share of a block's negatives in the model's lowest decile — at an 84% base rate, *avoiding the bad 16% is the real economic value* |
| Brier score + reliability curve | is P(y) actually a probability |
| In-sample accuracy/AUC | diagnostic ONLY, printed in a section titled "in-sample (not evidence)" |

### 5.2 Realistic expectations vs the measured base rate — stated up front

- Base rate 0.842 ⇒ mathematical ceiling on positive-class lift at the top decile =
  1/0.842 ≈ **1.19×**. Even a perfect model cannot show a dramatic positive-class lift.
  Anyone promising 2× lift on this label misunderstands it.
- The discriminating room is on the negative side: a bottom decile capturing 30–50% of
  negatives (vs 10% for random) — i.e., **3–5× negative-class lift** — is a realistic and
  genuinely useful target.
- Plausible honest range for within-anchor rank-AUC given NAV-only features and ~475
  effective samples: **0.55–0.65** (guess, clearly marked as such; persistence literature
  on Indian equity funds supports weak-but-positive short-horizon persistence). If it
  comes out at 0.52, that is the number that gets reported.
- Accuracy will be ~0.84–0.90 no matter what; it appears only alongside the trivial
  predict-all-1 baseline of 0.842.

### 5.3 Presentation rules (so nothing misleads)

Every reported table carries: the block's base rate in an adjacent column; nominal n AND
effective n (≈ n/30 for monthly-overlapping 3y windows); the sentence "a constant
all-positive classifier scores accuracy 0.842 and AUC 0.5 on this label."

### 5.4 Companion diagnostic head (flagged for sign-off, not a relitigation)

Because condition B is regime-driven (§0), the report will *additionally* show the same
model architecture trained on **condition A alone** (excess ≥ +2pp; base rate 0.295,
regime-stable 24–34% every year). The OR-label model remains the confirmed deliverable;
the condA companion is a diagnostic that isolates fund-skill discrimination from regime
luck, and is where the ScoringEngine weight review (§4.3) draws its coefficients if the
two disagree. Needs user sign-off (open decision O2, end of doc).

---

## 6. Integration design

### 6.1 RealNAVStore — replacing MockMarketDataStore

`mf_agent_orchestrator.py` already anticipates this: agents talk to the store through the
narrow interface bound in `_bind_store_interface()` (resolve, fund, snapshots,
stock_prices, sector_index, benchmark_series, sibling_equity_schemes, news), and
`MasterOrchestrator.__init__(store=...)` accepts any store. Plan (this also clears the
existing "pipeline not wired to real data" memory item):

- New module `mf_real_store.py` (new file, Phase-B build): class `RealNAVStore`
  implementing that interface from `mf_cache/`:
  - `resolve` → `universe_manifest.csv` + `AMFIRegistryLive.resolve` fallback;
  - `fund` → manifest row + cleaned NAV (`MFAPIAdapter.nav_series`, which already applies
    NAVCleaner);
  - `benchmark_series(key)` → the **peer/sector composites of §1.3/§2.2** (equal-weight
    peer NAV indices) — these are the "created benchmarks";
  - `sector_index(sector, start)` → the thematic sector composite;
  - `snapshots` / `stock_prices` → return empty-with-flag until a disclosures feed exists
    (Agent B then degrades exactly as the existing confidence-capping logic in
    `RecommendationEngine` already handles: `readiness()` reports "Historical disclosures
    MISSING" and confidence caps at 0.45);
  - `news` → existing `AnthropicNewsAgent` or neutral.
- Orchestrator invocation becomes `MasterOrchestrator(store=RealNAVStore())`; the mock
  remains for tests.

### 6.2 Exposing the model: calibrated P(target)

- Fitted artifacts (feature scaler medians/stds, coefficients, calibration map) serialized
  to `mf_cache/phase_b_model.json` — plain JSON of coefficients, deliberately **not**
  pickle, so the scorer has no hard sklearn dependency at inference time and the weights
  are human-readable (interpretability doubles as the review artifact).
- New agent-style component `TargetProbabilityScorer` ("Agent E"): input = fund code +
  as-of date; computes §2 features from RealNAVStore (same code path as training —
  single-source feature definitions to prevent train/serve skew); output =
  `P(good 3y window)`, the per-factor contribution breakdown
  (`coef × standardized feature value`), and the model-version tag.
- `RecommendationEngine.run` gains an optional `target_probability` input surfaced in the
  report (pros/cons line + component). **Whether P(target) enters the composite formula
  (and at what weight) is a user decision (O3)** — default design: display-only in v1,
  composite-weighted after one review cycle.

### 6.3 CapBandAdapter → true-to-label verifier (live checks become real)

`TrueToLabelVerifier.verify` already consumes `cap_band` per holding. Wiring: when a
current-holdings snapshot is supplied (manual CSV drop-in per `bootstrap.py` [5/5], or a
future disclosures adapter), map each holding through `CapBandAdapter.band_lookup()`
(5,427 symbols already cached in `amfi_cap_classification.csv`) to populate `cap_band`,
then the SEBI 80%/65% fidelity checks run on real data instead of being decorative.

### 6.4 Live-scoring layer — CURRENT holdings only (CONFIRMED scope)

A separate, clearly-labeled overlay on top of P(target), never fed to the backtested
model:

- cap-band mix (large/mid/small %) via CapBandAdapter → SEBI fidelity + style check
  against the fund's category;
- top-sector concentration → reuses the existing unhedged-thematic flag logic in
  `ProfileRiskScorerAgent`;
- output shape: `{"model_p_target": 0.71, "holdings_overlay": {...flags...},
  "overlay_disclaimer": "current holdings — not part of the backtested signal"}`.

---

## 7. Leakage / robustness audit (enumerated risks → exact mitigations)

| # | Risk | Mitigation |
|---|---|---|
| 7.1 | **Survivorship bias**: the 136 funds are today's survivors; dead/merged funds absent. Base rate and peer medians biased up; model never sees the failure mode "fund disappears." | Cannot be fixed with data on disk — **disclosed on every report**. Directional analysis: inflates condB base rate; makes condA *harder* (stronger peer median), so excess-based findings are conservative. Task 11 quantifies sensitivity by re-running base rates on the ≥ 10y-history subset vs full universe. |
| 7.2 | **Overlapping-window label leakage** (measured 0.935 adjacent-label agreement). | Purged + embargoed CPCV (§3.1–3.2); no random splits anywhere, including hyperparameter search and calibration; effective-n reported next to nominal n. |
| 7.3 | **Benchmark built with future data**: (a) peer forward returns are contemporaneous with the label — legitimate by construction (§1.5); (b) peer *membership* from today's manifest — survivorship, see 7.1; (c) trailing peer composites in features must use only NAV ≤ t — enforced by the same truncation as all features. | (c) covered by the as-of test harness (7.4); (a)/(b) documented. |
| 7.4 | **Feature as-of violations** (accidental use of post-t data, scaler fitted on full panel, imputation with pooled medians). | Feature engine API takes `nav.loc[:t]` only; unit test mutates all post-t NAVs by ±20% and asserts features unchanged; scalers/imputers fit inside each training fold only. |
| 7.5 | **NAVCleaner micro-lookahead**: spike removal inspects the next observation (reversion test). | One-observation lookahead on data-error detection only; at monthly anchors with 3y windows the effect is nil; documented rather than re-engineered. |
| 7.6 | **Calibration leakage**: isotonic fitted on data overlapping test windows. | Calibration slice carved from the training side with its own purge gap vs test (§4.1). |
| 7.7 | **Model-selection leakage**: many CPCV runs during development quietly overfit the blocks. | Causal holdout (anchors 2022-08…2023-07) evaluated exactly once, after all design freezes; result reported regardless of outcome. |
| 7.8 | **Regime memorization**: raw return levels let the model learn "2020 was good." | Cohort-z features (§2.4), within-anchor rank metrics as headline (§5.1), condA companion model (§5.4). |
| 7.9 | **Peer-pool cross-contamination in CV**: fund i's features include peer composites containing test-fold funds. | Accepted by design: splits are on *time*, not funds (all funds share every period); peer composites at time t use only ≤ t data, so no future information crosses. Documented so nobody "fixes" it into a worse design. |
| 7.10 | **Duplicated share classes / near-identical funds** inflating agreement. | Manifest is one Direct-Growth plan per scheme (verified 136 unique codes); no action needed. |

---

## 8. Honest limitations & risks

1. **This is a NAV-only signal.** No TER history, no AUM history, no manager tenure, no
   point-in-time holdings. Published research on fund persistence with return-only
   features finds weak (though real) predictability; a within-anchor rank-AUC in the high
   0.50s would be a *normal* result, not a failure of implementation. If the user needs
   90%+ headline accuracy, only the trivial base-rate accuracy can deliver it, and it is
   worthless — this is stated now, before a line of model code exists.
2. **~475 effective samples** bound everything: no deep feature interactions, no per-sector
   models, wide confidence intervals (± ~0.04–0.06 on AUC via block bootstrap). The
   interpretable-first decision is not just philosophy; it is the statistically correct
   capacity for this dataset.
3. **Survivorship + selection**: universe = 136 hand-picked surviving funds. Scores
   generalize to "funds like these," not to the full AMFI universe, and especially not to
   NFOs (the existing `EligibilityGate` continues to refuse those).
4. **Small-sector benchmarks**: Defence and ESG (1 fund each) have no true sector peer;
   Quant/Housing/Exports/Commodities/Business-Cycle (2 each) have one. Their condA is
   measured against a pooled thematic median (§1.3) — flagged per-sample, excludable in
   sensitivity runs.
5. **History left-censoring at 2013** (direct-plan inception): trailing-5y features exist
   for only 51% of samples; the negative-rich 2015–2017 anchors have the *least* feature
   history. Missingness indicators partially compensate; the imbalance is irreducible.
6. **Regime dependence of the label itself**: the OR-label mostly measures "did the next
   3 years have a decent equity market" (condB) plus a thinner skill margin (condA adds
   3.6pp). Reports will always decompose predictions into the two legs so nobody mistakes
   regime luck for fund selection.
7. **Category drift over 13 years**: funds recategorized by SEBI 2017/2021/2026 waves are
   mapped to today's category for their whole history. Not fixable from NAV data;
   disclosed.
8. **Calibration drift live**: probabilities calibrated on 2014–2023 anchor cohorts;
   applied at 2026 anchors they inherit any regime shift. The live scorer reports its
   training-coverage window with every score.

---

## 9. Implementation task breakdown (numbered, tiered)

Model tiers: **[Fable]** = design-sensitive modeling/validation correctness;
**[Sonnet]** = mechanical data plumbing/wiring per this spec. Per global rule G2, each
task lands as its own reviewed local commit.

| # | Task | Tier | Notes |
|---|---|---|---|
| 1 | `mf_real_store.py`: RealNAVStore over mf_cache (manifest loader, cleaned-NAV panel via MFAPIAdapter/NAVCleaner, as-of NAV lookup with the 10-day staleness rule, monthly anchor grid util) | Sonnet | pure plumbing against §6.1 interface list |
| 2 | Label builder: forward returns, LOO peer/sector benchmarks per §1.3, label table → `mf_cache/phase_b_labels.parquet` (columns: code, anchor, fwd3y, bench, condA, condB, y, peer_key, peer_n, benchmark_quality) | Sonnet | must reproduce §0 numbers as its acceptance test (base rate 0.842 ± 0.002) |
| 3 | Feature engine per §2 with truncation-at-t API + the anti-lookahead unit test (perturb post-t NAV ⇒ identical features) | **Fable** | as-of correctness is the whole game |
| 4 | Purged/embargoed CPCV harness per §3 (block definitions, purge logic, causal-holdout lockout) + tests on synthetic data with known leakage | **Fable** | subtle; test: a deliberately leaky feature must show inflated random-split AUC but not survive CPCV |
| 5 | Primary model: elastic-net logistic + fold-safe standardization/imputation + purged calibration (§4.1) | **Fable** | |
| 6 | Challenger HistGB + the > 3-AUC-point comparison gate (§4.2) | Sonnet | mechanical once the harness (4) exists |
| 7 | Metrics/report module per §5 (per-block tables incl. base-rate column, effective-n, within-anchor rank-AUC, negative-capture, reliability curve; in-sample quarantined section) | Sonnet | presentation rules of §5.3 are requirements, not suggestions |
| 8 | Factor-weight review: coefficient → FACTOR_MAP mapping table + block-bootstrap sign-stability (§4.3, §3.4); deliverable doc for user sign-off | **Fable** | this is deferred Step 2 |
| 9 | TargetProbabilityScorer ("Agent E") + JSON model artifact + orchestrator wiring (`store=RealNAVStore()`, display-only P(target) in RecommendationEngine output) | Sonnet | composite-formula change deferred to O3 |
| 10 | Live-scoring overlay: current holdings CSV → CapBandAdapter cap-mix + sector concentration + true-to-label wiring (§6.3–6.4) | Sonnet | |
| 11 | Leakage-audit rerun + survivorship sensitivity (7.1) + one-shot causal-holdout evaluation + final honest report | **Fable** | evaluated once, reported verbatim |
| 12 | `requirements.txt`: add `scikit-learn>=1.7` (NOT currently installed in `.venv` — verified missing; check py3.14 wheel availability at install time); optionally `matplotlib` for reliability curves | Sonnet | venv rule: install with `.\.venv\Scripts\python.exe -m pip` |

Suggested order: 1 → 2 → 3 → 4 → 5 → 7 → 6 → 8 → 11 → 9 → 10 → 12 (12 actually first,
since 4–6 need sklearn).

---

## Appendix A — Probe provenance

Probe scripts (read-only, scratch dir, not committed): monthly-anchor forward-return and
base-rate probe over all 136 cached mfapi series with NAVCleaner-pass-1 cleaning inline;
follow-up probe for per-year condition rates, CPCV block balance, thematic pool sizes and
trailing-history availability. Key intermediate: 15,037 samples, LOO peer benchmarks for
98.3%, peer-pool median size 9. Numbers in §0 are direct script output, unrounded beyond
3 decimals.
