# App integration plan — pipeline → Hisaab Kitaab

The Python in this repo **does not run in the app**. The handoff is a batch artifact:
one versioned gzipped JSON per nightly run, which the Flutter app downloads and loads
into Drift/SQLite. This document is the end-to-end plan. Two pieces already have their
own specs and are not repeated here:

- **Fetching it** — `docs/ci.md`, *How the app consumes the artifact*.
- **The model-health panel** — `docs/app_evaluation_contract.md`, the `evaluation{}`
  block and the disclaimer the UI must show.

## Status: unblocked — the artifact is publicly fetchable

The nightly publishes to a rolling `latest-artifact` GitHub Release, so the URL no
longer moves, and **as of 2026-08-06 the repository is public, so that URL serves
anonymously** — verified by unauthenticated `curl` returning HTTP 200 with the real
payload. The decision that used to block step 1 is settled (option 1 in `docs/ci.md`);
no token ships in the app, which was the constraint that ruled out the alternatives.

**Step 1 is now buildable.** The remaining open call is D4 below (milestone 6): whether
the app re-derives a profile-specific verdict in Dart — where it can drift from the
Python — or renders the shipped facts + colours. Settle it before step 3 hardens.

## Sequence

**1. Fetch.** Poll `latest.json` (a few hundred bytes) and compare `generated_at`
against what's stored. Download `mf_artifact_latest.json.gz` only when it has moved.
Verify the payload's `sha256` against the sidecar.

**2. Validate before writing.** Reject the payload rather than half-load it if
`artifact_version` is not one the app knows, or `coverage.n_ok` is zero. A stale
local copy beats a corrupt one — the fetch is a cache refresh, not a source of truth
the app must accept.

**3. Load into Drift.** One row per fund keyed on `amfi_code`. Replace wholesale
inside a transaction; the artifact is a complete snapshot, not a delta. Keep the
previous snapshot until the new one commits.

**4. Render.** See the UI contract below.

## Schema mapping (per fund record → app)

Per-fund top-level keys are `amfi_code, isin, scheme_name, category, sector,
eligibility, facts, signal_a, alerts_b, nfo_dossier, coverage_flags, data_flags`.
Note the verdict lives **inside `signal_a`**, not at the top level, and the alerts key
is **`alerts_b`** (System B), not `alerts`.

| Artifact field | App use | Non-negotiable handling |
|---|---|---|
| `signal_a.verdict`, `.verdict_color` | the 🟢/🔵/🔴 chip | Must be shown with `signal_a.verdict_caveat` attached — a BUY carries "manager skill & holdings unverified". |
| `signal_a.verdict_basis.is_default_profile` | disclaimer banner | **When true, the app MUST say the verdict assumes a default profile** (horizon 7.5y, no liquidity need, high risk) — the batch runs before the app knows the real user. |
| `facts.*` | the metric rows | `null` means unknown. Render grey/"not available". Never render a null as 0, "—0%", or a neutral-looking pass. |
| `signal_a.metric_colors.*` | per-metric dot | Includes `concentration`, which is deliberately **not** counted toward the verdict — a concentrated book is a legitimate style, not a defect. Show it; don't let it imply the verdict. |
| `signal_a.cohort_q1_prob`, `.cohort_percentile`, `.cohort_n` | the weak supporting signal | Label as weak and never as "accuracy" or "confidence". Show `signal_context.holdout_auc` alongside it. |
| `signal_a.cohort_status` | why the probability is null | Branch on this, not on the null itself. |
| `coverage_flags[]` | "what we don't know" list | Human-readable gaps. These are the honesty surface — showing them is what separates this from a screen that hides its gaps. |
| `data_flags[]` | machine-branchable gap codes | The stable enum (`THIN_COHORT`, `STALE_NAV`, `SECTOR_UNRESOLVED`, …). Branch on these; render `coverage_flags`. |
| `alerts_b[]` | Sentinel alerts | Each has a stable `code`, a severity, and an `AlertBasis`. **Only `REGULATORY` may reach a push notification** — `_NEVER_HIGH` bars everything else structurally, and the app must not re-escalate. |
| `eligibility` | NFO/newborn gating | A young fund's factor rules are dormant; don't render dormancy as a pass. |
| `nfo_dossier` | new-fund detail | Present for NFOs; null otherwise. |
| `evaluation{}` (top level) | model-health panel | Per `app_evaluation_contract.md`. `PENDING` ≠ bad. |
| `signal_a.verdict_branches{score_red, score_not_red}` | the personalised verdict | The two possible verdicts. Pick with the formula below. **Do not reimplement the verdict rule** — that is the whole point of shipping both. |
| `signal_a.screen_score_red_below` | the cutoff | Read it; **never hardcode 45 in Dart**. If the threshold is retuned server-side, a hardcoded copy silently diverges. |
| `signal_a.weight_matrix`, `.utility_score` | drift self-check | The **default-profile** values. Use them to verify your ported weight function on every record (see below), not to score the user. |
| `signal_a.n_imputed`, `.imputed_fraction`, `.imputed_features[]` | how much of the score is real | **`null` means "not measured", NOT "nothing imputed"** — a refused fund never had imputation computed. Only ever compare these when `cohort_status == "OK"`. A fund at `imputed_fraction` 0.6 is mostly training-median; grade confidence down, don't render it beside a 0.05 fund as equivalent. |
| `signal_a.cohort_status == "INSUFFICIENT_HISTORY"` | under 1y of NAV | A permanent-for-now refusal: ~77% of the vector would be training median, so no probability is emitted. Distinct from `THIN_COHORT` (the *cohort* is too small) — this one is about *this fund's* history. It resolves itself as the fund ages. |

### Personalising the verdict (D4 option C — `docs/d4_profile_verdicts.md`)

The investor profile reaches the verdict rule through exactly one term, so the app
needs to port **only the weight function**, not the rule:

```dart
final w = utilityWeights(userProfile);                  // ~20 lines, ported from
                                                        // ProfileRiskScorerAgent.utility_weights
final u = 100 * sum(w[k] * signalA.subScores[k]);       // sub_scores are profile-agnostic
final b = u < signalA.screenScoreRedBelow ? 'score_red' : 'score_not_red';
final verdict = signalA.verdictBranches[b];             // verdict + color + caveat
```

**The null case is not optional.** A NEWBORN/YOUNG fund has `null` `sub_scores` and a
`null` `utility_score` (Agent C runs with no eligibility gate, so its scores are
genuinely undefined, not zero). Dart arithmetic on `null` throws, and coercing to 0
would fabricate a utility of 0 — below the cutoff, so it would manufacture a
`score_red` verdict out of missing data, which is a direct honesty-invariant breach.

> **If `utility_score` is null, or ANY `sub_scores` value is null: do not
> personalise. Render the shipped `signal_a.verdict` with its default-profile
> disclaimer.** Python degrades the same way — a NaN utility bands as "blue", never
> "red" — so this fallback matches server behaviour rather than inventing a rule.

**Verify the port on every record rather than trusting it.** When
`verdict_basis.is_default_profile` is true, `utilityWeights` applied to
`verdict_basis.profile` must reproduce the shipped `weight_matrix`, and the formula
must reproduce `utility_score` **to within ±0.15**. If either disagrees, the Dart port
has drifted from the Python — fail loudly or fall back to the shipped
`signal_a.verdict`; do not show a number you cannot reproduce. `mf_artifact.py
--selftest` asserts both server-side at the same tolerance.

That tolerance is rounding, not slack: `sub_scores` ship at 3dp and `utility_score` at
1dp, which bounds the residual near 0.1. `weight_matrix` deliberately ships at **6dp**
for this reason — at 3dp the error reached ~0.3, enough to put a fund near the cutoff
on the wrong branch.

Three things this does **not** change:

- The weights are a **normalised distribution** — a port that drops the final `/total`
  silently rescales every utility.
- **A fund within ~0.1 of the cutoff is genuinely ambiguous.** The comparison is a hard
  threshold against a rounded number; don't present a borderline verdict as more
  precise than it is.
- The cutoff is a tunable default with **no out-of-sample validation**. Personalising
  which side of it a fund lands on does not make the verdict more correct.

### Two cohort statuses that mean opposite things

`NOT_IN_UNIVERSE` is a **fixable** gap (fetch the NAV, add to the manifest) —
reasonable to show as "not yet scored". `OUT_OF_TRAINING_UNIVERSE` is a **permanent
refusal**: the model was never fitted on that category and a probability there would
claim the holdout AUC transfers to a population it never saw. These demand opposite
copy, and the app must not collapse them into one "unavailable" state.

## UI requirements that are not negotiable

These are not styling preferences; each one exists because the alternative
misrepresents what the pipeline knows.

1. **Missing data renders as missing.** Never as zero, never as neutral-pass.
2. **The default-profile disclaimer appears wherever a batch verdict does.**
3. **`PENDING` is visually distinct from `GREEN`.** Not-yet-measurable is not good news.
4. **The model-health panel is never labelled "accuracy", "win rate", or "returns".**
   It reflects drift and pipeline health. Whether predictions are *right* is
   unmeasurable until ~2029.
5. **The weak signal is labelled weak, next to the number.** AUC 0.558 / lift 1.10x.
6. **No verdict without its caveat.**

## Milestones

| # | Deliverable | Blocked on |
|---|---|---|
| 0 | Resolve artifact access (public / mirror / object storage) | **owner decision — blocks all of the below** |
| 1 | Fetch + verify + Drift load, with the stale-beats-corrupt rule | 0 |
| 2 | Fund list + detail screen: verdict chip, caveat, metric rows with grey nulls | 1 |
| 3 | Coverage-flags surface and the two distinct cohort-status states | 2 |
| 4 | Model-health panel per `app_evaluation_contract.md` | 2 |
| 5 | Alerts, with `REGULATORY`-only notifications | 2 |
| 6 | Real user profile replaces the default, and the disclaimer drops | 2 |

Milestone 6 is worth calling out: today every verdict is computed against
`InvestorProfile()`'s defaults because the batch runs before the app knows the user.
Re-deriving a profile-specific verdict on-device means reimplementing the verdict
rule in Dart — at which point the rule exists in two languages and can drift. The
alternative is to ship the *facts* and *colours* and let the app apply the rule; that
keeps one source of truth for thresholds. **This is an open design decision, not a
settled one**, and it should be settled before milestone 2 hardens.

## Mirror decisions into the app repo

Anything decided here that changes app behaviour goes into Hisaab Kitaab's
`DECISIONS.md` — the integration contract (`f99fa96`) and the verdict amendment
(`9885565`) are already there. Two repos with one contract drift unless the decision
log is single-sourced.
