# App integration plan — pipeline → Hisaab Kitaab

The Python in this repo **does not run in the app**. The handoff is a batch artifact:
one versioned gzipped JSON per nightly run, which the Flutter app downloads and loads
into Drift/SQLite. This document is the end-to-end plan. Two pieces already have their
own specs and are not repeated here:

- **Fetching it** — `docs/ci.md`, *How the app consumes the artifact*.
- **The model-health panel** — `docs/app_evaluation_contract.md`, the `evaluation{}`
  block and the disclaimer the UI must show.

## Status: one decision blocks everything

The nightly publishes to a rolling `latest-artifact` GitHub Release, so the URL no
longer moves. **But this repository is private, and a private repo's release assets
are not anonymously downloadable** — the app cannot fetch it, and shipping a token
inside a mobile binary is not an option. Until this is resolved (make the repo public,
mirror to a public artifacts repo, or move to object storage — see `docs/ci.md`),
every step below is blocked at step 1. Nothing else about the integration is unknown.

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
