# D4 — profile-specific verdicts: where the rule should live

**Status: DECIDED and IMPLEMENTED (2026-08-06) — option C.** The owner chose C; the
server side ships. `RecommendationEngine.run` now emits `verdict_branches` +
`screen_score_red_below`, and `mf_artifact.py` ships those alongside `weight_matrix`
and `utility_score`. What remains is the app-side port of `utilityWeights` — the client
contract and its self-check are in `docs/integration_plan.md`.

This document exists because the framing in `docs/STATUS.md` ("re-deriving a verdict
on-device means reimplementing the verdict rule in Dart, where it can drift from the
Python") turned out to overstate the problem. Reading `RecommendationEngine.run` closed
most of the question before any trade-off had to be made.

Everything below was verified against the code, not recalled. The relevant sources are
`mf_agent_orchestrator.py` (`ProfileRiskScorerAgent.utility_weights`, ~L847; `.run`, ~L876;
`RecommendationEngine.run`, ~L997) and `mf_artifact.py` (`build_fund_record`, ~L204).

## The finding that resizes the decision

**The verdict rule is almost entirely profile-independent.** Its inputs are:

| Verdict input | Source | Profile-dependent? |
|---|---|---|
| `critical_breach` | SEBI critical findings + Agent D regulatory red flags | **No** |
| `greens` / `reds` over the 5 core metrics | `facts` (cagr, excess_cagr_3y, sortino_3y, max_dd_3y) + `sub_scores["consistency"]` | **No** |
| `score_band` | `utility = 100 · Σ wₖ(profile) · sₖ` | **Yes — the only one** |

All six `sub_scores` are computed from NAV and holdings alone. The profile enters the
entire system at exactly one point: as a **weight vector** in `utility_weights(p)`. The
scores it weighs never move.

**And `score_band` only matters as a single boolean.** In the rule —

```python
if critical_breach or reds >= 3 or (reds > greens and score_band == "red"):
    SELL
elif greens >= 3 and reds == 0 and score_band != "red":
    BUY
else:
    HOLD
```

— `score_band` appears only as `== "red"` / `!= "red"`, i.e. `utility < 45`. The green
threshold (`utility >= 65`) is **display only; it never reaches the verdict.**

Two consequences fall straight out:

1. What a Dart port would need to reimplement is **not the verdict rule**. It is
   `utility_weights()` — about 20 lines of pure branch arithmetic with no data
   dependencies — plus a 6-term dot product and one comparison against 45.
2. With `greens`, `reds` and `critical_breach` fixed, the verdict is a function of one
   boolean. **Each fund therefore has exactly two possible verdicts**, and Python can
   compute both.

## What the artifact ships today

| Field | Ships? | Needed to personalize |
|---|---|---|
| `signal_a.sub_scores` (6 values) | ✅ | yes |
| `signal_a.metric_colors` (incl. `screen_score` band) | ✅ | display |
| `signal_a.verdict` + `verdict_basis.is_default_profile` | ✅ | fallback |
| `weight_matrix` (the profile's weights) | ❌ | no — but see drift check |
| `utility_score` (the numeric screen score) | ❌ | no — only its band ships |
| `greens` / `reds` / `critical_breach` counts | ❌ | yes, under option C |

## The options

### A — ship the default verdict only (status quo)
App renders the shipped verdict with the "default profile assumed" disclaimer, forever.

- **Zero drift risk**, zero new code.
- **The profile feature is dead.** A conservative investor on a 2-year horizon reads a
  verdict computed for a high-risk 7.5-year one. `InvestorProfile`, `utility_weights` and
  the whole Agent C weight surface become decorative — and the disclaimer becomes
  permanent rather than transitional, which is not what it was designed for
  (`mf_artifact.py` L18-23 describes it as a state the app leaves once the user
  personalizes).

### B — app re-derives the whole verdict in Dart
Port `_band` and its seven thresholds, the greens/reds tally, the compliance colour
logic, `critical_breach`, and `utility_weights`.

- Maximum flexibility.
- **Largest possible drift surface, and most of it is drift you cannot detect.** Worse,
  parts of it are not the app's judgement to make: whether a SEBI finding is `critical`,
  or whether a news item is a regulatory red flag, is an upstream determination. An app
  that re-decides it can silently disagree with the ledger about the same fund on the
  same day.

### C — ship the rule's output space; port only the weight function ✅ recommended
Python keeps the verdict rule. Per fund it additionally ships:

- `utility_score` and `weight_matrix` (for the default profile),
- `verdict_if_score_red` and `verdict_if_score_not_red` — the two possible verdicts,
  with their caveats, both computed by the existing Python rule.

The app then does only this:

```
w = utilityWeights(userProfile)        // the one ported function, ~20 lines
u = 100 * Σ w[k] * subScores[k]        // shipped, profile-agnostic
verdict = (u < 45) ? verdictIfScoreRed : verdictIfScoreNotRed
```

- **The verdict rule never leaves Python.** Rule drift is impossible by construction —
  not "unlikely", impossible, because the rule is not expressed in Dart at all.
- **The remaining port is self-checking.** For the default profile the app can recompute
  `w` and assert it matches the shipped `weight_matrix`, and recompute `u` against the
  shipped `utility_score`. A divergent Dart port is then caught **at runtime on every
  fund record**, not by a two-language test suite somebody has to maintain. This is what
  makes C strictly better than B rather than merely smaller: it converts silent
  divergence into a loud, automatic check.
- **Cost:** a small, real change to `mf_artifact.py` (four fields) and a widened contract
  in `docs/integration_plan.md`. Note this contradicts "nothing buildable is left" — C is
  buildable work, roughly an afternoon.

## Recommendation

**Take C.** It preserves the profile feature that A discards, at a fraction of B's risk,
and it is the only option where the thing most likely to break announces itself.

Two caveats worth stating rather than discovering later:

- **C does not make the verdict *more* correct.** The thresholds (45, and the seven
  `_band` cutoffs) remain tunable defaults that no out-of-sample test has validated —
  they are a transparent rule, not a measured one, and personalizing which side of 45 a
  fund lands on does not change that. The honesty invariant is untouched either way.
- **If the verdict rule itself is ever changed**, `verdict_if_score_*` keeps the app
  correct automatically, but any Dart-side `utility_weights` still has to track changes
  to the Python weight surface. The self-check catches it; the fix is still manual. Bump
  `artifact_version` when the weight surface changes, so a stale app fails loudly instead
  of computing a quietly wrong `u`.

**Settle this before `docs/integration_plan.md` step 3 (Drift load) hardens**, because
the answer determines whether the app stores one verdict per fund or the two-branch pair.

## Post-implementation review outcomes (2026-08-06)

Three things the code review surfaced that the design above did not anticipate. The
first two are fixed; the third is an open decision.

1. **Rounding vs a hard cutoff — fixed.** `weight_matrix` originally shipped at 3dp
   like `sub_scores`. Recomputing `100·Σ w·s` from 3dp weights carries a worst-case
   error of ~0.3 on a 0–100 scale, and the branch selection is a *hard* comparison
   against 45.0 — so a fund near the cutoff could recompute onto the opposite branch
   from the one Python picked. `weight_matrix` now ships at **6dp**, dropping the
   weights' contribution to ~3e-4 and leaving a residual near 0.1 dominated by
   `sub_scores`' own 3dp. The selftest tolerance was tightened 0.5 → **0.15** to match;
   a loose tolerance would have hidden precisely the drift the fix prevents.
   **A fund within ~0.1 of the cutoff remains genuinely ambiguous** — that is inherent
   to thresholding a rounded number, and the app is told not to present such a verdict
   as more precise than it is.

2. **The null case — fixed in the contract.** A NEWBORN/YOUNG fund has `null`
   `sub_scores` and `utility_score`. Dart arithmetic on null throws, and coercing to 0
   would fabricate a utility *below* the cutoff — manufacturing a `score_red` verdict
   out of missing data, a direct honesty-invariant breach. `docs/integration_plan.md`
   now requires the app to fall back to the shipped verdict instead, which is what
   Python already does (NaN bands "blue", never "red").

3. **A negative `growth` weight was reachable — FIXED (D5).** `conservative` +
   `liquidity=high` + `horizon < 4` yielded `growth = -0.03`: a fund with better
   realised growth scored lower. Pre-existing in `utility_weights`, but **this design
   is what exposed it** — the batch only ever computed the default profile's weights,
   so no non-default vector existed in production until the app started deriving them,
   and the perverse case is precisely the risk-averse short-horizon investor the
   personalisation feature is for.

   Resolved by **flooring each weight at 0 before normalising**: a factor may become
   irrelevant to a profile, but must never actively penalise. The order matters — a
   negative term left in place shrinks `total` and silently inflates every other
   weight, so clamping after the division would fix the sign and keep the distortion.

   Verified by enumeration rather than assumed: **exactly one combination changes**,
   the **default profile is bit-identical** (so no verdict in the shipped artifact
   moves), and all 90 sampled profiles are non-negative and sum to 1.0. Locked by a new
   `mf_artifact --selftest` block, because the batch itself exercises a single profile
   and could never have caught this.
