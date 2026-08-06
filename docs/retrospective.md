# Build retrospective

What this project actually learned, kept because the lessons generalise and because
several of them cost real sessions to relearn. `STATUS.md` records *what happened*;
this records *what it means*.

## The theme: every serious bug here was a silent success

Not one of the significant defects announced itself. Every single one produced a
green run, a plausible number, and no error. Ranked by how long it went unnoticed:

| Defect | What it looked like | What it actually was |
|---|---|---|
| **Frozen clock** | Nightly green for weeks, `rows_appended: 0` | `TODAY` hardcoded to a past date, so every scoring anchor was pinned there. The prediction ledger — whose entire purpose is to accumulate now and mature in ~2029 — was structurally unable to grow. Zero-appended reads exactly like ordinary idempotence. |
| **Resolver collision** | `n_total=136, n_ok=136, n_errors=0` | Token-subset name matching resolved 3 funds onto *longer* scheme names. 136 names produced 133 distinct evaluations: 3 funds scored twice, 3 never scored, and the coverage count reported perfection. |
| **Cold benchmarks** | Run "succeeded" | A missing directory raised, discarding 118 of 136 otherwise-complete evaluations. Nothing returned non-zero. |
| **Calibration floor** | Passed the new guard | The *fix* for a zero-probability bug was itself an overclaim: it floored at 0.0004 using the rule of three over the whole calibration slice, while the flattened region held 6 samples whose honest bound was 0.50 — three orders of magnitude tighter than its own evidence. |
| **Sector proposer** | 118 clean-looking proposals | A bare `services` pattern also matched "Banking and Financial **Services**", blanking 24 of 25 financials funds — the largest clean group — as a "broad services theme". |

**The generalised lesson: a green exit code is not a green run.** Every gate this
repo has was added after a failure that a boolean pass/fail could not have caught.
The gates that work all assert something *positive and specific*:

- `--max-error-rate 0.25` — a batch that drops most of the universe is failed, not thin.
- `--max-anchor-age-days 7` — keyed on the anchor drifting behind the clock, deliberately
  **not** on "zero rows appended", because zero is correct and common on a re-run.
- Exhaustive status→flag mapping whose coverage is selftest-asserted, because an
  if/elif chain is *how* the flags silently drifted apart from their meanings.

## Honesty is a design constraint, not a disclaimer

The load-bearing decisions all cost something measurable, which is what made them real:

- **The composite was deleted, not demoted.** It measured at or below chance. Keeping
  it as a smaller number would have been laundering.
- **The retrain reported a *worse* headline and shipped anyway.** Widening 136 → 367
  funds moved AUC 0.578 → 0.558. Reusing the old number would have been the actual
  dishonesty: it described a different label on a different population. The holdout
  was re-forced exactly once, with the outcome committed to in advance.
- **The obvious calibration repair was measured, then rejected in public.** Flooring
  at the honest bound pinned 98% of predictions at the floor and collapsed AUC to
  0.514. It is written into the docs so nobody re-derives it.
- **`realized_ic` stayed null rather than becoming a shorter-horizon proxy.** The model
  was never validated at any horizon but three years, so a 1-year "interim" number
  would have been a new, unvalidated claim wearing a validated name.

The pattern: **the honest option was usually the one that made the numbers look
worse, and it was chosen every time.** That is only visible in retrospect as a
discipline; in the moment each one just felt like losing something.

## Three structural traps, each hit more than once

1. **Unbackfillable work inside a regenerable directory.** `mf_cache/` is gitignored
   *and* restored in CI from an evictable cache. The universe manifest, the
   benchmarks, the shipped model and the prediction ledger were each found in there
   and moved out — four separate instances of one mistake. The test is not "is this
   expensive to recompute" but **"can any script in this repo rebuild it at all?"**
2. **CI writes to the repo.** The nightly commits the appended ledger back to
   `master`, so `origin` moves with nobody pushing. A local commit on a stale base
   turns into a conflict in an append-only file that must never be resolved by
   picking a side. This cost two sessions before it was written down.
3. **Stale notes are worse than no notes.** A "no `gh` CLI on this box" claim survived
   long after it stopped being true, and it is why a CI failure went undiagnosed and
   a review tool went unused. Notes that assert an environment fact need a date on
   them and a way to be falsified.

## What would be done differently

- **Build the gate with the feature.** Every gate here is a scar. `--max-anchor-age-days`
  would have caught the frozen clock on day one; it was written weeks after.
- **Run against real data sooner.** The sector proposer's 24-fund bug was invisible in
  a hand-written selftest and obvious in the first real run. Synthetic tests confirm
  the logic you thought of; real data finds the pattern you didn't.
- **Distinguish "unmeasured" from "measured as fine" in the schema, early.** The
  PENDING-vs-GREEN distinction had to be retrofitted into the evaluation panel, and it
  is the single most important thing that panel communicates.
- **Write down the decision *and its cost* at the time.** The entries that aged well
  are the ones recording what was given up ("this collapses AUC to 0.514"). The ones
  that aged badly just say what was chosen.

## The open question this project has not answered

Whether any of it works. The verdict rule has never been evaluated against outcomes —
it *can't* be until predictions mature around 2029 — and the one validated signal is
weak enough (AUC 0.558, lift 1.10x) that its main honest use is ranking within a peer
group, not deciding anything. The infrastructure to find out is now in place and
running nightly: an append-only ledger, a frozen model stamped on every row, and a
monthly realization job that will grade them.

That is the actual deliverable. Not a screen — **a claim that will be checked.**
