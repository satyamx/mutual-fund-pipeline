# What this is, in plain terms

A companion for anyone — including future-you — deciding how much weight to put on
this pipeline's output. No jargon, no code. If you read one document here, read this
one, because the honest version of what this tool does is narrower than what a screen
full of green ticks looks like it's saying.

## The one-sentence version

It reads the daily published price of ~367 Indian mutual funds, computes some
well-understood statistics about each one, checks a few regulatory rules, and sorts
them into 🟢 / 🔵 / 🔴 — **and that is a measurement, not a prediction that you will
make money.**

## What it can actually see

Exactly one thing: **NAV** — the daily per-unit price every mutual fund publishes.
From that it derives returns, volatility, drawdowns, consistency, and performance
against a benchmark or peer group.

That's a real and useful input. It is also a *shadow* of the fund. NAV is the
outcome of a manager's decisions; it is not the decisions. Two funds with identical
five-year NAV curves can be run completely differently, and NAV cannot tell them
apart.

## What it cannot see, and does not pretend to

- **What the fund actually owns.** Holdings data is published monthly by each AMC as
  a spreadsheet, and this pipeline has none on file. Where you see concentration
  greyed out or "NOT EVALUATED", that is the truth being reported, not a pass.
- **Whether the manager is any good.** There is no manager-history data on file
  either. A 🟢 verdict carries an explicit "manager skill and holdings unverified"
  caveat for exactly this reason.
- **Fees, in most cases.** Expense ratios are not reliably free to fetch.
- **Anything about the future.** See below — this is the important one.

The rule the whole project is built around: **missing information is never treated as
good news.** Absence of evidence shows up as a coverage flag. It never becomes a pass.

## The prediction part, stated honestly

There is one genuine statistical model here. It asks a narrow question: *within a
group of similar funds, will this one land in the top quarter over the next three
years?*

Measured on data it had never seen, it scores **AUC ≈ 0.558**. Plain reading: a
coin flip is 0.500 and perfect foresight is 1.000. It is **barely better than
chance** — real, but weak. Its lift is about **1.10x**, meaning it finds top-quartile
funds roughly 10% more often than picking at random would.

Two things follow, and both are honoured in the output:

1. It is shown as a *supporting datapoint*, never folded into the verdict.
2. **Nothing has been verified as correct yet.** The model predicts a three-year
   outcome, and the first predictions were logged in 2026 — so the earliest date any
   of them can be graded is **around 2029**. Until then the app shows `PENDING`,
   which means *not yet measurable*, not *good*.

Everything else that was tried — the older weighted "composite score" — measured at or
below chance and was deleted rather than displayed.

## How the verdict is decided

Deliberately boring, and that is the point. Each metric gets a colour against a fixed
threshold (e.g. 3-year CAGR ≥10% is green, ≤6% is red). Then:

- 🔴 **SELL** — any hard regulatory breach, **or** 3+ red metrics, **or** more reds
  than greens with a weak overall screen.
- 🟢 **BUY** — 3+ greens, zero reds, and the overall screen isn't weak.
- 🔵 **HOLD** — everything else, including *"we don't know enough"*.

No weights, no black box. You can always ask "why did it say that?" and get a list of
which metrics were which colour. HOLD is the default when evidence is thin.

## Four things to be genuinely sceptical about

1. **The verdict is permissive.** On the 2026-08-05 batch it returned **288 BUY, 70
   HOLD, 9 SELL** across 367 funds — roughly 78% BUY. A screen that approves of most
   of the market is not narrowing your choices very much. The thresholds are tunable
   defaults that have not been tuned against outcomes.
2. **A rising market makes everything look good.** These are absolute thresholds. In a
   market where most equity funds returned >10%, "≥10% CAGR is green" flatters
   nearly everyone. The benchmark-relative metric partly offsets this; it does not
   eliminate it.
3. **Survivorship.** The universe is funds that exist *today*. Funds that did badly
   and were merged away are not in it, which quietly makes the surviving set look
   better than the historical reality.
4. **Past performance genuinely does not imply future returns.** This is a regulatory
   boilerplate line that happens to be the single most accurate sentence about what a
   NAV-only tool can tell you.

## What it is legitimately good for

- Filtering out funds with objectively poor risk-adjusted history.
- Catching hard compliance breaches (e.g. a single issuer above the SEBI 10%-of-NAV
  limit) that force a 🔴 regardless of returns.
- Comparing a fund against genuinely similar peers rather than against the whole market.
- Being explicit about what it doesn't know, which most retail-facing fund screens are not.

## What it is not

A buy engine. A robo-adviser. A substitute for reading the scheme document. And not
financial advice — it is a measurement tool whose author is also its only user.
