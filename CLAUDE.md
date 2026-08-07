# CLAUDE.md — MF Analysis

Guidance for Claude Code working in this repo. (User-global rules in `~/.claude/CLAUDE.md` also apply.)

## What this is
A **NAV-only Indian mutual-fund analysis pipeline** — a Python batch that fetches free public data, computes honest quant/compliance facts per fund, and emits a screen + recommendation. The intended consumer is the **Hisaab Kitaab** Flutter app (separate repo) via a batch-artifact handoff (versioned JSON → Drift/SQLite). Python does not run in the app.

**It is a measurement / screening tool, NOT a skill-verified buy engine.** NAV data cannot see inside a fund's holdings.

## THE HONESTY INVARIANT (read before changing any scoring/output)
This is the load-bearing constraint of the whole project:
- **Never surface fabricated accuracy or a laundered composite.** The only out-of-sample validated signal is the within-cohort `cohort_q1` model — as of `phase_b_v4` (556-fund fit, 2026-08-06) **holdout AUC 0.541, lift 1.39x, on an effective n of 135: weak.** Successive retrains have bought coverage, never skill; do not read a moving headline AUC as the model improving or degrading, because each fit is measured on a different population. `phase_b_v3` was 0.558 / 1.10x on 367 funds. Those two numbers are NOT comparable: v1 was measured on 1,288 test rows whose cohorts were too small to quartile cleanly (base rate 0.325 against a true 0.25), v2 on 3,332 with a better-posed label (0.280). Scored like-for-like on the same rows, v2 beats v1 (AUC 0.542 vs 0.535, lift 1.33x vs 1.20x) — the drop is the old estimate having been optimistic, not the model getting worse. Everything else (the old utility composite) measured at/below chance.
- **Never turn missing data into a positive verdict.** Absence of evidence is a coverage flag, never a pass. HOLD is the honest default under thin evidence; a hard compliance breach forces SELL.
- **Never invent facts** — no fabricated holdings/TER/AUM/manager. Missing = `null` / "NOT AVAILABLE".
- **Never manufacture the target** by loosening a label or leaking. Report true OOS metrics (AUC / lift / precision), not raw accuracy on an imbalanced label.
- The current user-facing verdict (🟢BUY/🔵HOLD/🔴SELL) is an **explainable RULE over colour-coded metrics + hard compliance gates**, not a weighted black box. Keep it that way.

## Architecture
Module map + data flow: `docs/architecture.md`. Orientation for a human rather than a
module map: **`docs/what_this_is.md`** (what this honestly does and does not claim),
`docs/retrospective.md` (what the build learned), `docs/integration_plan.md` (the app
handoff). Four things the code won't tell you:
- `mf_overrides.py` / `overrides/` are **INPUTS ONLY** — never write scores back into them.
- `mf_sentinel.py`'s `_NEVER_HIGH` structurally bars non-regulatory alert bases from HIGH.
- `mf_holdings.py` is **never an ML feature** (unbacktestable); it reaches the verdict only via the SEBI single-issuer compliance rule.
- The live orchestrator does **NOT** use `mf_pipeline.py`'s `ScoringEngine` — its FACTOR_MAP composite is being retired into Sentinel alert thresholds. `mf_pipeline.py` is research-only.

## Environment & running
- **Python: venv only** — `./.venv/Scripts/python.exe`. The bare `python`/`python3`/`pip` on PATH are broken Windows-Store stubs; never use them. (Base interpreter: `C:\Python314\python.exe`.)
- **Always set `PYTHONIOENCODING=utf-8`** when running anything that prints ₹ or emoji (cp1252 crash otherwise).
- Common commands:
  - `python bootstrap.py --funds "Parag Parikh Flexi Cap" ...` — fetch/refresh real data into `mf_cache/`.
  - `python mf_model.py --stage cohort` — (re)train + write the cohort model artifacts.
  - `python mf_infer.py --selftest` — verify numpy inference == sklearn (bit-exact).
  - `python mf_features.py --selftest` / `python mf_cv.py --selftest` — anti-lookahead / leakage checks.
  - `python mf_overrides.py --propose --out overrides/_sector_proposals.csv` — SUGGEST sectors for the blocked funds (review surface; never auto-applied).
  - `python mf_managers.py --template --out mf_cache/managers_template.csv` — blank skeleton for the hand-sourced manager history; `--validate` a filled one.
  - `python mf_agent_orchestrator.py` — mock demo (interactive profile prompts; pass `profile_config=` in code to skip).
  - Live scoring: `MasterOrchestrator(live=True).evaluate("<fund name or ISIN>", profile_config={...})`.
- `mf_cache/` is **gitignored** — fetched data and the Phase-B *research* outputs (features/labels parquets, CPCV results, reports) are generated and never committed. Regenerate as needed.
- **Three exceptions live outside it, and each is there because it is unbackfillable, not for convenience** — losing any of them cannot be undone by re-running anything:
  - `ledger/` — predictions are unrealizable once lost; they must survive as repo history. Append-only: a model is retired by naming it in `ledger/superseded_models.json` (readers skip it), **never** by deleting its rows.
  - `overrides/` — the hand-curated universe manifest + category/sector overrides no free source publishes.
  - `benchmarks/` — index series and a partly hand-annotated availability sheet; **no script in this repo rebuilds them**.
  - `model/` — the **shipped** `model_artifact_cohort.json` + its `psi_reference.json`. This is the frozen, holdout-validated payload that actually scores funds, and `mf_ledger` stamps its version as the `model_id` on every prediction. It is versioned rather than generated so a 2029 outcome ties back to the exact model that made the call. **Never retrain it on a schedule** — the holdout AUC (~0.558) belongs to one specific fit; regenerate deliberately via `python mf_model.py --stage cohort` and commit that as a model change, **bumping `mf_model.MODEL_VERSION`** so the ledger can tell the two apart (it stamps that string as `model_id`, and `mf_eval` segments outcomes by it).

## Conventions
- **`git pull` before you start, and again before you commit — CI WRITES TO `master`.** The nightly workflow commits the appended prediction ledger back to the repo (`nightly: append prediction ledger <date> [skip ci]`), so `origin/master` moves with no human pushing anything. A local commit made against a stale base turns a one-line push into a merge conflict on `ledger/predictions.jsonl` — an append-only file that must never be resolved by picking a side. This has cost this project two sessions already; it is not a hypothetical.
- **Commit per logical step** to `master`. The repo now HAS a remote (`origin` → `github.com/satyamx/mutual-fund-pipeline`), which is also the CI deploy path — but pushing stays an explicit, separate step: don't push unless asked. Run `/code-review` + `/security-review` before each commit. Co-Author trailer required.
- Surgical edits; match existing dense, comment-rich style. New modules follow the "pure, testable, injected-dependency, ships a `--selftest`" pattern the codebase already uses.
- Flag Fable-level tasks and confirm before deep-reasoning passes.

## Current status & roadmap
**Live status, the ordered to-do list, and all design decisions are in project memory — read `resume-point.md` first**, then `mf-architecture-decisions.md` and `build-roadmap.md` (`~/.claude/projects/.../memory/`). Don't duplicate volatile status here; that memory is the source of truth for "where we are / what's next." Phase B design detail: `docs/phase_b_design.md`.
