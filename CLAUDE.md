# CLAUDE.md — MF Analysis

Guidance for Claude Code working in this repo. (User-global rules in `~/.claude/CLAUDE.md` also apply.)

## What this is
A **NAV-only Indian mutual-fund analysis pipeline** — a Python batch that fetches free public data, computes honest quant/compliance facts per fund, and emits a screen + recommendation. The intended consumer is the **Hisaab Kitaab** Flutter app (separate repo) via a batch-artifact handoff (versioned JSON → Drift/SQLite). Python does not run in the app.

**It is a measurement / screening tool, NOT a skill-verified buy engine.** NAV data cannot see inside a fund's holdings.

## THE HONESTY INVARIANT (read before changing any scoring/output)
This is the load-bearing constraint of the whole project:
- **Never surface fabricated accuracy or a laundered composite.** The only out-of-sample validated signal is the within-cohort `cohort_q1` model (holdout AUC ~0.578, lift 1.76x — *weak*). Everything else (the old utility composite) measured at/below chance.
- **Never turn missing data into a positive verdict.** Absence of evidence is a coverage flag, never a pass. HOLD is the honest default under thin evidence; a hard compliance breach forces SELL.
- **Never invent facts** — no fabricated holdings/TER/AUM/manager. Missing = `null` / "NOT AVAILABLE".
- **Never manufacture the target** by loosening a label or leaking. Report true OOS metrics (AUC / lift / precision), not raw accuracy on an imbalanced label.
- The current user-facing verdict (🟢BUY/🔵HOLD/🔴SELL) is an **explainable RULE over colour-coded metrics + hard compliance gates**, not a weighted black box. Keep it that way.

## Architecture (data → model → orchestrator)
- **Data layer** — `mf_datasources.py` (cache-first adapters over free no-key APIs: AMFI `NAVAll.txt`, mfapi.in NAV history, AMFI cap-band XLSX, yfinance; 20h cache TTL, self-healing), `bootstrap.py` (one-command fetch → `mf_cache/`), `mf_realstore.py` (`RealNAVStore` — real funds for the orchestrator; raises rather than fabricates).
- **Model layer (Phase B)** — `mf_features.py` (40 features, anti-lookahead `--selftest`), `mf_labels.py` (3y-forward OR-label + within-cohort targets), `mf_benchmarks.py` (per-sector/category real benchmarks + `PeerProxyResolver`), `mf_cv.py` (purged/embargoed CPCV + evaluate-once causal holdout + leakage self-test), `mf_model.py` (elastic-net primary + HGBT challenger; writes JSON artifacts to `mf_cache/phase_b/`), `mf_infer.py` (pure-numpy live inference of the cohort model — no sklearn needed).
- **Orchestrator (live product)** — `mf_agent_orchestrator.py`: Agent A (ingest/categorize) → B (backtest/manager-alpha) → C (`ProfileRiskScorerAgent`: sub-scores + raw facts) → D (news/sentiment) → `RecommendationEngine` (**System A** = the honest SCREEN + tri-colour verdict). `SEBI_2026` true-to-label compliance checks live here.
- **System B ("Sentinel")** — `mf_sentinel.py`, BUILT + wired: typed compliance/factor/manager/holdings ALERTS (INFO/WATCH/HIGH, each with an `AlertBasis` + evidence) + NFO dossier. Emits alerts, never a fund number. `_NEVER_HIGH` structurally bars non-regulatory bases from HIGH.
- **Live scoring + universe gating** — `mf_live_score.py` (`score_live()`: one-anchor cohort_q1 scoring, honesty gates, and frozen-panel insertion for funds outside the trained 136), `mf_universe.py` (AMFI category → canonical category + trained-universe refusal gate), `mf_overrides.py` (+ git-tracked `overrides/`: hand-curated `category`/`sector`, INPUTS ONLY).
- **Handoff + monitoring** — `mf_artifact.py` (versioned gzipped batch JSON for the app), `mf_ledger.py` (git-tracked `ledger/predictions.jsonl`; `realized_ic` honestly null until ~2029; PSI), `mf_eval.py` (model-health GREEN/AMBER/RED panel; a PENDING outcome never reddens).
- **NFO handling** — `mf_nfo_gate.py`: `EligibilityGate` refuses to score funds without enough NAV history; `NFOAssessor` emits a qualitative manager-proxy dossier.
- **Holdings** — `mf_holdings.py`: deterministic portfolio-structure facts + concentration screen from a disclosure snapshot. Never an ML feature (unbacktestable); drives the verdict only via the SEBI single-issuer compliance rule.
- **Research-only** — `mf_pipeline.py` `ScoringEngine` (`FACTOR_MAP`), `mf_factor_backtest.py`. The live orchestrator does NOT use `ScoringEngine`; its FACTOR_MAP composite is being retired into Sentinel alert thresholds.

## Environment & running
- **Python: venv only** — `./.venv/Scripts/python.exe`. The bare `python`/`python3`/`pip` on PATH are broken Windows-Store stubs; never use them. (Base interpreter: `C:\Python314\python.exe`.)
- **Always set `PYTHONIOENCODING=utf-8`** when running anything that prints ₹ or emoji (cp1252 crash otherwise).
- Common commands:
  - `python bootstrap.py --funds "Parag Parikh Flexi Cap" ...` — fetch/refresh real data into `mf_cache/`.
  - `python mf_model.py --stage cohort` — (re)train + write the cohort model artifacts.
  - `python mf_infer.py --selftest` — verify numpy inference == sklearn (bit-exact).
  - `python mf_features.py --selftest` / `python mf_cv.py --selftest` — anti-lookahead / leakage checks.
  - `python mf_agent_orchestrator.py` — mock demo (interactive profile prompts; pass `profile_config=` in code to skip).
  - Live scoring: `MasterOrchestrator(live=True).evaluate("<fund name or ISIN>", profile_config={...})`.
- `mf_cache/` is **gitignored** — fetched data and the Phase-B *research* outputs (features/labels parquets, CPCV results, reports) are generated and never committed. Regenerate as needed.
- **Three exceptions live outside it, and each is there because it is unbackfillable, not for convenience** — losing any of them cannot be undone by re-running anything:
  - `ledger/` — predictions are unrealizable once lost; they must survive as repo history.
  - `overrides/` — the hand-curated universe manifest + category/sector overrides no free source publishes.
  - `benchmarks/` — index series and a partly hand-annotated availability sheet; **no script in this repo rebuilds them**.
  - `model/` — the **shipped** `model_artifact_cohort.json` + its `psi_reference.json`. This is the frozen, holdout-validated payload that actually scores funds, and `mf_ledger` stamps its version as the `model_id` on every prediction. It is versioned rather than generated so a 2029 outcome ties back to the exact model that made the call. **Never retrain it on a schedule** — the holdout AUC (~0.578) belongs to one specific fit; regenerate deliberately via `python mf_model.py --stage cohort` and commit that as a model change.

## Conventions
- **Commit per logical step** to `master`. The repo now HAS a remote (`origin` → `github.com/satyamx/mutual-fund-pipeline`), which is also the CI deploy path — but pushing stays an explicit, separate step: don't push unless asked. Run `/code-review` + `/security-review` before each commit (`origin/HEAD` now resolves, so the security-review skill's auto-target works — the old "do it manually" workaround is obsolete). Co-Author trailer required.
- Surgical edits; match existing dense, comment-rich style. New modules follow the "pure, testable, injected-dependency, ships a `--selftest`" pattern the codebase already uses.
- Flag Fable-level tasks and confirm before deep-reasoning passes.

## Current status & roadmap
**Live status, the ordered to-do list, and all design decisions are in project memory — read `resume-point.md` first**, then `mf-architecture-decisions.md` and `build-roadmap.md` (`~/.claude/projects/.../memory/`). Don't duplicate volatile status here; that memory is the source of truth for "where we are / what's next." Phase B design detail: `docs/phase_b_design.md`.
