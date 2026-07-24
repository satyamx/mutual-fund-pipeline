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
- **System B ("Sentinel")** — NOT built yet (to-do #3): will emit typed compliance/factor/manager ALERTS + NFO manager-proxy dossier, never a fund number.
- **NFO handling** — `mf_nfo_gate.py`: `EligibilityGate` refuses to score funds without enough NAV history; `NFOAssessor` emits a qualitative manager-proxy dossier.
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
- `mf_cache/` is **gitignored** — all fetched data and model artifacts are generated, never committed. Regenerate as needed.

## Conventions
- **Commit per logical step** to `master`. The repo now HAS a remote (`origin` → `github.com/satyamx/mutual-fund-pipeline`), which is also the CI deploy path — but pushing stays an explicit, separate step: don't push unless asked. Run `/code-review` + `/security-review` before each commit (`origin/HEAD` now resolves, so the security-review skill's auto-target works — the old "do it manually" workaround is obsolete). Co-Author trailer required.
- Surgical edits; match existing dense, comment-rich style. New modules follow the "pure, testable, injected-dependency, ships a `--selftest`" pattern the codebase already uses.
- Flag Fable-level tasks and confirm before deep-reasoning passes.

## Current status & roadmap
**Live status, the ordered to-do list, and all design decisions are in project memory — read `resume-point.md` first**, then `mf-architecture-decisions.md` and `build-roadmap.md` (`~/.claude/projects/.../memory/`). Don't duplicate volatile status here; that memory is the source of truth for "where we are / what's next." Phase B design detail: `docs/phase_b_design.md`.
