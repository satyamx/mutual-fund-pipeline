# STATUS — MF Analysis (session handoff)

Repo-tracked mirror of the build status, so a `git push` hands off the full picture without relying on `~/.claude` memory syncing. See `CLAUDE.md` for orientation and the honesty invariant; deeper design rationale is in project memory (`resume-point.md`, `mf-architecture-decisions.md`) if that syncs to your environment.

**As of 2026-07-17: MF repo HEAD = `4c11ce6` (clean). Hisaab Kitaab repo HEAD = `9885565` (clean). Neither repo has a git remote — these commits are local-only until pushed.**

## Product shape (current)
- **System A (live, `mf_agent_orchestrator.py` → `RecommendationEngine`)** = the honest SCREEN: raw NAV facts (CAGR/vol/maxDD/Sortino/excess-vs-benchmark) + profile-weighted sub-scores + a **tri-colour 🟢BUY / 🔵HOLD / 🔴SELL verdict**. The verdict is a **transparent RULE over colour-coded metrics + hard compliance gates**, NOT the old below-chance weighted composite (which is deleted). The weighted "screen score" (0–100) is shown only as a banded *supporting datapoint*.
- **System B ("Sentinel")** = NOT built yet (to-do #3): typed compliance/factor/manager ALERTS + NFO manager-proxy dossier; emits alerts/dossiers, never a fund number.
- Honesty invariant: HOLD is the default under thin evidence; a hard compliance breach forces SELL; BUY carries a "manager skill & holdings unverified" caveat; missing data = coverage flag, never a pass. The only OOS-validated signal is the within-cohort `cohort_q1` model (holdout AUC ~0.578, weak).

## Verdict rule + coloring thresholds (as implemented, `RecommendationEngine.run`)
Metric coloring `_band(x, good, bad, higher_better)` → green/red/amber, grey if n/a:
- `cagr` ≥0.10 green / ≤0.06 red · `excess_cagr_3y` ≥+0.01 / ≤−0.01 · `sortino_3y` ≥0.75 / ≤0.0 · `max_dd_3y` ≥−0.20 / ≤−0.35 (shallower better) · `consistency` ≥0.60 / ≤0.40 · `expense` ≤0.010 green / ≥0.020 red (lower better) · `compliance` = red if any critical, amber if warnings or (active & holdings missing), else green · `screen_score` ≥65 green / <45 red / else blue.
- Core metrics counted = [cagr, excess_cagr_3y, sortino_3y, max_dd_3y, consistency].
- **Rule:** `SELL` if (any critical breach OR any regulatory red_flag OR reds≥3 OR (reds>greens AND screen_score red)); else `BUY` if (greens≥3 AND reds==0 AND screen_score≠red); else `HOLD`.
- Return adds: `verdict, verdict_color, verdict_caveat, metric_colors`. Thresholds are TUNABLE defaults — revisit once the cohort percentile is wired into the verdict.

## Done this session (MF commits)
- `b44a029` — `build_cohort_artifact` (mf_model.py) writes the FULL cohort inference payload → `mf_cache/phase_b/model_artifact_cohort.json`; NEW `mf_infer.py` scores a live fund with numpy alone (no sklearn). `python mf_infer.py --selftest` proves numpy == sklearn to 2e-14.
- `31d42a2` — killed the surfaced composite; `RecommendationEngine` returns an honest SCREEN (facts + sub-scores + `[DATA GAP]` flags); `ProfileRiskScorerAgent` now returns a `facts` dict; compliance inflation gone (no numeric compliance score). print_report rewritten.
- `29a633a` — restored the tri-colour verdict honestly (rule over coloured metrics); fixed a real bug — regulatory red-flag matcher used naive substring ("ban" fired on "bank margins" → spurious SELL), now word-boundary matched.
- `4c11ce6` — added `CLAUDE.md`.
- Hisaab Kitaab `DECISIONS.md`: `f99fa96` (integration contract) + `9885565` (verdict amendment).

## To-do (ordered; resume at #3)
1. ✅ Serialize cohort inference payload (`b44a029`). Remaining sub-part: wire `cohort_q1` percentile into System A's output — BLOCKED on #4 (latest-anchor feature path).
2. ✅ Kill composite → honest SCREEN + tri-colour verdict (`31d42a2`, `29a633a`).
3. **RESUME HERE → Convert System B into the Sentinel alert engine.** Delete the `FACTOR_MAP` composite in `mf_pipeline.ScoringEngine`; define a typed alert registry (compliance breach = HIGH; factor thresholds e.g. down_capture>1.15, expense outlier; manager cross-fund red flags — each citing evidence values); wire `NFOAssessor` dossiers as B's output. **OPEN QUESTION for the user: alert severity THRESHOLDS (INFO vs WATCH vs HIGH) — proceed with sensible defaults & surface for review, or user sets cutoffs.**
4. Latest-anchor live-scoring path (score 1 anchor/fund, ~100× speedup). Unblocks #1's cohort wiring.
5. Artifact emitter (versioned gzipped JSON contract; now also carries verdict + verdict_color + metric_colors).
6. Prediction ledger (append-only) + monitoring block = the pipeline-evaluation item (interim IC, PSI drift, coverage, later realized cohort_q1 hit-rate). Ledger must exist BEFORE deploy.
7. Scheduler + throttle + incremental NAVAll append (GitHub Actions nightly, $0).
8. Phase 2: whole-market scoring (NAVAll category normalization + `OUT_OF_TRAINING_UNIVERSE` flag).
9. DEFERRED/GATED: 1y-forward cohort head (only if it clears CPCV >0.5 post phase-2).

**Then:** Phase C (evaluate NFOs JM Multi Asset + TRUSTMF Large & Mid via B's dossier) → Step 3 (manager-proxy dossiers, subsumed into NFOAssessor) → capstone (layman summary + retrospective + integration plan). App integration itself still deferred.

## Env / workflow
- Python: venv only — `./.venv/Scripts/python.exe`. Bare `python`/`pip` are broken stubs.
- `PYTHONIOENCODING=utf-8` for anything printing ₹ / emoji.
- Commit per step to `master` after `/code-review` + `/security-review` (security-review auto-targets `origin/HEAD` → fails, no remote; do it manually). Co-Author trailer required.
- `mf_cache/` is gitignored (data + model artifacts are generated; regenerate via `python mf_model.py --stage cohort`).
