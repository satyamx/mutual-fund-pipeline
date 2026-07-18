# STATUS — MF Analysis (session handoff)

Repo-tracked mirror of the build status, so a `git push` hands off the full picture without relying on `~/.claude` memory syncing. See `CLAUDE.md` for orientation and the honesty invariant; deeper design rationale is in project memory (`resume-point.md`, `mf-architecture-decisions.md`) if that syncs to your environment.

**As of 2026-07-17: MF repo HEAD = `4c11ce6` (clean). Hisaab Kitaab repo HEAD = `9885565` (clean). Neither repo has a git remote — these commits are local-only until pushed.**

## Product shape (current)
- **System A (live, `mf_agent_orchestrator.py` → `RecommendationEngine`)** = the honest SCREEN: raw NAV facts (CAGR/vol/maxDD/Sortino/excess-vs-benchmark) + profile-weighted sub-scores + a **tri-colour 🟢BUY / 🔵HOLD / 🔴SELL verdict**. The verdict is a **transparent RULE over colour-coded metrics + hard compliance gates**, NOT the old below-chance weighted composite (which is deleted). The weighted "screen score" (0–100) is shown only as a banded *supporting datapoint*.
- **System B ("Sentinel", `mf_sentinel.py` → `SentinelEngine`)** = BUILT + wired into `MasterOrchestrator.evaluate()`/`print_report()` this session. Typed compliance/factor/manager ALERTS (INFO/WATCH/HIGH, each with an `AlertBasis` and evidence) + NFO manager-proxy dossier via `NFOAssessor`. Emits alerts/dossiers, never a fund number. `ScoringEngine.FACTOR_MAP` in `mf_pipeline.py` is kept (not deleted) as research-only infra — `mf_factor_backtest.py` still needs it live to regenerate the AUC~0.46 guardrail number Sentinel's `GUARDRAIL_ONLY` alerts cite; the live orchestrator never imports `ScoringEngine`.
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
- **NEW `mf_sentinel.py`** (System B, item #3) + wiring into `mf_agent_orchestrator.py` (`MasterOrchestrator.sentinel`, `evaluate()` returns `sentinel=`, `print_report()` prints alerts+NFO dossier). Multi-angle code review before commit surfaced and fixed 8 real issues: `EQ_DRAWDOWN_DEEP` always emitted INFO (copy-paste — deep tier now WATCH); `SEBI_CATEGORY_UNKNOWN` (a coverage gap) was mapped to WATCH/REGULATORY instead of INFO/COVERAGE — a genuine honesty-invariant violation, now fixed + selftest-locked; passive-fund expense INFO threshold was dead code (== the WATCH threshold, no `PASSIVE_EXPENSE_INFO` const existed); `NFO_MANAGER_PROXY_UNAVAILABLE` fired unconditionally because `self._managers_df` was never threaded into `_nfo()` (now wired via new `_manager_prior_funds()` helper); legacy Solution-Oriented bucket had no branch in `_factor_alerts` (silently produced zero alerts/zero dormant, breaking the module's own fire-or-dormant invariant); hand-rolled drawdown math replaced with the existing safe-divide-guarded `QuantEngine.max_drawdown`; dead `dossier` param removed from `_manager_alerts`; **known limitation** — `gate.assess()` needs `allotment_date` for NEWBORN/YOUNG gating but `FundDossier` has no such field; worked around by deriving it from `nav.dropna().index.min()` (a good proxy since real stores start `nav` at inception) rather than a schema change — revisit if this proves insufficient once real NFOs are scored.
- Confirmed with user: `ScoringEngine.FACTOR_MAP` in `mf_pipeline.py` stays (research-only, backs `mf_factor_backtest.py`'s AUC~0.46 number that Sentinel's `GUARDRAIL_ONLY` alerts cite) — NOT deleted, contrary to the original to-do #3 wording.
- Hisaab Kitaab `DECISIONS.md`: `f99fa96` (integration contract) + `9885565` (verdict amendment).

## To-do (ordered; resume at #4)
1. ✅ Serialize cohort inference payload (`b44a029`). Remaining sub-part: wire `cohort_q1` percentile into System A's output — BLOCKED on #4 (latest-anchor feature path).
2. ✅ Kill composite → honest SCREEN + tri-colour verdict (`31d42a2`, `29a633a`).
3. ✅ **Sentinel alert engine built + wired** (`mf_sentinel.py` new, wired into `MasterOrchestrator`). Typed alert registry (compliance breach=HIGH; factor thresholds e.g. downside_capture>1.15, drawdown-vs-COVID, expense outlier; manager MACS/tenure red flags — each citing evidence values), `NFOAssessor` dossier wired as B's output including manager cross-fund proxy lookup from `mf_cache/managers.csv`. Alert severity thresholds shipped as sensible defaults (documented as tunable constants at the top of `mf_sentinel.py`) — proceeded per user's earlier call, still open to revisit. `FACTOR_MAP` in `mf_pipeline.ScoringEngine` kept (research-only backtest infra), not deleted — see product-shape note above.
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
