"""
====================================================================================
mf_artifact.py — versioned batch JSON artifact emitter (to-do #5)
====================================================================================
Runs MasterOrchestrator(live=True) over every fund in the curated universe
manifest and serializes the honest, per-fund SCREEN + Sentinel + cohort_signal
fields into ONE versioned, gzipped JSON artifact for the Hisaab Kitaab app to
ingest later (still gated "DO NOT START INTEGRATION YET" — this is the batch
producer side only). Never a fabricated number: every gap the live orchestrator
already reports (coverage_flags, every mf_live_score refusal status, missing
TER/AUM/manager) is carried through as `null` or a `data_flags` entry, not
guessed. Note that a refusal is CORRECT behaviour, not a pipeline error: the
refusal counts in coverage{} are deliberately kept out of mf_eval's error rate.

DEFAULT-PROFILE VERDICT (user decision, 2026-07-19): each fund's `signal_a`
carries a verdict/verdict_color/metric_colors computed against InvestorProfile's
OWN DEFAULT (horizon=7.5y, liquidity=none, risk=high) — the batch artifact runs
before the app knows the real user's profile. Every record is stamped with
`verdict_basis.is_default_profile=true` + the exact default profile values so
the app can render an explicit "default profile assumed — personalize for your
own verdict" disclaimer until the user sets their own. The RAW facts/sub_scores
underneath are profile-agnostic (see mf-architecture-decisions memory: "profile
weighting moves client-side") — only the verdict/colours are profile-specific,
and only a default one is baked in here.

D4 OPTION C (2026-08-06, docs/d4_profile_verdicts.md): the default verdict above
is no longer the app's only option. The investor profile reaches the verdict rule
through EXACTLY ONE term — `utility < screen_score_red_below` — so each fund has
exactly two possible verdicts, and BOTH ship in `verdict_branches`. The app
personalises by recomputing only the weighted utility from its own profile and
picking a branch; the verdict RULE never leaves Python, so it cannot drift into a
Dart reimplementation. `weight_matrix`/`utility_score` are the DEFAULT-profile
values, shipped so a ported weight function can be VERIFIED against them on every
record instead of trusted.

CONTRACT (see mf-architecture-decisions memory for the original 2026-07-16
design; extended here with the restored verdict + its default-profile stamp):
  top-level: artifact_version, generated_at, pipeline_sha, model_id,
             default_profile{}, coverage{}, monitoring{ledger, realized_ic,
             psi, rank_stability, note},
             evaluation{status, headline, metrics[], outcome, disclaimer} (the
               app-facing MODEL-HEALTH panel — mf_eval grades the raw monitoring
               signals GREEN/AMBER/RED; outcome skill stays PENDING pre-maturity
               and never reddens the headline), funds[]
  per-fund:  amfi_code, isin, scheme_name, category, sector, eligibility,
             facts{cagr, vol, max_dd, sortino, benchmark, expense},
             signal_a{sub_scores, verdict, verdict_color, verdict_caveat,
                      metric_colors, verdict_basis,
                      verdict_branches{score_red{}, score_not_red{}},
                      screen_score_red_below, weight_matrix, utility_score,
                      cohort_status, cohort_q1_prob, cohort_percentile,
                      cohort_n, signal_context},
             alerts_b[], nfo_dossier, data_flags[], lock{type, exit_load_days}

monitoring{} (to-do #6, mf_ledger.py): every OK cohort_signal this run is
appended to the git-tracked prediction ledger (ledger/predictions.jsonl —
NOT mf_cache/, which is gitignored/regenerable; predictions are the opposite).
`realized_ic` stays honestly null (PENDING_MATURITY/INSUFFICIENT_MATURED)
until ~3 real years of matured outcomes exist — no untested-horizon proxy is
ever published under that name. `psi` (population stability index) is
computed FRESH every run from the live scoring population against an
empirical training-time reference — no ledger/waiting needed. `rank_stability`
is a separate, honestly-named run-to-run consistency QA proxy, never a skill
metric. lock{} is always null — no free data source publishes exit-load/
lock-in terms (the same honesty gap as expense_ratio/aum_cr/manager name
elsewhere in this pipeline).
====================================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import mf_eval
import mf_ledger
from mf_agent_orchestrator import TODAY, InvestorProfile, MasterOrchestrator
from mf_datasources import CACHE
from mf_labels import load_manifest

ARTIFACT_VERSION = "mf_artifact_v1"
OUT_DIR = CACHE / "artifacts"

LOGGER = logging.getLogger("MFArtifact")

DATA_FLAG_THIN_COHORT = "THIN_COHORT"
DATA_FLAG_EXPENSE_UNAVAILABLE = "EXPENSE_UNAVAILABLE"
DATA_FLAG_NOT_IN_UNIVERSE = "NOT_IN_UNIVERSE"
DATA_FLAG_OUT_OF_TRAINING_UNIVERSE = "OUT_OF_TRAINING_UNIVERSE"
DATA_FLAG_SECTOR_UNRESOLVED = "SECTOR_UNRESOLVED"
DATA_FLAG_STALE_NAV = "STALE_NAV"
DATA_FLAG_SHORT_HISTORY = "SHORT_HISTORY"
DATA_FLAG_SECTOR_UNMAPPED = "SECTOR_UNMAPPED"
DATA_FLAG_COHORT_UNSCORED_OTHER = "COHORT_UNSCORED_OTHER"

# One flag per mf_live_score refusal status, as an exhaustive mapping rather than
# the if/elif chain this used to be — that chain is precisely how these drifted.
# Phase 2 (#8) added OUT_OF_TRAINING_UNIVERSE and SECTOR_UNRESOLVED to
# mf_live_score and this module was never updated, so both produced NO flag at
# all: a fund the model PERMANENTLY refuses looked, in the artifact, identical to
# one carrying a clean signal. Worse, NOT_IN_UNIVERSE was mapped onto the
# OUT_OF_TRAINING_UNIVERSE flag, and since #8 those mean opposite things — a
# fixable data gap (fetch the NAV / add to the manifest) versus a permanent
# refusal (the model was never fitted on this category and a probability here
# would claim the holdout AUC transfers to a population it never saw). The app
# branches on these codes, so the two demand opposite caller behaviour.
#
# The fallback is a STABLE generic code, never an interpolated one — the same
# reason SEBI_OTHER[{rule_id}] was replaced by a typed SEBI_SINGLE_ISSUER_BREACH
# in mf_sentinel: a status embedded in a string is not stably branchable. The
# exact status always travels in signal_a.cohort_status, so nothing is lost.
# The selftest asserts this mapping covers mf_live_score's whole vocabulary, so
# a future status addition fails loudly instead of silently vanishing again.
_COHORT_STATUS_FLAG = {
    "THIN_COHORT": DATA_FLAG_THIN_COHORT,
    "STALE_NAV": DATA_FLAG_STALE_NAV,
    "NOT_IN_UNIVERSE": DATA_FLAG_NOT_IN_UNIVERSE,
    "OUT_OF_TRAINING_UNIVERSE": DATA_FLAG_OUT_OF_TRAINING_UNIVERSE,
    "SECTOR_UNRESOLVED": DATA_FLAG_SECTOR_UNRESOLVED,
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True,
                              timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 — a missing/broken git binary must not crash the batch run
        return "unknown"


def _finite_or_none(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None


def _json_safe(obj: Any) -> Any:
    """Recursively map non-finite floats to None inside a nested structure.

    The flat numeric leaves below are guarded field-by-field with
    `_finite_or_none`, but three payloads are nested dicts this module does NOT
    enumerate — alert `evidence`, the NFO dossier, and the cohort `signal_context`.
    A single NaN anywhere in one of them would raise inside write_artifact()'s
    `json.dumps(..., allow_nan=False)` and discard the WHOLE batch, including every
    other fund's already-computed record. The established behaviour for one bad
    fund is to drop just that record, never to nuke the batch.

    Today nothing trips this (every alert author routes floats through
    `mf_sentinel._fmt()`), but nothing enforces that convention either — this makes
    the guarantee structural instead of conventional.

    bools pass through untouched: `isinstance(True, int)` is True in Python, so a
    naive numeric branch would silently rewrite True as 1.0.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj if np.isfinite(obj) else None
    # numpy scalars: np.float64 subclasses float and is caught above, but np.float32
    # / np.int32 do not — and json.dumps rejects those outright ("not JSON
    # serializable"), which is the same batch-wide failure by a different exception.
    # Unwrap to a Python scalar so the guarantee doesn't depend on which numpy width
    # an alert author happened to produce.
    if isinstance(obj, np.generic):
        v = obj.item()
        return v if not isinstance(v, float) or np.isfinite(v) else None
    return obj


def _data_flags(dossier, cohort_signal: Optional[Dict[str, Any]], eligibility: str) -> List[str]:
    flags: List[str] = []
    if dossier.category in ("Sectoral", "Thematic") and not dossier.declared_sector:
        flags.append(DATA_FLAG_SECTOR_UNMAPPED)
    if not np.isfinite(dossier.expense_ratio):
        flags.append(DATA_FLAG_EXPENSE_UNAVAILABLE)
    if eligibility in ("NEWBORN", "YOUNG"):
        flags.append(DATA_FLAG_SHORT_HISTORY)
    if cohort_signal is not None:
        status = cohort_signal["status"]
        if status != "OK":
            flags.append(_COHORT_STATUS_FLAG.get(status, DATA_FLAG_COHORT_UNSCORED_OTHER))
    return flags


def build_fund_record(result: Dict[str, Any], profile: InvestorProfile,
                      is_default_profile: bool) -> Optional[Dict[str, Any]]:
    """One evaluate() result -> the per-fund contract dict, or None (logged) if
    the evaluation itself failed — a failed fund is dropped from the artifact
    entirely rather than emitted with fabricated fields."""
    if "error" in result:
        LOGGER.warning("Dropping %r from artifact — evaluation failed: %s",
                       result.get("query"), result["error"])
        return None

    d = result["dossier"]
    rec = result["recommendation"]
    ps = result["profile_score"]
    sr = result["sentinel"]
    cs = result.get("cohort_signal")
    f = rec["facts"]

    return dict(
        amfi_code=d.amfi_code or None,
        isin=d.isin,
        scheme_name=d.scheme_name,
        category=d.category,
        sector=d.declared_sector,
        eligibility=sr.eligibility,
        facts=dict(
            cagr=_finite_or_none(f["cagr"]), vol=_finite_or_none(f["vol_1y"]),
            max_dd=_finite_or_none(f["max_dd_3y"]), sortino=_finite_or_none(f["sortino_3y"]),
            benchmark=d.benchmark, expense=_finite_or_none(d.expense_ratio)),
        signal_a=dict(
            # sub_scores can be NaN for a NEWBORN/YOUNG fund (Agent C runs
            # unconditionally, no eligibility gate) — guard the same way every
            # other numeric leaf is guarded, so one degraded fund's NaN can't
            # trip write_artifact()'s allow_nan=False over the WHOLE batch.
            sub_scores={k: _finite_or_none(v) for k, v in ps["sub_scores"].items()},
            verdict=rec["verdict"], verdict_color=rec["verdict_color"],
            verdict_caveat=rec["verdict_caveat"], metric_colors=rec["metric_colors"],
            verdict_basis=dict(is_default_profile=is_default_profile,
                               profile=profile.model_dump(mode="json")),
            # --- D4 option C: everything the app needs to re-derive the verdict for
            # the REAL user's profile without reimplementing the verdict rule.
            # The profile enters the rule through one term only (utility < the cutoff
            # below), so both possible verdicts are precomputed here and the app picks:
            #     u = 100 * Σ weights(userProfile)[k] * sub_scores[k]
            #     verdict = verdict_branches[u < screen_score_red_below ? "score_red"
            #                                                          : "score_not_red"]
            # weight_matrix + utility_score are the DEFAULT-profile values, shipped so
            # a ported weight function can be checked against them on every record
            # rather than trusted. See docs/d4_profile_verdicts.md.
            verdict_branches=rec["verdict_branches"],
            screen_score_red_below=rec["screen_score_red_below"],
            weight_matrix={k: _finite_or_none(v) for k, v in ps["weight_matrix"].items()},
            # NaN for a NEWBORN/YOUNG fund, same as the sub_scores it is built from:
            # utility is their weighted sum, so one NaN sub-score propagates here.
            utility_score=_finite_or_none(rec["profile_fit_score"]),
            # The exact mf_live_score status — OK, or WHICH refusal. data_flags[]
            # below carries only the coarse enum subset, so this is the app's
            # stable source of truth for *why* cohort_q1_prob is null, and the
            # one field that survives a future status this module doesn't know.
            cohort_status=cs["status"] if cs else None,
            cohort_q1_prob=_finite_or_none(cs["probability"]) if cs else None,
            cohort_percentile=_finite_or_none(cs["cohort_percentile"]) if cs else None,
            cohort_n=cs["cohort_n"] if cs else None,
            signal_context=_json_safe(cs["signal_context"]) if cs else None),
        # Nested payloads this module doesn't enumerate leaf-by-leaf — guarded
        # wholesale so one NaN can't take out the entire batch write. See _json_safe.
        alerts_b=_json_safe([dataclasses.asdict(a) for a in sr.alerts]),
        nfo_dossier=_json_safe(sr.nfo_dossier),
        # Human-readable DATA GAP strings (same ones print_report shows) —
        # data_flags[] below is only the coarse machine-enum subset.
        coverage_flags=rec["coverage_flags"],
        data_flags=_data_flags(d, cs, sr.eligibility),
        # No free data source publishes exit-load/lock-in terms — never guessed.
        lock=dict(type=None, exit_load_days=None),
    )


def build_artifact(scheme_names: List[str], orch: Optional[MasterOrchestrator] = None,
                   profile: Optional[InvestorProfile] = None,
                   ledger_path: Path = mf_ledger.PREDICTIONS_PATH,
                   realizations_path: Path = mf_ledger.REALIZATIONS_PATH) -> Dict[str, Any]:
    """Runs ONE MasterOrchestrator(live=True) (reused across every fund so its
    RealNAVStore fund cache and cohort-scoring resources build once) against
    `profile` (defaults to InvestorProfile()'s own baseline), and assembles the
    full versioned artifact. Every OK cohort_signal is appended to the
    (git-tracked) prediction ledger — see mf_ledger.py for why this must happen
    now even though nothing can be realized for ~3 years."""
    orch = orch if orch is not None else MasterOrchestrator(live=True)
    profile = profile if profile is not None else InvestorProfile()
    is_default_profile = profile.model_dump(mode="json") == InvestorProfile().model_dump(mode="json")
    profile_config = profile.model_dump(mode="json")

    run_id = datetime.now(timezone.utc).isoformat()
    pipeline_sha = _git_sha()

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    ledger_rows: List[Dict[str, Any]] = []
    for name in scheme_names:
        result = orch.evaluate(name, profile_config=profile_config, argv=[])
        rec = build_fund_record(result, profile, is_default_profile)
        if rec is None:
            errors.append(dict(query=name, error=result.get("error", "unknown")))
            continue
        records.append(rec)
        row = mf_ledger.build_row(rec["amfi_code"], result.get("cohort_signal"),
                                  run_id=run_id, pipeline_sha=pipeline_sha,
                                  model_id=orch.cohort_model_id)
        if row is not None:
            ledger_rows.append(row)
    n_appended = mf_ledger.append_predictions(ledger_rows, path=ledger_path)

    live_res = orch.cohort_live_resources()
    psi = (mf_ledger.compute_psi_live(live_res[0], live_res[1], live_res[3], TODAY,
                                      orch.cohort_model_id)
          if live_res is not None else
          dict(status="UNAVAILABLE", psi_max=None, psi_mean=None, worst=[],
              n_features=0, skipped_features=[]))
    monitoring = mf_ledger.monitoring_block(run_id=run_id, rows_appended=n_appended,
                                            current_rows=ledger_rows, psi=psi,
                                            predictions_path=ledger_path,
                                            realizations_path=realizations_path)

    coverage = dict(
        n_total=len(scheme_names), n_ok=len(records), n_errors=len(errors),
        # Reasons kept alongside the count — a systemic failure (e.g. every fund
        # erroring the same way) must be visible in the artifact, not just a bare
        # number indistinguishable from ordinary per-fund attrition.
        errors=errors,
        n_evaluable=sum(1 for r in records if r["eligibility"] == "EVALUABLE"),
        n_thin_cohort=sum(1 for r in records if DATA_FLAG_THIN_COHORT in r["data_flags"]),
        n_stale_nav=sum(1 for r in records if DATA_FLAG_STALE_NAV in r["data_flags"]),
        # Counted separately, not summed into one "unscored" number: NOT_IN_UNIVERSE
        # is a backlog item (fetch it), OUT_OF_TRAINING_UNIVERSE is a permanent
        # property of the trained model, and SECTOR_UNRESOLVED is a curation
        # worklist (mf_overrides.py --gaps). Collapsing them would hide which of
        # the three is actually growing.
        n_not_in_universe=sum(
            1 for r in records if DATA_FLAG_NOT_IN_UNIVERSE in r["data_flags"]),
        n_out_of_training_universe=sum(
            1 for r in records if DATA_FLAG_OUT_OF_TRAINING_UNIVERSE in r["data_flags"]),
        n_sector_unresolved=sum(
            1 for r in records if DATA_FLAG_SECTOR_UNRESOLVED in r["data_flags"]),
        n_expense_available=sum(1 for r in records if r["facts"]["expense"] is not None),
    )
    # App-facing MODEL-HEALTH panel: the raw monitoring{}/coverage{} signals graded
    # against documented GREEN/AMBER/RED thresholds (mf_eval), so the app can render
    # "when should I distrust this model" directly instead of re-deriving it. Outcome
    # skill stays PENDING (unmeasurable pre-maturity) and never reddens the headline.
    evaluation = mf_eval.build_report_from_ledger(
        monitoring, coverage, model_id=orch.cohort_model_id,
        realizations_path=realizations_path, predictions_path=ledger_path)
    return dict(
        artifact_version=ARTIFACT_VERSION,
        generated_at=run_id,
        pipeline_sha=pipeline_sha,
        model_id=orch.cohort_model_id,
        default_profile=profile_config,
        coverage=coverage,
        monitoring=monitoring,
        evaluation=evaluation,
        funds=records,
    )


def write_artifact(artifact: Dict[str, Any], out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{artifact['artifact_version']}_{stamp}.json.gz"
    payload = json.dumps(artifact, indent=None, allow_nan=False).encode("utf-8")
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return path


def error_rate_exceeded(coverage: Dict[str, Any], max_rate: float) -> Optional[str]:
    """None if the batch's failure rate is acceptable, else a human-readable reason.

    A batch that drops most of the universe is a FAILED run, not a thin one. Dropping
    ONE bad fund is deliberate (the alternative — nuking the batch — is worse), but
    that same tolerance meant the first real CI run emitted 18 of 136 funds and still
    reported success. Pure so the selftest can prove it fires, not just that it passes.
    """
    n_total = coverage.get("n_total", 0) or 0
    if n_total <= 0:
        return "no funds were attempted at all"
    n_err = coverage.get("n_errors", 0) or 0
    rate = n_err / n_total
    if rate <= max_rate:
        return None
    reasons = sorted({e.get("error", "?") for e in coverage.get("errors", [])})
    return (f"{n_err}/{n_total} funds ({rate:.1%}) failed to evaluate, above the "
            f"--max-error-rate of {max_rate:.1%}. Distinct reasons: "
            + ("; ".join(reasons[:5]) or "none recorded"))


def ledger_stalled(monitoring: Dict[str, Any], today: pd.Timestamp,
                   max_anchor_age_days: int) -> Optional[str]:
    """None if the ledger is progressing, else a human-readable reason.

    Guards the failure this pipeline actually had: with `TODAY` frozen at
    2026-07-13, every scoring anchor was pinned to that date, so the nightly
    re-scored one historical anchor forever, deduped against rows that already
    existed and appended NOTHING — for weeks, while reporting success. The
    prediction ledger exists to accumulate predictions that mature in ~2029; a
    run that cannot add one is doing 24 minutes of work for no purpose.

    Deliberately keyed on ANCHOR AGE, not on "zero rows appended". Zero appended
    is correct and common — a same-day re-run, or a manual dispatch after the
    cron, has nothing new to say and must not fail. What is never correct is the
    newest anchor drifting far behind the run date: that means the pipeline has
    stopped tracking time, whatever the append count says.
    """
    ledger = (monitoring or {}).get("ledger") or {}
    newest = ledger.get("last_anchor")
    if not newest:
        # Empty ledger, or a schema too old to carry last_anchor. Deliberately NOT
        # falling back to first_anchor: that never moves once seeded, so it would
        # fail every healthy run the moment the ledger got older than the window.
        return None
    age = (pd.Timestamp(today).normalize() - pd.Timestamp(newest).normalize()).days
    if age <= max_anchor_age_days:
        return None
    return (f"newest prediction anchor is {newest} — {age} days behind the run date, "
            f"above --max-anchor-age-days of {max_anchor_age_days}. The scoring "
            f"anchor has stopped advancing, so no new prediction can ever be logged. "
            f"Check that TODAY is not pinned (MF_TODAY) and that NAV is refreshing.")


def _load_scheme_names(limit: Optional[int] = None) -> List[str]:
    names = list(load_manifest()["scheme_name"])
    return names[:limit] if limit else names


# ==============================================================================
# SELFTEST — contract shape, honesty gates, gzip round-trip. Runs against real
# cached data (a small real slice of the universe, not synthetic funds) since
# the whole point is verifying MasterOrchestrator's actual live output serializes
# honestly — no artifact-only unit test would catch a schema drift there.
# ==============================================================================
def _selftest() -> None:
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp())
    try:
        ledger_path = tmpdir / "predictions.jsonl"   # never touch the real ledger/ from a selftest
        realizations_path = tmpdir / "realizations.jsonl"

        names = _load_scheme_names(limit=3)
        orch = MasterOrchestrator(live=True)
        profile = InvestorProfile()
        artifact = build_artifact(names, orch=orch, profile=profile, ledger_path=ledger_path,
                                  realizations_path=realizations_path)

        assert artifact["coverage"]["n_total"] == 3
        assert artifact["coverage"]["n_ok"] + artifact["coverage"]["n_errors"] == 3
        assert artifact["model_id"], "FAIL: model_id missing (cohort artifact not loaded)"
        assert artifact["pipeline_sha"] != "unknown", "FAIL: git sha resolution failed"
        print(f"[selftest] top-level shape + coverage counters consistent "
              f"(n_ok={artifact['coverage']['n_ok']}, model_id={artifact['model_id']}) — PASS")

        # Honesty: realized_ic stays null pre-maturity (no untested-horizon proxy published);
        # PSI/rank_stability degrade to a known status rather than fabricating a number.
        m = artifact["monitoring"]
        assert m["realized_ic"]["status"] in ("PENDING_MATURITY", "INSUFFICIENT_MATURED")
        assert m["realized_ic"]["value"] is None
        assert m["psi"]["status"] in ("OK", "MODERATE_SHIFT", "SIGNIFICANT_SHIFT",
                                      "INSUFFICIENT_DATA", "REFERENCE_MISSING", "REFERENCE_STALE")
        assert m["rank_stability"]["status"] == "FIRST_RUN"   # fresh tmp ledger, no prior run
        assert m["ledger"]["rows_appended_this_run"] == artifact["coverage"]["n_ok"]
        print(f"[selftest] monitoring{{}}: realized_ic honestly null "
              f"(status={m['realized_ic']['status']}), psi status={m['psi']['status']}, "
              f"ledger rows_appended={m['ledger']['rows_appended_this_run']} — PASS")

        # evaluation{}: the app-facing model-health panel. Fresh ledger has no matured
        # outcomes, so outcome_skill MUST be PENDING and MUST NOT redden the overall
        # status; the disclaimer separating health from accuracy must be present.
        ev = artifact["evaluation"]
        assert ev["status"] in ("GREEN", "AMBER", "RED", "PENDING")
        outcome_m = next(x for x in ev["metrics"] if x["name"] == "outcome_skill")
        assert outcome_m["status"] == "PENDING", outcome_m
        assert ev["status"] != "RED" or "outcome_skill" not in {x["name"] for x in ev["metrics"]
                                                                 if x["status"] == "RED"}, \
            "FAIL: a pending outcome must never drive the model to RED"
        assert "not fund-outcome accuracy" in ev["disclaimer"].lower() or \
               "not fund" in ev["disclaimer"].lower()
        assert ev["model_id"] == artifact["model_id"]
        print(f"[selftest] evaluation{{}}: overall={ev['status']}, outcome_skill=PENDING "
              f"(never reddens), disclaimer present — PASS")

        # Re-running the same batch against the same ledger must be idempotent (dedupe).
        artifact2 = build_artifact(names, orch=orch, profile=profile, ledger_path=ledger_path,
                                   realizations_path=realizations_path)
        assert artifact2["monitoring"]["ledger"]["rows_appended_this_run"] == 0, \
            "FAIL: re-running the same anchors should append zero new ledger rows"
        # Only 3 funds in this test batch (< mf_ledger.MIN_COMMON_FOR_STABILITY=20), so
        # the honest outcome is INSUFFICIENT_OVERLAP, not a fabricated correlation —
        # it must still find last run's rows (prev_run_id set), just not enough of them.
        rs = artifact2["monitoring"]["rank_stability"]
        assert rs["status"] == "INSUFFICIENT_OVERLAP" and rs["n_common"] == 3 and rs["prev_run_id"], \
            f"FAIL: expected INSUFFICIENT_OVERLAP with n_common=3, got {rs}"
        print("[selftest] ledger idempotency across repeated batch runs — PASS")

        required_top = {"amfi_code", "isin", "scheme_name", "category", "sector", "eligibility",
                        "facts", "signal_a", "alerts_b", "nfo_dossier", "coverage_flags",
                        "data_flags", "lock"}
        n_branch_differ = 0     # how many funds the branch choice actually SWINGS
        for rec in artifact["funds"]:
            assert required_top <= rec.keys(), f"FAIL: missing keys in {rec['scheme_name']}"
            assert isinstance(rec["coverage_flags"], list)
            # lock has no free data source anywhere in this pipeline — must always be null.
            assert rec["lock"] == dict(type=None, exit_load_days=None)
            # Every verdict in the batch artifact is against the default profile, stamped as such.
            vb = rec["signal_a"]["verdict_basis"]
            assert vb["is_default_profile"] is True
            assert vb["profile"]["horizon_years"] == profile.horizon_years
            assert vb["profile"]["liquidity_need"] == profile.liquidity_need.value
            assert vb["profile"]["risk_appetite"] == profile.risk_appetite.value
            # expense missing (no free TER source) must show as null + a data_flag, never guessed.
            if rec["facts"]["expense"] is None:
                assert DATA_FLAG_EXPENSE_UNAVAILABLE in rec["data_flags"]
            for v in rec["signal_a"]["sub_scores"].values():
                assert v is None or (isinstance(v, float) and np.isfinite(v))
            # Why cohort_q1_prob is null must always be readable, not inferred from
            # the absence of a number.
            assert "cohort_status" in rec["signal_a"]
            if rec["signal_a"]["cohort_q1_prob"] is None:
                assert rec["signal_a"]["cohort_status"] != "OK"

            # ---- D4 option C: the app's re-derivation must be exactly reproducible.
            sa = rec["signal_a"]
            assert {"verdict_branches", "screen_score_red_below",
                    "weight_matrix", "utility_score"} <= sa.keys(), \
                f"FAIL: D4 fields missing in {rec['scheme_name']}"
            br = sa["verdict_branches"]
            assert set(br) == {"score_red", "score_not_red"}
            for b in br.values():
                assert {"verdict", "verdict_color", "verdict_caveat"} == b.keys()
            if br["score_red"]["verdict"] != br["score_not_red"]["verdict"]:
                n_branch_differ += 1
            # THE load-bearing invariant. The app picks a branch by comparing the
            # utility it recomputed against the shipped cutoff; if the branch that
            # rule selects were not the verdict Python actually issued, every
            # personalised verdict would be subtly wrong AND the default-profile
            # verdict shown beside it would disagree with itself.
            if sa["utility_score"] is not None:
                picked = br["score_red" if sa["utility_score"] < sa["screen_score_red_below"]
                            else "score_not_red"]
                assert picked["verdict"] == sa["verdict"], (
                    f"FAIL: branch/verdict disagree on {rec['scheme_name']}: "
                    f"utility={sa['utility_score']} picked={picked['verdict']} "
                    f"shipped={sa['verdict']}")
                assert picked["verdict_caveat"] == sa["verdict_caveat"]
                # And the recomputation FORMULA itself — this is the exact arithmetic
                # a Dart port will run, so if `100 * Σ w·s` does not reproduce the
                # shipped utility, the documented client-side formula is wrong.
                if all(v is not None for v in sa["sub_scores"].values()):
                    recomputed = 100.0 * sum(sa["weight_matrix"][k] * sa["sub_scores"][k]
                                             for k in sa["sub_scores"])
                    # weight_matrix/sub_scores are rounded for transport (3dp), so this
                    # is a tolerance on ROUNDING, not on agreement.
                    assert abs(recomputed - sa["utility_score"]) < 0.5, (
                        f"FAIL: utility formula does not reproduce on {rec['scheme_name']}: "
                        f"recomputed={recomputed:.3f} shipped={sa['utility_score']}")
                    # Weights are a normalised distribution — a Dart port that forgets
                    # the final /total would silently scale every utility.
                    assert abs(sum(sa["weight_matrix"].values()) - 1.0) < 1e-6
        print(f"[selftest] {len(artifact['funds'])} fund records: required keys present, "
              f"lock always null, verdict_basis stamped default-profile, "
              f"expense gap flagged not guessed, sub_scores NaN-safe, "
              f"cohort_status always present — PASS")
        # Say plainly how much the branch/verdict assertion above actually proved on
        # THIS batch. Where both branches agree the assertion still holds but cannot
        # discriminate, and a vacuous pass reported as a real one is worse than no
        # test at all — so report the discriminating count instead of implying it.
        print(f"[selftest] D4 branch pair: verdict formula reproduces and branch "
              f"selection matches the shipped verdict on {len(artifact['funds'])} "
              f"records; the branch choice SWINGS the verdict on {n_branch_differ} of "
              f"them{' (so the selection assert is structural only on this batch)' if not n_branch_differ else ''} — PASS")

        # ---- Every mf_live_score refusal reaches the app as its OWN flag ----------
        # Phase 2 added two statuses that this module silently dropped, and mapped a
        # third onto a flag meaning the OPPOSITE thing. These asserts are what stop
        # that recurring: the mapping must stay in sync with mf_live_score's whole
        # vocabulary, and the two universe statuses must never collapse together.
        import types

        import mf_live_score
        stub = types.SimpleNamespace(category="Flexi Cap", declared_sector="",
                                     expense_ratio=0.01)
        expected_statuses = set(mf_live_score._STATUSES) - {"OK"}
        assert expected_statuses <= set(_COHORT_STATUS_FLAG), (
            "FAIL: mf_live_score gained a status with no artifact flag — the app would "
            f"see no reason at all for a null score: {expected_statuses - set(_COHORT_STATUS_FLAG)}")

        seen = {}
        for st in sorted(expected_statuses):
            got = _data_flags(stub, {"status": st}, "EVALUABLE")
            assert got == [_COHORT_STATUS_FLAG[st]], f"FAIL: {st} -> {got}"
            seen[st] = got[0]
        assert len(set(seen.values())) == len(seen), \
            f"FAIL: two statuses share one flag — they demand different caller behaviour: {seen}"
        # The specific regression: a fixable data gap must NOT read as a permanent refusal.
        assert seen["NOT_IN_UNIVERSE"] != seen["OUT_OF_TRAINING_UNIVERSE"]
        assert _data_flags(stub, {"status": "OK"}, "EVALUABLE") == []
        # An unknown future status degrades to a STABLE generic code — never vanishes,
        # and never an interpolated one the app can't branch on.
        assert _data_flags(stub, {"status": "SOME_FUTURE_STATUS"}, "EVALUABLE") == \
            [DATA_FLAG_COHORT_UNSCORED_OTHER]
        print(f"[selftest] all {len(seen)} refusal statuses map to distinct flags "
              f"(NOT_IN_UNIVERSE != OUT_OF_TRAINING_UNIVERSE), OK is silent, "
              f"unknown status degrades to a stable code — PASS")

        # ---- The batch-failure gate must FIRE, not merely not-crash ---------------
        # This exists because the first real CI run reported success while dropping
        # 118 of 136 funds, so a test that only proves the happy path would reproduce
        # exactly the blindness it is meant to remove.
        assert error_rate_exceeded(artifact["coverage"], 0.25) is None, \
            "FAIL: a clean batch must pass the gate"
        fired = error_rate_exceeded(
            dict(n_total=136, n_errors=118,
                 errors=[{"query": "x", "error": "No such file: benchmark_availability.csv"}]), 0.25)
        assert fired and "118/136" in fired and "86.8%" in fired, fired
        # The real CI failure was a SINGLE systemic cause, so the reason must survive
        # into the message — a bare count is indistinguishable from ordinary attrition.
        assert "benchmark_availability.csv" in fired, fired
        # Boundary: exactly at the ceiling is acceptable, one fund past it is not.
        assert error_rate_exceeded(dict(n_total=100, n_errors=25, errors=[]), 0.25) is None
        assert error_rate_exceeded(dict(n_total=100, n_errors=26, errors=[]), 0.25)
        # A batch that attempted nothing is a failure, never a vacuous 0% error rate.
        assert error_rate_exceeded(dict(n_total=0, n_errors=0, errors=[]), 0.99)
        # ---- stalled-ledger gate --------------------------------------------
        # The real defect: TODAY frozen at 2026-07-13 pinned every anchor, so the
        # nightly appended nothing for weeks and still reported success.
        run_day = pd.Timestamp("2026-08-05")
        def _mon(last_anchor, appended=0):
            return dict(ledger=dict(rows_total=1161, rows_appended_this_run=appended,
                                    first_anchor="2026-07-13", last_anchor=last_anchor))
        stalled = ledger_stalled(_mon("2026-07-13"), run_day, 7)
        assert stalled and "stopped advancing" in stalled, stalled
        # ...but a same-day re-run appends nothing and MUST NOT fail: zero appended
        # is ordinary idempotence, and failing on it would break every manual rerun.
        assert ledger_stalled(_mon("2026-08-05", appended=0), run_day, 7) is None, \
            "FAIL: a fresh anchor with 0 appended rows is a re-run, not a stall"
        assert ledger_stalled(_mon("2026-07-29"), run_day, 7) is None, "FAIL: exact boundary"
        assert ledger_stalled(_mon("2026-07-28"), run_day, 7), "FAIL: one day past the boundary"
        # An empty ledger has nothing to be stale about, and must not fall back to
        # first_anchor — that never moves, so it would fail every healthy run later.
        assert ledger_stalled(dict(ledger=dict(first_anchor="2020-01-01")), run_day, 7) is None
        assert ledger_stalled({}, run_day, 7) is None
        print("[selftest] stalled-ledger gate fires on a pinned anchor, stays silent on "
              "a same-day re-run and an empty ledger — PASS")

        print("[selftest] batch-failure gate fires at 118/136 with the cause named, "
              "passes a clean batch, is exact at the boundary, and refuses an empty batch — PASS")

        # A NON-default profile must NOT be mislabeled as the default. Separate tmp
        # ledger path — a custom profile's predictions are still real cohort_q1
        # scores and shouldn't dedupe-collide with the default-profile run above.
        custom = InvestorProfile(horizon_years=3.0, liquidity_need="high", risk_appetite="conservative")
        custom_artifact = build_artifact(names, orch=orch, profile=custom,
                                         ledger_path=tmpdir / "predictions_custom.jsonl",
                                         realizations_path=tmpdir / "realizations_custom.jsonl")
        for rec in custom_artifact["funds"]:
            assert rec["signal_a"]["verdict_basis"]["is_default_profile"] is False, \
                "FAIL: a custom profile was mislabeled as the default"
        print("[selftest] custom (non-default) profile correctly NOT stamped is_default_profile — PASS")

        # A NaN anywhere in an alert's `evidence` must NOT be able to discard the whole
        # batch. Nothing shipped trips this today (alert authors route floats through
        # mf_sentinel._fmt()), but the convention isn't enforced — so prove the guard
        # holds structurally by feeding a hostile alert through the real record builder.
        import mf_sentinel
        ok_result = next(r for r in (orch.evaluate(n, profile_config=profile.model_dump(mode="json"))
                                     for n in names) if "error" not in r)
        ok_result["sentinel"].alerts.append(mf_sentinel.Alert(
            "TEST_HOSTILE_EVIDENCE", mf_sentinel.AlertSeverity.INFO.value, "factor",
            mf_sentinel.AlertBasis.DESCRIPTIVE_RISK.value,
            {"raw_nan": float("nan"), "raw_inf": float("inf"),
             "np32_nan": np.float32("nan"), "np32_ok": np.float32(1.5), "np_int": np.int32(7),
             "nested": {"deep_nan": float("nan"), "flag": True, "name": "ok"}},
            "hostile evidence fixture"))
        hostile_rec = build_fund_record(ok_result, profile, is_default_profile=True)
        ev = next(a["evidence"] for a in hostile_rec["alerts_b"] if a["code"] == "TEST_HOSTILE_EVIDENCE")
        assert ev["raw_nan"] is None and ev["raw_inf"] is None, ev
        assert ev["nested"]["deep_nan"] is None, ev
        # numpy scalars: json.dumps rejects np.float32/np.int32 outright, so an
        # unwrapped one is the same batch-wide failure by a different exception.
        assert ev["np32_nan"] is None and abs(ev["np32_ok"] - 1.5) < 1e-6 and ev["np_int"] == 7, ev
        # bools must survive as bools — isinstance(True, int) is True, so a naive
        # numeric guard would silently rewrite this as 1.0 and corrupt the contract.
        assert ev["nested"]["flag"] is True and ev["nested"]["name"] == "ok", ev
        json.dumps(dict(funds=[hostile_rec]), allow_nan=False)   # must not raise
        print("[selftest] hostile NaN/Inf in alert evidence is neutralised, bools preserved, "
              "batch write survives — PASS")

        # gzip round-trip
        path = write_artifact(artifact, tmpdir)
        with gzip.open(path) as fh:
            reloaded = json.load(fh)
        assert reloaded == artifact, "FAIL: gzip round-trip changed the artifact"
        print(f"[selftest] gzip round-trip ({path.stat().st_size} bytes) — PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("[selftest] PASS — artifact emitter produces an honest, schema-complete, "
          "round-trippable batch artifact from real cached data, ledger append is "
          "idempotent, and monitoring degrades honestly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="contract shape + honesty gates + gzip round-trip on 3 real funds")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N funds in the manifest (quick test)")
    ap.add_argument("--out", type=str, default=None, help="output directory")
    ap.add_argument("--max-error-rate", type=float, default=None,
                    help="exit non-zero if more than this FRACTION of funds failed to "
                         "evaluate (e.g. 0.25). Off by default; the scheduled job sets it "
                         "so a run that silently drops most of the universe cannot report "
                         "success. The artifact is still written first, for diagnosis.")
    ap.add_argument("--max-anchor-age-days", type=int, default=None,
                    help="exit non-zero if the NEWEST prediction anchor is more than this "
                         "many days behind the run date. Off by default; the scheduled job "
                         "sets it. Catches a pipeline that has stopped tracking time — a "
                         "frozen TODAY pinned every anchor for weeks while the nightly "
                         "reported success and appended nothing.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.selftest:
        _selftest()
    else:
        names = _load_scheme_names(args.limit)
        LOGGER.info("Scoring %d funds against the default investor profile...", len(names))
        artifact = build_artifact(names)
        path = write_artifact(artifact, Path(args.out) if args.out else OUT_DIR)
        LOGGER.info("Wrote %s (%d funds, %d errors) -> %s",
                   artifact["artifact_version"], artifact["coverage"]["n_ok"],
                   artifact["coverage"]["n_errors"], path)

        # A batch that drops most of the universe is a FAILED run, not a thin one.
        # The first real CI run emitted 18 of 136 funds and still reported success,
        # because nothing here ever returned non-zero — the artifact is written
        # before this check precisely so a failing run is still inspectable.
        # Zero appended rows is normal (a same-day re-run has nothing new to say),
        # so it is logged, not failed on. What IS reported loudly, every run.
        led = (artifact.get("monitoring") or {}).get("ledger") or {}
        LOGGER.info("Ledger: %s rows total, %s appended this run, newest anchor %s",
                    led.get("rows_total"), led.get("rows_appended_this_run"),
                    led.get("last_anchor"))
        if not led.get("rows_appended_this_run"):
            LOGGER.warning("No new prediction rows were appended — expected on a re-run, "
                           "but if it repeats across days the anchor has stopped advancing.")

        failures = []
        # A batch that drops most of the universe is a FAILED run, not a thin one.
        # The first real CI run emitted 18 of 136 funds and still reported success,
        # because nothing here ever returned non-zero — the artifact is written
        # before these checks precisely so a failing run is still inspectable.
        if args.max_error_rate is not None:
            failures.append(error_rate_exceeded(artifact["coverage"], args.max_error_rate))
        if args.max_anchor_age_days is not None:
            failures.append(ledger_stalled(artifact.get("monitoring") or {}, TODAY,
                                           args.max_anchor_age_days))
        failures = [f for f in failures if f]
        if failures:
            for reason in failures:
                LOGGER.error("FAILING: %s", reason)
            sys.exit(1)
