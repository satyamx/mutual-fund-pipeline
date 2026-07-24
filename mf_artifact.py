"""
====================================================================================
mf_artifact.py — versioned batch JSON artifact emitter (to-do #5)
====================================================================================
Runs MasterOrchestrator(live=True) over every fund in the curated universe
manifest and serializes the honest, per-fund SCREEN + Sentinel + cohort_signal
fields into ONE versioned, gzipped JSON artifact for the Hisaab Kitaab app to
ingest later (still gated "DO NOT START INTEGRATION YET" — this is the batch
producer side only). Never a fabricated number: every gap the live orchestrator
already reports (coverage_flags, NOT_IN_UNIVERSE/STALE_NAV/THIN_COHORT, missing
TER/AUM/manager) is carried through as `null` or a `data_flags` entry, not
guessed.

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
                      metric_colors, verdict_basis, cohort_q1_prob,
                      cohort_percentile, cohort_n, signal_context},
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
DATA_FLAG_OUT_OF_TRAINING_UNIVERSE = "OUT_OF_TRAINING_UNIVERSE"
DATA_FLAG_STALE_NAV = "STALE_NAV"
DATA_FLAG_SHORT_HISTORY = "SHORT_HISTORY"
DATA_FLAG_SECTOR_UNMAPPED = "SECTOR_UNMAPPED"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True,
                              timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 — a missing/broken git binary must not crash the batch run
        return "unknown"


def _finite_or_none(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None


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
        if status == "THIN_COHORT":
            flags.append(DATA_FLAG_THIN_COHORT)
        elif status == "STALE_NAV":
            flags.append(DATA_FLAG_STALE_NAV)
        elif status == "NOT_IN_UNIVERSE":
            flags.append(DATA_FLAG_OUT_OF_TRAINING_UNIVERSE)
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
            cohort_q1_prob=_finite_or_none(cs["probability"]) if cs else None,
            cohort_percentile=_finite_or_none(cs["cohort_percentile"]) if cs else None,
            cohort_n=cs["cohort_n"] if cs else None,
            signal_context=cs["signal_context"] if cs else None),
        alerts_b=[dataclasses.asdict(a) for a in sr.alerts],
        nfo_dossier=sr.nfo_dossier,
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
        n_out_of_training_universe=sum(
            1 for r in records if DATA_FLAG_OUT_OF_TRAINING_UNIVERSE in r["data_flags"]),
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
        print(f"[selftest] {len(artifact['funds'])} fund records: required keys present, "
              f"lock always null, verdict_basis stamped default-profile, "
              f"expense gap flagged not guessed, sub_scores NaN-safe — PASS")

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
