"""
====================================================================================
mf_ledger.py — append-only prediction ledger + monitoring (to-do #6)
====================================================================================
Design: Fable subagent pass (2026-07-19, flagged per G1 for subtle correctness:
avoiding leakage in "how do you monitor a model whose label needs 3 years to
realize", and picking a defensible drift-reference methodology). Implementation
+ verification here.

WHY A LEDGER AT ALL: `cohort_q1`'s label needs ~3y of forward NAV data to
realize (mf_labels.forward_return, min_yrs=2.75). You cannot know today whether
today's prediction was right — that can only be checked once ~3 years pass, and
only if you logged EXACTLY what was predicted, for EXACTLY which fund, against
EXACTLY which cohort peers, using EXACTLY which model. None of that can be
reconstructed after the fact (NAV data keeps arriving, cohorts reshuffle,
models retrain) — hence "monitoring is unbackfillable" and the ledger must
exist BEFORE deploy, logging from day one even though nothing can be realized
for years.

WHY GIT-TRACKED, NOT mf_cache/: mf_cache/ is gitignored and documented as
"generated, never committed — regenerate as needed" (CLAUDE.md). A prediction
log is the opposite of regenerable — it IS the historical record. It lives in
a plain `ledger/` directory at the repo root, tracked in git, so it survives a
fresh clone / a fresh CI runner with an empty mf_cache/.

WHY JSONL, NOT PARQUET: parquet has no safe cross-process append — "appending"
means read-whole-file, concat, rewrite, and a crash mid-rewrite corrupts every
previously-logged prediction (exactly what "unbackfillable" forbids). JSONL
appends are O(1): open "a", write one line, flush + fsync. A crash can at
worst leave one torn trailing line, which is trivially detected (fails
json.loads) and skipped by the reader; the next append seals any torn tail
with a newline first.

WHAT'S HONEST HERE, EXPLICITLY:
  realized_ic  — stays null (status PENDING_MATURITY / INSUFFICIENT_MATURED)
                until enough REAL 3y-matured outcomes exist (~2029 earliest,
                and only once >= MIN_MATURED_FOR_IC of them). No "interim"
                proxy computed against an untested horizon is ever published
                under this name — the model was trained/calibrated/holdout-
                validated against a 3y label ONLY; presenting a 1y/6mo
                correlation as a monitoring signal would implicitly claim
                validated skill at a horizon never measured. That is exactly
                "manufacturing a metric by loosening its definition."
  rank_stability — a SEPARATE, honestly-named QA proxy: does this fund's
                predicted probability stay reasonably consistent run-to-run?
                This claims only "the pipeline behaves sanely," never "the
                model is right" — it does not touch outcomes at all.
  psi          — drift of the LIVE scoring population's raw features against
                the TRAINING population's empirical (not assumed-normal)
                decile distribution. Computed fresh every run from the
                already-available live panel — no ledger dependency, no
                waiting required.
====================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from mf_benchmarks import forward_return
from mf_horizon import DEFAULT, HORIZONS
from mf_infer import DEFAULT_TARGET, MODEL_DIR
from mf_labels import COHORT_MIN_SIZE, load_manifest, load_nav_panel
from mf_live_score import build_live_panel, score_on_panel

LOGGER = logging.getLogger("MFLedger")

SCHEMA_VERSION = 1
LEDGER_DIR = Path(__file__).parent / "ledger"
# LAYOUT (docs/horizons.md §7). Every append goes to `<root>/<target>/<YYYY-MM of
# the row's anchor>.jsonl` (realizations: `<root>/<target>/realized/<YYYY-MM>.jsonl`),
# root = the parent of the path a caller passes. The two paths below are kept as
# that HANDLE and as the LEGACY single files: predictions.jsonl holds every 3y row
# logged before rotation (68 MiB, heading for GitHub's 100 MiB push limit) and is
# never rewritten, moved or appended to again; readers for cohort_q1 take it plus
# cohort_q1/, readers for any other target only that target's directory, so one
# horizon's rows can never reach another's monitoring. realizations.jsonl was never
# written (nothing had matured) and is read the same way, for symmetry.
PREDICTIONS_PATH = LEDGER_DIR / "predictions.jsonl"
REALIZATIONS_PATH = LEDGER_DIR / "realizations.jsonl"
_BY_TARGET = {h.target: h for h in HORIZONS.values()}  # maturity, divisor: looked up by a row's target
# Hand-maintained DECLARATION, not a derived file: which model_ids are retired.
# The predictions file is append-only precisely so a prediction can never be
# rewritten or deleted, which means "this payload was live for twenty minutes and
# should not be graded in 2029" cannot be expressed by editing it. It is expressed
# here instead: the rows stay exactly as written, and readers skip the models named
# in this file. Removing an entry un-retires the model, losing nothing.
SUPERSEDED_PATH = LEDGER_DIR / "superseded_models.json"
# Git-tracked alongside the shipped model (see mf_infer.MODEL_DIR): this is the
# empirical training-time decile distribution the live PSI drift check compares
# against, so it is only meaningful paired with the exact model it was built from.
# Left in an evictable cache it would silently vanish and turn every drift check
# into REFERENCE_MISSING — a monitoring blind spot that looks like "no drift".
PSI_REFERENCE_PATH = MODEL_DIR / "psi_reference.json"

# Raw, non-derived features only. Excluded (with reasons): z_* (9, within-anchor
# z-scores — mean~0/std~1 at every anchor by construction, mf_features.py
# assemble_panel; PSI would mostly measure small-pool noise); rank_3y (uniform
# by construction); has_* (5) + is_thematic (binary — decile PSI ill-posed;
# missingness is monitored directly via missing_rate below instead); age_years
# (verified empirically, 2026-07-19: the single biggest PSI driver on the very
# first live run, psi~3.2 — because training pools anchors 2013-2023 while any
# live anchor is simply years further on, every fund is mechanically older than
# the pooled historical distribution. This drift is guaranteed by the passage
# of time, not informative about the LIVE population being unusual, and would
# have permanently dominated psi_max/worst[] with a non-actionable signal).
PSI_FEATURES = [
    "cagr_1y",
    "cagr_3y",
    "cagr_5y",
    "vol_1y",
    "vol_3y",
    "sharpe_3y",
    "sortino_3y",
    "max_dd_3y",
    "current_dd",
    "mom_6m",
    "mom_12m_ex1m",
    "rr3y_neg_share",
    "excess_1y",
    "excess_3y",
    "beta_3y",
    "te_3y",
    "ir_3y",
    "upcap_3y",
    "downcap_3y",
    "excess_persist",
    "sector_mom_12m",
    "sector_vol_1y",
    "sector_rel_strength",
]
PSI_BINS = 10
PSI_MIN_UNIVERSE = 50  # below this, a decile histogram is noise, not a drift signal
PSI_MIN_FINITE = 30  # per-feature: skip rather than compute on a thin sample
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25  # standard PSI rule-of-thumb thresholds

# 3y's window, pinned == HORIZONS["3y"].window_end by mf_horizon's selftest; realization reads the registry
MATURITY = pd.DateOffset(years=3)
MONTH_END_WINDOW_DAYS = 10  # §7: a month-end anchor is logged only within this many days of it
MIN_MATURED_FOR_IC = 50
MIN_COMMON_FOR_STABILITY = 20

# model_id is part of the key on purpose. Without it, one fund+target+anchor can
# hold only ONE prediction ever, so the first model to log an anchor permanently
# blocks every later model at that anchor — which is precisely backwards: a model
# change is the one time you most want both predictions on record, side by side and
# distinguishable. Shipping phase_b_v2 over an anchor phase_b_v1 had already
# written silently dropped 117 of 348 v2 predictions before this was caught.
# mf_eval segments outcome history by model_id, so the two never pool into one
# accuracy claim; the key just has to let both exist.
DEDUPE_KEY = ("amfi_code", "target", "anchor", "model_id")


# ==============================================================================
# JSONL append / read — crash-safe, torn-tail tolerant
# ==============================================================================
def read_jsonl(path: Path) -> List[dict]:
    """Tolerates (and warns on) any unparseable line — a torn trailing line from
    a crash mid-write is the expected case, but a line is skipped regardless of
    its position, since a sealed torn line is no longer literally the last line
    once later appends continue past it."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    rows: List[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn write mid-append leaves a permanently-unparseable line in
            # place even after later appends seal the file and continue past
            # it (it's never the literal last line again) — skip unconditionally
            # rather than trying to distinguish "torn tail" from "corruption" by
            # position, which breaks the moment more valid data follows it.
            LOGGER.warning("Unparseable line in %s (%d chars) — skipping.", path, len(line))
    return rows


def _dedupe_key_of(row: dict, dedupe_key: tuple) -> Optional[tuple]:
    try:
        return tuple(row[k] for k in dedupe_key)
    except KeyError:
        # A row that parsed as valid JSON but is missing an expected field (e.g.
        # a future/older schema_version) must not crash the whole append — skip
        # it for dedupe purposes rather than letting one odd historical row take
        # down every subsequent batch run.
        LOGGER.warning("Ledger row missing dedupe key %s — treating as unmatchable: %r", dedupe_key, row)
        return None


def _target_files(handle: Path, target: str, sub: str = "") -> List[Path]:
    """The files holding `target`'s rows: the legacy single file (cohort_q1 only)
    first, then the target's rotated month files in order."""
    legacy = [handle] if target == DEFAULT_TARGET else []
    return legacy + sorted((handle.parent / target / sub).glob("*.jsonl"))


def _read_target(handle: Path, target: str, sub: str = "") -> List[dict]:
    return [r for f in _target_files(handle, target, sub) for r in read_jsonl(f)]


def append_predictions(
    rows: Sequence[dict], path: Path = PREDICTIONS_PATH, dedupe_key: tuple = DEDUPE_KEY, sub: str = ""
) -> int:
    """Append-only, idempotent on `dedupe_key` across the legacy file AND the
    rotated files, crash-safe (seals a torn tail before writing, fsyncs after).
    Each row goes to `path.parent/<target>/<sub>/<YYYY-MM of anchor>.jsonl`;
    `path` itself is only ever read. Returns the number of NEW rows written."""
    if not rows:
        return 0
    existing = {
        k
        for t in {r["target"] for r in rows}
        for k in (_dedupe_key_of(r, dedupe_key) for r in _read_target(path, t, sub))
        if k is not None
    }
    # Dedupe against the incoming batch too, not just the file. On 2026-08-03 a
    # resolver collision handed this function two rows with an identical
    # (amfi_code, target, anchor) inside ONE run; both were written, because
    # `existing` was a snapshot of the file taken before the loop. The resolver is
    # fixed, but "idempotent on dedupe_key" has to hold for the argument as well
    # as the file or it isn't a property of the ledger, only of repeated calls.
    seen = set(existing)
    to_write: List[dict] = []
    for r in rows:
        k = _dedupe_key_of(r, dedupe_key)
        if k is not None and k in seen:
            continue
        if k is not None:
            seen.add(k)
        to_write.append(r)
    if not to_write:
        return 0
    by_file: Dict[Path, List[dict]] = {}
    for r in to_write:
        by_file.setdefault(path.parent / r["target"] / sub / f"{str(r['anchor'])[:7]}.jsonl", []).append(r)
    for f, group in by_file.items():
        _append_lines(f, group)
    return len(to_write)


def _append_lines(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                with open(path, "a", encoding="utf-8") as fh2:
                    fh2.write("\n")
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_predictions(
    path: Path = PREDICTIONS_PATH, dedupe_key: tuple = DEDUPE_KEY, target: str = DEFAULT_TARGET
) -> List[dict]:
    """`target`'s rows (legacy file + rotated files, see LAYOUT) plus the ledger's
    own uniqueness invariant, enforced on READ.

    Every consumer of the prediction ledger wants "one row per prediction", which
    is what append_predictions promises. The 2026-08-03 run broke that promise
    (see the note in append_predictions) and left three duplicate pairs in a
    file that is append-only and git-tracked — deleting them would rewrite an
    unbackfillable history to hide a mistake, which is the opposite of what the
    ledger is for. So the bytes stay and readers dedupe: first occurrence wins,
    which is well-defined here because the affected pairs are identical except
    for `logged_at`. Rows with no extractable key are kept — they are already
    unmatchable and dropping them would lose data rather than restore an invariant.
    """
    rows = _read_target(path, target)
    seen, out, dropped = set(), [], 0
    for r in rows:
        k = _dedupe_key_of(r, dedupe_key)
        if k is None:
            out.append(r)
            continue
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        out.append(r)
    if dropped:
        LOGGER.info(
            "Prediction ledger: %d duplicate row(s) collapsed on %s (append-only file, deduped on read).",
            dropped,
            dedupe_key,
        )
    return out


def ledger_stats(path: Path = PREDICTIONS_PATH, target: str = DEFAULT_TARGET) -> Dict[str, Any]:
    rows = load_predictions(path, target=target)
    if not rows:
        return dict(rows_total=0, first_anchor=None, last_anchor=None, n_runs=0)
    anchors = sorted(r["anchor"] for r in rows)
    return dict(
        rows_total=len(rows), first_anchor=anchors[0], last_anchor=anchors[-1], n_runs=len({r["run_id"] for r in rows})
    )


# ==============================================================================
# Building a prediction row from a score_live() OK result
# ==============================================================================
def build_row(
    amfi_code: str, cohort_signal: Optional[dict], *, run_id: str, pipeline_sha: str, model_id: Optional[str]
) -> Optional[dict]:
    """None for any non-OK cohort_signal — a THIN_COHORT/STALE_NAV/NOT_IN_UNIVERSE
    result is not a prediction and can never be realized; those gaps already
    live honestly in the artifact's own coverage{}, not duplicated here."""
    if cohort_signal is None or cohort_signal.get("status") != "OK":
        return None
    row = dict(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        logged_at=datetime.now(timezone.utc).isoformat(),
        pipeline_sha=pipeline_sha,
        model_id=model_id,
        target=cohort_signal["target"],
        amfi_code=amfi_code,
        anchor=cohort_signal["anchor"],
        probability=cohort_signal["probability"],
        cohort_percentile=cohort_signal["cohort_percentile"],
        cohort_key=cohort_signal["cohort_key"],
        cohort_n=cohort_signal["cohort_n"],
        universe_n=cohort_signal["universe_n"],
        cohort_codes=cohort_signal["cohort_codes"],
        signal_context=cohort_signal["signal_context"],
        imputed_features=cohort_signal["imputed_features"],
        features=cohort_signal["features"],
    )
    canonical = json.dumps(
        dict(
            amfi_code=amfi_code,
            target=row["target"],
            anchor=row["anchor"],
            model_id=model_id,
            probability=row["probability"],
            features=row["features"],
        ),
        sort_keys=True,
    )
    row["row_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return row


# ==============================================================================
# rank_stability — a model-behavior QA proxy, NOT a skill/outcome metric
# ==============================================================================
def _previous_run_rows(all_rows: List[dict], current_run_id: str, model_id: Optional[str] = None) -> List[dict]:
    """The newest run OTHER than this one, optionally restricted to one model.

    Restricting matters: this feeds rank_stability, which reports run-to-run
    consistency. Without it, the run before a model rollout is the PREVIOUS
    model's, so an intended model change is graded as pipeline instability.
    """
    rows = [r for r in all_rows if model_id is None or r.get("model_id") == model_id]
    other_runs = sorted({r["run_id"] for r in rows if r["run_id"] != current_run_id})
    if not other_runs:
        return []
    prev_run_id = other_runs[-1]
    return [r for r in rows if r["run_id"] == prev_run_id]


def rank_stability(current_rows: List[dict], previous_rows: List[dict]) -> Dict[str, Any]:
    note = "model-stability QA (run-to-run consistency) — not a skill metric"
    if not previous_rows:
        return dict(status="FIRST_RUN", spearman=None, n_common=0, prev_run_id=None, note=note)
    # model_id is in the key so a probability is never compared against one from a
    # different model — that would grade an intended model change as instability.
    prev_by_key = {(r["amfi_code"], r["target"], r.get("model_id")): r["probability"] for r in previous_rows}
    cur, prev = [], []
    for r in current_rows:
        key = (r["amfi_code"], r["target"], r.get("model_id"))
        if key in prev_by_key:
            cur.append(r["probability"])
            prev.append(prev_by_key[key])
    if len(cur) < MIN_COMMON_FOR_STABILITY:
        return dict(
            status="INSUFFICIENT_OVERLAP",
            spearman=None,
            n_common=len(cur),
            prev_run_id=previous_rows[0]["run_id"],
            note=note,
        )
    spearman = float(pd.Series(cur).rank().corr(pd.Series(prev).rank()))
    return dict(
        status="OK", spearman=round(spearman, 4), n_common=len(cur), prev_run_id=previous_rows[0]["run_id"], note=note
    )


# ==============================================================================
# PSI (population stability index) — empirical training deciles as reference
# ==============================================================================
def _decile_edges(values: np.ndarray, bins: int = PSI_BINS) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size < PSI_MIN_FINITE:
        return np.array([])
    edges = np.unique(np.nanquantile(finite, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return np.array([])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bin_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or edges.size < 2:
        return np.array([])
    idx = np.clip(np.searchsorted(edges, finite, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return counts / counts.sum()


def build_psi_reference(out: Optional[Path] = None, horizon: str = DEFAULT) -> dict:
    """Training-time only: empirical decile edges + reference bin proportions
    per monitored feature, over the cohort model's ACTUAL training rows (same
    filter mf_model.load_cohort_dataset applies). Rebuild whenever the cohort
    model retrains — stamped with that artifact's version so monitoring-time
    code can detect a stale reference rather than silently comparing against
    the wrong model's training population. Per horizon: its own training rows
    (mf_cache/phase_b/<key>/), artifact and output (model/psi_reference_<key>.json)."""
    import mf_model as M  # sklearn-adjacent imports stay out of the monitoring/scoring path

    h = HORIZONS[horizon]
    out = out or h.psi_reference_path
    df, _ = M.load_cohort_dataset(h)
    artifact = json.loads(h.artifact_path.read_text(encoding="utf-8"))
    ref: Dict[str, Any] = dict(
        model_id=artifact["version"], created=datetime.now(timezone.utc).isoformat(), features={}
    )
    for col in PSI_FEATURES:
        vals = df[col].to_numpy(dtype=float)
        edges = _decile_edges(vals)
        if edges.size < 2:
            continue
        ref["features"][col] = dict(
            edges=[float(e) for e in edges],
            bin_props=[float(p) for p in _bin_proportions(vals, edges)],
            missing_rate=float(np.mean(~np.isfinite(vals))),
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return ref


def compute_psi(live_features: Dict[str, np.ndarray], reference: dict) -> Dict[str, Any]:
    """Pure function — unit-testable without any live orchestrator/NAV state."""
    per_feature: Dict[str, float] = {}
    skipped: List[str] = []
    for col, ref_f in reference.get("features", {}).items():
        vals = live_features.get(col)
        if vals is None:
            skipped.append(col)
            continue
        vals = np.asarray(vals, dtype=float)
        if np.isfinite(vals).sum() < PSI_MIN_FINITE:
            skipped.append(col)
            continue
        edges = np.array(ref_f["edges"])
        cur = np.clip(_bin_proportions(vals, edges), 1e-4, None)
        ref_p = np.clip(np.array(ref_f["bin_props"]), 1e-4, None)
        per_feature[col] = float(np.sum((cur - ref_p) * np.log(cur / ref_p)))
    if not per_feature:
        return dict(
            status="INSUFFICIENT_DATA", psi_max=None, psi_mean=None, worst=[], n_features=0, skipped_features=skipped
        )
    psi_max = max(per_feature.values())
    status = "SIGNIFICANT_SHIFT" if psi_max > PSI_SIGNIFICANT else "MODERATE_SHIFT" if psi_max > PSI_MODERATE else "OK"
    worst = sorted(per_feature.items(), key=lambda kv: -kv[1])[:3]
    return dict(
        status=status,
        psi_max=round(psi_max, 4),
        psi_mean=round(float(np.mean(list(per_feature.values()))), 4),
        worst=[dict(feature=f, psi=round(v, 4)) for f, v in worst],
        n_features=len(per_feature),
        skipped_features=skipped,
    )


def compute_psi_live(
    manifest: pd.DataFrame,
    nav_panel: Dict[str, pd.Series],
    engine,
    today: pd.Timestamp,
    model_id: Optional[str],
    reference_path: Path = PSI_REFERENCE_PATH,
) -> Dict[str, Any]:
    """Builds ONE fresh live panel at the universe's latest common anchor and
    compares it to the shipped training reference — no ledger needed; this is
    computable from day one, unlike realized_ic."""
    if not reference_path.exists():
        return dict(
            status="REFERENCE_MISSING", psi_max=None, psi_mean=None, worst=[], n_features=0, skipped_features=[]
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if model_id is not None and reference.get("model_id") != model_id:
        return dict(
            status="REFERENCE_STALE",
            psi_max=None,
            psi_mean=None,
            worst=[],
            n_features=0,
            skipped_features=[],
            reference_model_id=reference.get("model_id"),
        )

    last_dates = [s.dropna().index.max() for s in nav_panel.values() if len(s.dropna())]
    if not last_dates:
        return dict(
            status="INSUFFICIENT_DATA",
            psi_max=None,
            psi_mean=None,
            worst=[],
            n_features=0,
            skipped_features=[],
            universe_n=0,
        )
    t = min(max(last_dates), pd.Timestamp(today).normalize())
    _, panel = build_live_panel(manifest, nav_panel, t, engine=engine)
    if len(panel) < PSI_MIN_UNIVERSE:
        return dict(
            status="INSUFFICIENT_DATA",
            psi_max=None,
            psi_mean=None,
            worst=[],
            n_features=0,
            skipped_features=[],
            universe_n=len(panel),
        )

    live_features = {c: panel[c].to_numpy(dtype=float) for c in PSI_FEATURES if c in panel.columns}
    result = compute_psi(live_features, reference)
    result["universe_n"] = len(panel)
    result["reference_model_id"] = reference.get("model_id")
    return result


# ==============================================================================
# Realization — an OFFLINE job, never part of the live batch. Joins matured
# ledger rows against fresh NAV using the EXACT mf_labels.add_cohort_targets
# math, over the FROZEN cohort_codes logged at prediction time.
# ==============================================================================
def _realize_one(row: dict, nav_panel: Dict[str, pd.Series]) -> dict:
    """The row's own horizon (registry, by `target`): its window end, minimum
    window and tolerance — the exact forward_return call mf_labels trained on."""
    h = _BY_TARGET[row["target"]]
    anchor = pd.Timestamp(row["anchor"])
    kw = dict(tol_days=h.tol_days, min_yrs=h.min_window_years, end=h.window_end(anchor))
    own_series = nav_panel.get(row["amfi_code"])
    own_fwd = forward_return(own_series, anchor, **kw) if own_series is not None else None

    peer_fwds: List[float] = []
    for peer in row["cohort_codes"]:
        s = nav_panel.get(peer)
        fr = forward_return(s, anchor, **kw) if s is not None else None
        if fr is not None:
            peer_fwds.append(fr.value)

    if own_fwd is None:
        status, y_realized = "DEAD_OR_GAP", None
    elif len(peer_fwds) < COHORT_MIN_SIZE:
        status, y_realized = "THIN_AT_REALIZATION", None
    else:
        q75 = float(np.quantile(peer_fwds, 0.75))
        status, y_realized = "REALIZED", bool(own_fwd.value >= q75)

    return dict(
        schema_version=SCHEMA_VERSION,
        amfi_code=row["amfi_code"],
        target=row["target"],
        anchor=row["anchor"],
        # Carried through so an outcome can be attributed to the model that made
        # the call. Two models may predict the same fund at the same anchor against
        # DIFFERENT frozen cohorts, so even y_realized can differ between them —
        # pooling the outcomes would average two different questions.
        model_id=row.get("model_id"),
        realized_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        y_realized=y_realized,
        own_r_fwd=own_fwd.value if own_fwd is not None else None,
        n_peer_r_fwd=len(peer_fwds),
        probability=row["probability"],
        cohort_percentile=row["cohort_percentile"],
    )


def realize_matured_predictions(
    *,
    today: Optional[pd.Timestamp] = None,
    predictions_path: Path = PREDICTIONS_PATH,
    realizations_path: Path = REALIZATIONS_PATH,
    manifest: Optional[pd.DataFrame] = None,
    nav_panel: Optional[Dict[str, pd.Series]] = None,
    superseded_path: Path = SUPERSEDED_PATH,
    target: str = DEFAULT_TARGET,
) -> Dict[str, Any]:
    """Offline job — run periodically (e.g. monthly cron), NOT from the live
    batch path. Idempotent: already-realized rows are skipped, and so are
    predictions from any model named in `superseded_path`. One target per call;
    outcomes append to `realizations_path.parent/<target>/realized/<YYYY-MM>.jsonl`."""
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.now().normalize()
    manifest = manifest if manifest is not None else load_manifest()
    nav_panel = nav_panel if nav_panel is not None else load_nav_panel(manifest)

    predictions = load_predictions(predictions_path, target=target)
    # Realization rows written before model_id joined DEDUPE_KEY have no such
    # field, so keying them strictly would drop them from `already` and realize
    # every one of them a second time. Match those on the legacy 3-tuple as well.
    _legacy = ("amfi_code", "target", "anchor")
    already, already_legacy = set(), set()
    for r in load_realizations(realizations_path, target):
        k = _dedupe_key_of(r, DEDUPE_KEY)
        if k is not None:
            already.add(k)
        else:
            lk = _dedupe_key_of(r, _legacy)
            if lk is not None:
                already_legacy.add(lk)

    retired = load_superseded(superseded_path)
    new_rows: List[dict] = []
    n_pending = n_superseded = n_unknown = 0
    for row in predictions:
        # A retired payload's predictions stay in the file as a record of what was
        # actually emitted, but they are never graded: realizing a model that was
        # live for twenty minutes would put its outcomes in the same pool as the
        # model people actually relied on.
        if row.get("model_id") in retired:
            n_superseded += 1
            continue
        # Maturity is the row's own horizon's. A target the registry does not know
        # has no window to wait for and no label to compute — counted, never guessed.
        h = _BY_TARGET.get(row.get("target"))
        if h is None:
            n_unknown += 1
            continue
        key = _dedupe_key_of(row, DEDUPE_KEY)
        if key is not None and key in already:
            continue
        legacy_key = _dedupe_key_of(row, _legacy)
        if legacy_key is not None and legacy_key in already_legacy:
            continue
        # Only once the fund's NAV reaches the window end, or the end is more than
        # the horizon's tolerance in the past: a job running on the 1st would
        # otherwise close a 1m window on a NAV a day or two short of it, for good.
        end = h.window_end(pd.Timestamp(row["anchor"]))
        own = nav_panel.get(row["amfi_code"])
        reached = own is not None and len(own.dropna()) > 0 and own.dropna().index.max() >= end
        if today < end or (not reached and (today - end).days <= h.tol_days):
            n_pending += 1
            continue
        new_rows.append(_realize_one(row, nav_panel))
    if n_superseded:
        LOGGER.info("Skipped %d prediction(s) from superseded model(s): %s", n_superseded, ", ".join(sorted(retired)))
    if n_unknown:
        LOGGER.warning("Skipped %d prediction(s) whose target is not in mf_horizon.HORIZONS.", n_unknown)

    n_appended = append_predictions(new_rows, path=realizations_path, dedupe_key=DEDUPE_KEY, sub="realized")
    return dict(
        n_predictions=len(predictions),
        n_pending=n_pending,
        n_matured_this_run=len(new_rows),
        n_newly_realized=n_appended,
        n_superseded_skipped=n_superseded,
        n_unknown_target=n_unknown,
    )


def load_realizations(path: Path = REALIZATIONS_PATH, target: str = DEFAULT_TARGET) -> List[dict]:
    """`target`'s realized outcomes: the legacy file (cohort_q1 only) + its rotated files."""
    return _read_target(path, target, "realized")


def load_superseded(path: Path = SUPERSEDED_PATH) -> Dict[str, dict]:
    """`{model_id: {superseded_at, superseded_by, reason}}`. Absent file = none.

    A malformed file degrades to "nothing is superseded" rather than raising:
    the failure mode of a bad edit here should be that retired models are graded
    again (visible, recoverable), never that a batch run dies.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Unreadable %s (%s) — treating no model as superseded.", path, exc)
        return {}
    if not isinstance(data, dict):
        LOGGER.warning("%s is not an object — treating no model as superseded.", path)
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}


def current_model_id(
    predictions_path: Path = PREDICTIONS_PATH, superseded_path: Path = SUPERSEDED_PATH, target: str = DEFAULT_TARGET
) -> Optional[str]:
    """The model_id of the most recent NON-superseded prediction run.

    Outcome reporting is scoped to ONE model. The ledger deliberately holds
    several series for the same fund and anchor (that is why model_id is in
    DEDUPE_KEY), so anything that pools them reports an accuracy figure
    describing no model that exists. A retired model must not be that one.
    """
    rows = load_predictions(predictions_path, target=target)
    retired = load_superseded(superseded_path)
    live = [r for r in rows if r.get("model_id") not in retired]
    if not live:
        return None
    latest = max(live, key=lambda r: str(r.get("run_id") or ""))
    return latest.get("model_id")


def realized_summary(
    realizations_path: Path = REALIZATIONS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    model_id: Optional[str] = None,
    superseded_path: Path = SUPERSEDED_PATH,
    target: str = DEFAULT_TARGET,
) -> Dict[str, Any]:
    """`model_id` scopes every count and the IC to one model; None means "the
    model behind the newest predictions". Pooling models here would average two
    different questions — each row froze its own cohort_codes, so even
    y_realized can differ between models for the same fund and anchor."""
    h = _BY_TARGET[target]
    preds = load_predictions(predictions_path, target=target)
    model_id = model_id or current_model_id(predictions_path, superseded_path, target)
    if model_id is not None:
        preds = [r for r in preds if r.get("model_id") == model_id]
    rows = [r for r in load_realizations(realizations_path, target) if model_id in (None, r.get("model_id"))]
    realized = [r for r in rows if r["status"] == "REALIZED"]
    dead = sum(1 for r in rows if r["status"] == "DEAD_OR_GAP")
    thin = sum(1 for r in rows if r["status"] == "THIN_AT_REALIZATION")

    earliest_maturity = None
    if preds:
        first_anchor = min(pd.Timestamp(r["anchor"]) for r in preds)
        earliest_maturity = str(h.window_end(first_anchor).date())

    if not rows:
        status, value = "PENDING_MATURITY", None
    elif len(realized) < MIN_MATURED_FOR_IC:
        status, value = "INSUFFICIENT_MATURED", None
    else:
        probs = pd.Series([r["probability"] for r in realized])
        ys = pd.Series([float(r["y_realized"]) for r in realized])
        status, value = "OK", round(float(probs.rank().corr(ys.rank())), 4)

    return dict(
        status=status,
        value=value,
        n_matured=len(rows),
        n_realized=len(realized),
        n_dead_or_gap=dead,
        n_thin_at_realization=thin,
        effective_n=round(len(realized) / h.divisor) if realized else 0,
        earliest_maturity=earliest_maturity,
        model_id=model_id,
    )


def log_month_end(
    *,
    run_id: str,
    pipeline_sha: str,
    manifest: pd.DataFrame,
    nav_panel: Dict[str, pd.Series],
    today: pd.Timestamp,
    path: Path = PREDICTIONS_PATH,
    inferencers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Month-end rows for every non-3y horizon (docs/horizons.md §7) — the forward
    test samples what the holdout sampled. M = the latest month-end <= the newest
    NAV in `nav_panel`. A target already holding rows at M for its model id is
    done; else, if `today` is within MONTH_END_WINDOW_DAYS of M, ONE panel is built
    as of M (NAV <= M only — nothing after M can reach a row) and every horizon
    scores it: one row per fund that is OK and whose own as-of NAV is within that
    horizon's tolerance of M, anchor = M. Later than that the month is skipped and
    counted, never backfilled. 3y keeps its daily path. `inferencers`: optional
    {horizon key: CohortInferencer}, else the process singletons.
    Returns {target: {anchor, appended, skipped_funds, skipped_month, already_logged}}."""
    from pandas.tseries.offsets import MonthEnd

    from mf_infer import load_default

    today = pd.Timestamp(today).normalize()
    newest = max(s.dropna().index.max() for s in nav_panel.values() if len(s.dropna()))
    M = MonthEnd().rollback(newest.normalize())
    anchor = str(M.date())
    out: Dict[str, Dict[str, Any]] = {}
    todo = []
    for h in HORIZONS.values():
        if h.legacy:
            continue
        # rows at anchor M can only live in M's month file (rotation is by anchor month)
        month_file = path.parent / h.target / f"{anchor[:7]}.jsonl"
        done = any(r.get("anchor") == anchor and r.get("model_id") == h.model_id for r in read_jsonl(month_file))
        late = (today - M).days > MONTH_END_WINDOW_DAYS
        out[h.target] = dict(
            anchor=anchor, appended=0, skipped_funds=0, skipped_month=int(late and not done), already_logged=done
        )
        if not done and not late:
            todo.append(h)
    if not todo:
        return out

    as_of = {c: s.loc[:M] for c, s in nav_panel.items()}
    engine, panel = build_live_panel(manifest, as_of, M)
    last = {c: s.dropna().index.max() for c, s in as_of.items() if len(s.dropna())}
    for h in todo:
        inf = (inferencers or {}).get(h.key) or load_default(h.key)
        rows, skipped = [], 0
        for code in manifest["amfi_code"]:
            stale = (M - last[code]).days if code in last else None
            ok = stale is not None and stale <= h.tol_days  # the fund's OWN as-of NAV, this horizon's tolerance
            cs = score_on_panel(code, engine, panel, inf, M, stale) if ok else None
            row = build_row(code, cs, run_id=run_id, pipeline_sha=pipeline_sha, model_id=h.model_id)
            if row is None:
                skipped += 1
            else:
                rows.append(row)
        out[h.target].update(appended=append_predictions(rows, path=path), skipped_funds=skipped)
    return out


# ==============================================================================
# monitoring{} assembly — the single entry point mf_artifact.py calls
# ==============================================================================
def monitoring_block(
    *,
    run_id: str,
    rows_appended: int,
    current_rows: List[dict],
    psi: dict,
    predictions_path: Path = PREDICTIONS_PATH,
    realizations_path: Path = REALIZATIONS_PATH,
    target: str = DEFAULT_TARGET,
) -> Dict[str, Any]:
    """Scoped to ONE target, cohort_q1 by default — the frozen mf_artifact_v1 block.
    `current_rows`: the rows THIS run scored (in memory, before dedup) — NOT
    re-derived by looking up run_id in the ledger file. Rows unchanged from a
    prior run are deliberately NOT re-appended (append-only + dedupe), so they
    never carry the new run_id in the file; looking them up by run_id there
    would find nothing on an ordinary no-op rerun and starve rank_stability."""
    all_rows = load_predictions(predictions_path, target=target)
    stats = ledger_stats(predictions_path, target)
    # Scope the comparison to the model this run actually scored with, so the
    # previous run of a DIFFERENT model is never mistaken for a previous run.
    this_model = next((r.get("model_id") for r in current_rows if r.get("model_id")), None)
    previous = _previous_run_rows(all_rows, run_id, model_id=this_model)
    return dict(
        # last_anchor is what tells you the pipeline is still tracking time.
        # first_anchor never moves once the ledger is seeded, so it can only ever
        # answer "when did this start", never "is this still alive".
        ledger=dict(
            rows_total=stats["rows_total"],
            rows_appended_this_run=rows_appended,
            first_anchor=stats["first_anchor"],
            last_anchor=stats["last_anchor"],
            path=str(predictions_path),
        ),
        realized_ic=realized_summary(realizations_path, predictions_path, target=target),
        psi=psi,
        rank_stability=rank_stability(current_rows, previous),
        note="realized_ic is structurally unmeasurable before the first logged anchor + "
        f"~{_BY_TARGET[target].key}; no interim proxy is published (untested horizon).",
    )


# ==============================================================================
# SELFTEST
# ==============================================================================
def _selftest() -> None:
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Every case gets its own root: appends rotate into <root>/<target>/, so two
        # cases sharing a directory would read each other's rows.
        pred_path = tmpdir / "predictions.jsonl"
        real_path = tmpdir / "realizations.jsonl"

        # ---- append/read round-trip + idempotency --------------------------
        cs_ok = dict(
            status="OK",
            target="cohort_q1",
            probability=0.4,
            cohort_percentile=0.6,
            cohort_key="('category', 'Flexi Cap')",
            cohort_n=10,
            universe_n=130,
            anchor="2020-01-31",
            cohort_codes=["A", "B", "C", "D"],
            signal_context={"holdout_auc": 0.578, "base_rate": 0.323},
            imputed_features=[],
            features={"cagr_1y": 0.1, "vol_1y": None},
        )
        row = build_row("A", cs_ok, run_id="run1", pipeline_sha="deadbeef", model_id="test_v1")
        assert row is not None and row["amfi_code"] == "A" and "row_hash" in row
        assert build_row("A", dict(status="THIN_COHORT"), run_id="run1", pipeline_sha="x", model_id="m") is None
        n = append_predictions([row], path=pred_path)
        assert n == 1
        n2 = append_predictions([row], path=pred_path)
        assert n2 == 0, "FAIL: re-appending the same row should be a no-op (dedupe)"
        assert len(_read_target(pred_path, "cohort_q1")) == 1
        print("[selftest] append/read round-trip + idempotency — PASS")

        # ---- idempotency WITHIN one batch, not just across calls -------------
        # The 2026-08-03 run passed two rows with the same (amfi_code, target,
        # anchor) in a single call and wrote both, because `existing` was a
        # snapshot of the file taken before the loop.
        dup_path = tmpdir / "intra" / "predictions.jsonl"
        twin = dict(row, logged_at="2026-08-03T19:22:26+00:00")  # same key, later stamp
        n_dup = append_predictions([row, twin], path=dup_path)
        assert n_dup == 1, f"FAIL: one key twice in ONE batch must write once, wrote {n_dup}"
        assert len(_read_target(dup_path, "cohort_q1")) == 1
        print("[selftest] intra-batch dedupe: same key twice in one call writes once — PASS")

        # ---- load_predictions collapses duplicates already on disk ----------
        # Restoring the invariant on read, because the historical duplicates are
        # in an append-only git-tracked file and deleting them would hide the bug.
        legacy_path = tmpdir / "dupes" / "predictions.jsonl"
        legacy_path.parent.mkdir()
        with open(legacy_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.write(json.dumps(twin, sort_keys=True) + "\n")
        assert len(read_jsonl(legacy_path)) == 2, "FAIL: raw read must not hide the bytes"
        loaded = load_predictions(legacy_path)
        assert len(loaded) == 1, f"FAIL: load_predictions must dedupe, got {len(loaded)}"
        assert loaded[0]["logged_at"] == row["logged_at"], "FAIL: first occurrence should win"
        print("[selftest] load_predictions collapses on-disk duplicates, keeps the bytes — PASS")

        # ---- a NEW model at an anchor an OLD model already logged --------------
        # The shipping bug this locks: with model_id outside the dedupe key, the
        # first model to write an anchor blocked every later one, and rolling out
        # phase_b_v2 silently dropped 117 of 348 predictions.
        two_model_path = tmpdir / "two" / "predictions.jsonl"
        v1 = build_row("A", cs_ok, run_id="r1", pipeline_sha="x", model_id="phase_b_v1_cohort")
        v2 = build_row("A", cs_ok, run_id="r2", pipeline_sha="x", model_id="phase_b_v2_cohort")
        assert append_predictions([v1], path=two_model_path) == 1
        assert append_predictions([v2], path=two_model_path) == 1, (
            "FAIL: a second model at the same (fund, target, anchor) must still be logged"
        )
        assert append_predictions([v2], path=two_model_path) == 0, (
            "FAIL: re-appending the same model's row must still be a no-op"
        )
        got = load_predictions(two_model_path)
        assert len(got) == 2 and {r["model_id"] for r in got} == {"phase_b_v1_cohort", "phase_b_v2_cohort"}, got
        print("[selftest] two models may log the same fund+anchor, and only re-runs dedupe — PASS")

        # ---- outcome reporting is scoped to ONE model ------------------------
        # Pooling model_ids would compute one lift/IC over a mixture of models,
        # describing none of them — and would trip MIN_MATURED_FOR_IC early.
        # `none_sup` keeps this hermetic: without it the assertion reads the REAL
        # ledger/superseded_models.json and changes meaning as models retire.
        none_sup = tmpdir / "no_superseded.json"  # never created
        assert current_model_id(two_model_path, none_sup) == "phase_b_v2_cohort", (
            "FAIL: current_model_id must follow the newest run, not the first row"
        )
        scoped = tmpdir / "two" / "realizations.jsonl"
        with open(scoped, "w", encoding="utf-8") as fh:
            for mid, yreal in (("phase_b_v1_cohort", True), ("phase_b_v2_cohort", False)):
                fh.write(
                    json.dumps(
                        dict(
                            schema_version=SCHEMA_VERSION,
                            amfi_code="A",
                            target="cohort_q1",
                            anchor="2020-01-31",
                            model_id=mid,
                            status="REALIZED",
                            y_realized=yreal,
                            probability=0.4,
                            cohort_percentile=0.9,
                            own_r_fwd=0.1,
                            n_peer_r_fwd=9,
                        )
                    )
                    + "\n"
                )
        summ = realized_summary(scoped, two_model_path, superseded_path=none_sup)
        assert summ["model_id"] == "phase_b_v2_cohort" and summ["n_realized"] == 1, (
            f"FAIL: realized_summary must scope to one model, got {summ}"
        )
        assert realized_summary(scoped, two_model_path, model_id="phase_b_v1_cohort")["n_realized"] == 1
        print("[selftest] realized_summary scopes to one model_id, never pools — PASS")

        # ---- rank_stability compares runs, not models ------------------------
        v1_run = [dict(v1, run_id="r1")]
        v2_run = [dict(v2, run_id="r2")]
        prev = _previous_run_rows(v1_run + v2_run, "r2", model_id="phase_b_v2_cohort")
        assert prev == [], "FAIL: the previous run of a DIFFERENT model must not count as a previous run"
        assert rank_stability(v2_run, prev)["status"] == "FIRST_RUN", (
            "FAIL: a model's first run must report FIRST_RUN, not a cross-model comparison"
        )
        print("[selftest] rank_stability never compares across model_id — PASS")

        # ---- legacy realization rows are not realized twice ------------------
        legacy_real = tmpdir / "legreal" / "realizations.jsonl"
        legacy_real.parent.mkdir()
        with open(legacy_real, "w", encoding="utf-8") as fh:  # pre-model_id schema
            fh.write(
                json.dumps(
                    dict(
                        amfi_code="A",
                        target="cohort_q1",
                        anchor="2020-01-31",
                        status="REALIZED",
                        y_realized=True,
                        probability=0.4,
                        cohort_percentile=0.9,
                    )
                )
                + "\n"
            )
        res = realize_matured_predictions(
            today=pd.Timestamp("2099-01-01"),
            predictions_path=two_model_path,
            realizations_path=legacy_real,
            manifest=pd.DataFrame(),
            nav_panel={},
        )
        assert res["n_matured_this_run"] == 0, (
            f"FAIL: a pre-model_id realization row must still count as already realized, got {res}"
        )
        print("[selftest] legacy realization rows are matched, not re-realized — PASS")

        # ---- superseded models: rows retained, never graded -------------------
        sup_path = tmpdir / "superseded.json"
        sup_path.write_text(json.dumps({"phase_b_v2_cohort": {"superseded_by": "phase_b_v3_cohort"}}), encoding="utf-8")
        assert current_model_id(two_model_path, sup_path) == "phase_b_v1_cohort", (
            "FAIL: a superseded model must never be the model outcomes are scoped to"
        )
        empty_real = tmpdir / "sup" / "realizations.jsonl"
        res = realize_matured_predictions(
            today=pd.Timestamp("2099-01-01"),
            predictions_path=two_model_path,
            realizations_path=empty_real,
            manifest=pd.DataFrame(),
            nav_panel={},
            superseded_path=sup_path,
        )
        assert res["n_superseded_skipped"] == 1, f"FAIL: superseded row not skipped: {res}"
        # ...and the rows themselves are untouched — this is a declaration, not a delete.
        assert len(load_predictions(two_model_path)) == 2, (
            "FAIL: superseding must not remove rows from the append-only ledger"
        )
        # A malformed declaration must degrade to "nothing superseded", never raise.
        bad = tmpdir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_superseded(bad) == {} and load_superseded(tmpdir / "absent.json") == {}
        print("[selftest] superseded models are skipped for grading, rows retained — PASS")

        # ---- torn-tail recovery ---------------------------------------------
        # In the month file appends actually go to — the legacy file is never written.
        month_file = tmpdir / "cohort_q1" / "2020-01.jsonl"
        with open(month_file, "a", encoding="utf-8") as fh:
            fh.write('{"amfi_code": "B", "target": "cohort_q1"')  # deliberately truncated
        recovered = read_jsonl(month_file)
        assert len(recovered) == 1, "FAIL: torn trailing line should be skipped, not crash"
        row2 = build_row("B", cs_ok, run_id="run1", pipeline_sha="deadbeef", model_id="test_v1")
        n3 = append_predictions([row2], path=pred_path)
        assert n3 == 1
        rows_after = read_jsonl(month_file)
        assert len(rows_after) == 2 and {r["amfi_code"] for r in rows_after} == {"A", "B"}
        print("[selftest] torn-tail recovery: seals + appends correctly — PASS")

        # ---- malformed-but-valid-JSON row must not crash dedupe -------------
        malformed_path = tmpdir / "malformed" / "predictions.jsonl"
        malformed_path.parent.mkdir()
        with open(malformed_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"amfi_code": "X", "no_target_or_anchor_here": True}) + "\n")
        n4 = append_predictions([row], path=malformed_path)  # 'row' has a real dedupe key
        assert n4 == 1, "FAIL: a malformed pre-existing row must not block appending a valid new one"
        print("[selftest] malformed row (missing dedupe key) doesn't crash append — PASS")

        # ---- maturity gate + honest degradation (THIN_AT_REALIZATION, DEAD) -
        far_future_row = dict(row2, amfi_code="C", anchor="2026-01-01")
        append_predictions([far_future_row], path=pred_path)
        summary = realize_matured_predictions(
            today=pd.Timestamp("2026-06-01"),
            predictions_path=pred_path,
            realizations_path=real_path,
            manifest=load_manifest(),
            nav_panel={},
        )  # empty nav_panel -> all matured rows degrade honestly
        assert summary["n_pending"] == 1, "FAIL: 2026-01-01 anchor should still be PENDING in mid-2026"
        realized_rows = read_jsonl(real_path)
        assert all(r["status"] == "DEAD_OR_GAP" for r in realized_rows), (
            "FAIL: with no NAV data at all, every matured row must degrade to DEAD_OR_GAP, never guessed"
        )
        print(
            f"[selftest] maturity gate (n_pending={summary['n_pending']}) + "
            f"honest DEAD_OR_GAP degradation on empty NAV data — PASS"
        )

        # ---- IC gate: insufficient matured -> null, never fabricated --------
        s = realized_summary(real_path, pred_path)
        assert s["status"] in ("PENDING_MATURITY", "INSUFFICIENT_MATURED") and s["value"] is None
        print(f"[selftest] realized_ic honestly null pre-threshold (status={s['status']}) — PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- Test D: realization replay against REAL stored labels.parquet -----
    # No waiting 3 years needed: pick a real anchor already matured relative to
    # the codebase's own frozen TODAY, replay it through _realize_one using the
    # SAME cohort membership mf_labels.add_cohort_targets used, and assert
    # bit-agreement with the stored ground truth.
    from mf_benchmarks import PeerProxyResolver

    manifest = load_manifest()
    nav_panel = load_nav_panel(manifest)
    labels = pd.read_parquet("mf_cache/phase_b/labels.parquet")
    anchor = pd.Timestamp("2016-03-31")
    sub = labels[(labels["anchor"] == anchor) & labels["y_cohort_q1"].notna()]
    assert not sub.empty, "FAIL: no cohort-eligible rows at the chosen historical anchor"
    stored = sub.iloc[0]
    code = stored["amfi_code"]

    resolver = PeerProxyResolver(manifest)
    gkey = resolver._group_key(code)
    assert str(gkey) == stored["cohort_key"], "FAIL: cohort_key convention drifted from labels.parquet"
    fake_row = dict(
        amfi_code=code,
        target="cohort_q1",
        anchor=str(anchor.date()),
        cohort_codes=resolver.groups[gkey],
        probability=0.5,
        cohort_percentile=0.5,
    )
    realized = _realize_one(fake_row, nav_panel)
    assert realized["status"] == "REALIZED"
    assert realized["n_peer_r_fwd"] == int(stored["cohort_n"]), (
        f"FAIL: peer R_fwd count {realized['n_peer_r_fwd']} != stored cohort_n {stored['cohort_n']}"
    )
    assert abs(realized["own_r_fwd"] - float(stored["R_fwd"])) < 1e-9
    assert realized["y_realized"] == bool(stored["y_cohort_q1"]), (
        "FAIL: realized y disagrees with the stored training label for the same (fund, anchor)"
    )
    print(
        f"[selftest] Test D (realization replay @ {anchor.date()}, fund {code}): "
        f"matches stored labels.parquet exactly — PASS"
    )

    # ...and a month-end horizon realizes on ITS window, not 3y's: same replay
    # against the 1m labels (mf_cache/phase_b/1m/, read only).
    from mf_horizon import HORIZONS

    h1 = HORIZONS["1m"]
    if not h1.labels_path.exists():
        # research output, never on CI's cache (only 3y's parquets ever are): skip this
        # one case loudly rather than crash the nightly's model-layer gate
        print(f"[selftest] SKIP Test D 1m replay — {h1.labels_path} absent (cold cache)")
    else:
        lab1 = pd.read_parquet(h1.labels_path)
        st1 = lab1[(lab1["anchor"] == anchor) & lab1["y_cohort_q1"].notna()].iloc[0]
        peers1 = resolver.groups[resolver._group_key(st1["amfi_code"])]
        r1 = _realize_one(dict(fake_row, amfi_code=st1["amfi_code"], target=h1.target, cohort_codes=peers1), nav_panel)
        assert r1["status"] == "REALIZED" and r1["n_peer_r_fwd"] == int(st1["cohort_n"]), (r1, st1["cohort_n"])
        assert abs(r1["own_r_fwd"] - float(st1["R_fwd"])) < 1e-9 and r1["y_realized"] == bool(st1["y_cohort_q1"]), r1
        print(f"[selftest] Test D (1m replay @ {anchor.date()}, fund {st1['amfi_code']}): matches 1m labels — PASS")

    # ---- PSI: null case (live == reference itself) and shock case ----------
    import mf_model as M

    df, _ = M.load_cohort_dataset()
    reference = build_psi_reference(out=Path(tempfile.mktemp(suffix=".json")))
    same_features = {c: df[c].to_numpy(dtype=float) for c in PSI_FEATURES if c in df.columns}
    psi_same = compute_psi(same_features, reference)
    assert psi_same["status"] == "OK" and psi_same["psi_max"] < 0.02, (
        f"FAIL: identical distribution should score near-zero PSI, got {psi_same}"
    )
    print(f"[selftest] PSI null case (live==reference): psi_max={psi_same['psi_max']} — PASS")

    shocked = dict(same_features)
    shocked["vol_1y"] = shocked["vol_1y"] * 1.8 + 0.05
    shocked["cagr_3y"] = shocked["cagr_3y"] + 0.15
    psi_shock = compute_psi(shocked, reference)
    assert psi_shock["status"] == "SIGNIFICANT_SHIFT", (
        f"FAIL: shocked features should trip SIGNIFICANT_SHIFT: {psi_shock}"
    )
    shocked_names = {w["feature"] for w in psi_shock["worst"]}
    assert "vol_1y" in shocked_names or "cagr_3y" in shocked_names
    print(f"[selftest] PSI shock case: psi_max={psi_shock['psi_max']} status={psi_shock['status']} — PASS")

    psi_thin = compute_psi({"cagr_1y": np.array([0.1, 0.2])}, reference)
    assert psi_thin["status"] == "INSUFFICIENT_DATA" and psi_thin["psi_max"] is None
    print("[selftest] PSI insufficient-data case honestly null — PASS")

    # ---- rank_stability: first run / identical run / permuted run ----------
    r1 = [dict(amfi_code=f"F{i}", target="cohort_q1", probability=0.5 + 0.01 * i, run_id="run1") for i in range(25)]
    assert rank_stability(r1, [])["status"] == "FIRST_RUN"
    r2_same = [dict(r, run_id="run2") for r in r1]
    stab_same = rank_stability(r2_same, r1)
    assert stab_same["status"] == "OK" and stab_same["spearman"] > 0.99
    r2_shuffled = [
        dict(r2_same[i], probability=r2_same[(i * 7 + 3) % len(r2_same)]["probability"]) for i in range(len(r2_same))
    ]
    stab_shuf = rank_stability(r2_shuffled, r1)
    assert stab_shuf["status"] == "OK" and stab_shuf["spearman"] < 0.9
    print(
        f"[selftest] rank_stability: first_run=FIRST_RUN, identical={stab_same['spearman']}, "
        f"shuffled={stab_shuf['spearman']} — PASS"
    )

    _selftest_horizons(manifest, nav_panel)

    print(
        "[selftest] PASS — ledger append/read is crash-safe and idempotent, realization "
        "replays real stored training labels exactly, PSI/rank_stability degrade honestly"
    )


# ==============================================================================
# SELFTEST — per-horizon ledger (docs/horizons.md §7). Each takes a fresh temp
# root; none can reach the real ledger/.
# ==============================================================================
_CS = dict(
    status="OK",
    target="cohort_q1",
    probability=0.4,
    cohort_percentile=0.6,
    cohort_key="('category', 'Flexi Cap')",
    cohort_n=10,
    universe_n=130,
    anchor="2020-01-31",
    cohort_codes=["A", "B", "C", "D"],
    signal_context={},
    imputed_features=[],
    features={"cagr_1y": 0.1},
)


def _row(code: str, run_id: str, model_id: str, **cs) -> dict:
    return build_row(code, dict(_CS, **cs), run_id=run_id, pipeline_sha="x", model_id=model_id)


def _t_isolation(tmp: Path) -> None:
    """Another target's rows never move cohort_q1's model id or monitoring block."""
    p, real, sup = tmp / "predictions.jsonl", tmp / "realizations.jsonl", tmp / "none.json"
    q1 = [_row(c, "r1", "phase_b_v5_cohort", anchor="2026-10-30") for c in "ABC"]
    append_predictions(q1, path=p)

    def snap():
        mb = monitoring_block(
            run_id="r1", rows_appended=0, current_rows=q1, psi={}, predictions_path=p, realizations_path=real
        )
        return current_model_id(p, sup), mb

    before = snap()
    other = [_row(c, "r2", "phase_b_1m_v1_cohort", target="cohort_q1_1m", anchor="2026-10-31") for c in "ABCD"]
    append_predictions(other, path=p)
    assert snap() == before, f"FAIL: another target's rows leaked into cohort_q1: {before} -> {snap()}"
    print("[selftest] isolation: a 1m target's rows leave cohort_q1's model id + monitoring block identical — PASS")


def _t_rotation(tmp: Path) -> None:
    """New 3y rows land in <root>/cohort_q1/<YYYY-MM>.jsonl; the legacy file is
    never written again, and a row already in it is never appended twice."""
    p = tmp / "predictions.jsonl"
    old = _row("A", "r0", "phase_b_v5_cohort", anchor="2026-10-30")
    p.write_text(json.dumps(old, sort_keys=True) + "\n", encoding="utf-8")
    legacy = p.read_bytes()
    new = _row("B", "r1", "phase_b_v5_cohort", anchor="2026-11-13")
    n = append_predictions([old, new], path=p)
    assert n == 1, f"FAIL: expected only the new row appended, got {n}"
    assert p.read_bytes() == legacy, "FAIL: the legacy predictions file was written to"
    f = tmp / "cohort_q1" / "2026-11.jsonl"
    assert f.exists() and [r["amfi_code"] for r in read_jsonl(f)] == ["B"], "FAIL: new row not rotated"
    assert append_predictions([old, new], path=p) == 0 and p.read_bytes() == legacy
    assert sorted(r["amfi_code"] for r in load_predictions(p)) == ["A", "B"], "FAIL: reader must take legacy + dir"
    print("[selftest] rotation: new 3y row -> cohort_q1/2026-11.jsonl, legacy bytes unchanged, no re-append — PASS")


def _t_unknown_target(tmp: Path) -> None:
    """A row whose target is not in the registry is skipped and counted, never guessed."""
    p = tmp / "predictions.jsonl"
    bad = _row("A", "r1", "m_2w", target="cohort_q1_2w", anchor="2020-01-31")
    p.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
    res = realize_matured_predictions(
        today=pd.Timestamp("2099-01-01"),
        predictions_path=p,
        realizations_path=tmp / "realizations.jsonl",
        manifest=pd.DataFrame(),
        nav_panel={},
        superseded_path=tmp / "none.json",
    )
    assert res.get("n_unknown_target") == 1 and res["n_matured_this_run"] == 0, f"FAIL: {res}"
    print("[selftest] unregistered target: skipped and counted (n_unknown_target=1), not realized — PASS")


def _t_maturity(tmp: Path) -> None:
    """Window end per target (3y anchor + 3y, 1m anchor + MonthEnd(1)), realized only
    once the fund's NAV reaches it or it is more than the tolerance in the past."""
    cases = [  # (target, anchor, today, last NAV date, matured?)
        ("cohort_q1", "2020-03-31", "2023-03-30", "2023-03-30", False),  # before window end
        ("cohort_q1", "2020-03-31", "2023-03-31", "2023-03-31", True),  # NAV reaches it
        ("cohort_q1", "2020-03-31", "2023-04-05", "2023-03-30", False),  # NAV short, 5d <= tol 10
        ("cohort_q1", "2020-03-31", "2023-04-11", "2023-03-30", True),  # 11d > tol: realize (gap)
        ("cohort_q1_1m", "2026-03-31", "2026-04-29", "2026-04-29", False),
        ("cohort_q1_1m", "2026-03-31", "2026-04-30", "2026-04-30", True),
        ("cohort_q1_1m", "2026-03-31", "2026-05-04", "2026-04-29", False),  # 4d <= tol 7
        ("cohort_q1_1m", "2026-03-31", "2026-05-08", "2026-04-29", True),  # 8d > tol 7
    ]
    for i, (target, anchor, today, last, want) in enumerate(cases):
        root = tmp / str(i)
        p = root / "predictions.jsonl"
        append_predictions([_row("A", "r1", "m", target=target, anchor=anchor)], path=p)
        nav = {"A": pd.Series(1.0, index=pd.bdate_range("2019-01-01", last))}
        kw = {} if target == DEFAULT_TARGET else dict(target=target)
        res = realize_matured_predictions(
            today=pd.Timestamp(today),
            predictions_path=p,
            realizations_path=root / "realizations.jsonl",
            manifest=pd.DataFrame(),
            nav_panel=nav,
            superseded_path=root / "none.json",
            **kw,
        )
        assert res["n_matured_this_run"] == int(want), f"FAIL: case {i} {target} {anchor} today={today} {res}"
    print("[selftest] maturity per target: 3y at anchor+3y, 1m at anchor+MonthEnd(1), once NAV reaches it — PASS")


def _t_month_end(tmp: Path, manifest: pd.DataFrame, nav_panel: Dict[str, pd.Series]) -> None:
    """Month-end logging: anchor = M, as-of M only, once per month, skipped after 10 days."""
    from pandas.tseries.offsets import MonthEnd

    from mf_horizon import HORIZONS

    newest = max(s.dropna().index.max() for s in nav_panel.values() if len(s.dropna()))
    M = MonthEnd().rollback(newest.normalize())
    month = [h for h in HORIZONS.values() if not h.legacy]

    def run(root, nav, days, run_id="r1"):
        return log_month_end(
            run_id=run_id,
            pipeline_sha="x",
            manifest=manifest,
            nav_panel=nav,
            today=M + pd.Timedelta(days=days),
            path=root / "predictions.jsonl",
        )

    res = run(tmp / "a", nav_panel, 3)
    for h in month:
        r = res.get(h.target, {})
        assert r.get("appended", 0) > 0 and r["skipped_month"] == 0, f"FAIL: {h.key} {r}"
        rows = load_predictions(tmp / "a" / "predictions.jsonl", target=h.target)
        assert len(rows) == r["appended"], f"FAIL: {h.key} {len(rows)} rows on disk vs {r}"
        assert all(x["anchor"] == str(M.date()) and x["model_id"] == h.model_id for x in rows), f"FAIL: {h.key} rows"
    assert not load_predictions(tmp / "a" / "predictions.jsonl"), "FAIL: month-end logging wrote 3y rows"

    bumped = {c: s.where(s.index <= M, s * 1.37) for c, s in nav_panel.items()}
    run(tmp / "b", bumped, 3)

    def strip(root, h):
        return sorted(
            json.dumps({k: v for k, v in r.items() if k != "logged_at"}, sort_keys=True)
            for r in load_predictions(root / "predictions.jsonl", target=h.target)
        )

    for h in month:
        assert strip(tmp / "a", h) == strip(tmp / "b", h), f"FAIL: {h.key} rows depend on NAV after {M.date()}"
    again = run(tmp / "a", nav_panel, 9, run_id="r2")
    assert all(again[h.target]["appended"] == 0 for h in month), f"FAIL: second call same month: {again}"
    late = run(tmp / "c", nav_panel, 11)
    assert all(late[h.target]["skipped_month"] == 1 and late[h.target]["appended"] == 0 for h in month), late
    assert not (tmp / "c").exists() or not any((tmp / "c").rglob("*.jsonl")), "FAIL: day 11 wrote rows"
    print(
        f"[selftest] month-end {M.date()}: rows at anchor=M for {[h.key for h in month]} "
        f"({', '.join(str(res[h.target]['appended']) for h in month)}), post-M NAV ignored, "
        "second call appends 0, day 11 skipped + counted — PASS"
    )


def _selftest_horizons(manifest: pd.DataFrame, nav_panel: Dict[str, pd.Series]) -> None:
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        for name, t in (("iso", _t_isolation), ("rot", _t_rotation), ("unk", _t_unknown_target), ("mat", _t_maturity)):
            (tmp / name).mkdir()
            t(tmp / name)
        _t_month_end(tmp / "me", manifest, nav_panel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--build-reference",
        action="store_true",
        help="(re)build the PSI reference from the cohort model's training rows",
    )
    ap.add_argument("--horizon", choices=list(HORIZONS), default=DEFAULT, help="with --build-reference (mf_horizon)")
    ap.add_argument(
        "--realize", action="store_true", help="join matured ledger predictions against fresh NAV data (offline job)"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.selftest:
        _selftest()
    elif args.build_reference:
        ref = build_psi_reference(horizon=args.horizon)
        out = HORIZONS[args.horizon].psi_reference_path
        LOGGER.info("psi reference built: %d features -> %s", len(ref["features"]), out)
    elif args.realize:
        # every registered target: one monthly job realizes every horizon's matured rows
        manifest = load_manifest()
        nav_panel = load_nav_panel(manifest)
        for h in HORIZONS.values():
            res = realize_matured_predictions(manifest=manifest, nav_panel=nav_panel, target=h.target)
            LOGGER.info("realize %s: %s", h.target, res)
    else:
        ap.print_help()
