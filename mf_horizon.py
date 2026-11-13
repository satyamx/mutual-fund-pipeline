"""
====================================================================================
mf_horizon.py — the horizon registry (docs/horizons.md §1-§5)
====================================================================================
One frozen record per forecast horizon (1m, 3m, 6m, 1y, 3y; default 3y): label
window rule, purge/embargo, staleness tolerance, anchor end, causal holdout,
CPCV blocks, effective-n divisor, model id and every per-horizon path.

Every number is PINNED as a literal below — docs/horizons.md was committed
before any non-3y model existed, and a registry that recomputed itself from the
cache could drift after the bar was set. The small rule functions next to it
are the §2/§3 recipe written as code; `--selftest` asserts that the rules
reproduce every pinned value, that the 3y record equals the constants the live
modules carry today (mf_cv, mf_benchmarks, mf_model, mf_ledger, mf_infer), and
that `standing()` scores the frozen 3y holdout stamp WEAK, exactly as §4 says.

3y is legacy by design: its window end stays `t + DateOffset(years=3)` (its
labels and ledger rows are frozen), every other horizon steps in month-ends so
adjacent windows tile the calendar.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from mf_datasources import CACHE

MODEL_DIR = Path(__file__).parent / "model"
MODEL_START = pd.Timestamp("2014-01-31")  # 2013 anchors only feed peer cohorts/benchmarks
CUTOFF = pd.Timestamp("2026-10-31")  # data cutoff D for 1m/3m/6m/1y (§2)
CUTOFF_3Y = pd.Timestamp("2026-08-07")  # 3y stays on its frozen parquet's data date
EMBARGO_MONTHS = 1  # every horizon: one-month-past autocorrelation < 0.10 (§2)
HOLDOUT_ANCHORS = 12  # causal holdout = last 12 labelled anchors
Z_ONE_SIDED_5 = 1.645

# §3 measured inputs, (DEFF12 all anchors, DEFF12 pre-holdout, CS). Produced once
# by mf_horizon_dependence.py; PINNED — re-running that script does not move a divisor.
MEASURED = {
    "1m": (1.39, 1.38, 5.69),
    "3m": (2.65, 2.66, 4.55),
    "6m": (4.28, 4.33, 4.10),
    "1y": (6.24, 6.38, 4.53),
    "3y": (8.76, 8.73, 4.33),
}
# Cross-fund allowance implied by 3y's long-standing divisor of 30 against its
# measured serial DEFF12 (§3). Carries to another horizon only via kappa >= 1.
CROSS_FUND_A = 30 / 8.76


# ==============================================================================
# §2 / §3 rules — derive the pinned values from (months, D)
# ==============================================================================
def shortfall_days(months: int) -> float:
    """max(h/12, 7 days): one twelfth of the window, floored at a weekend + holiday run."""
    return max(months * 365.25 / 144, 7.0)


def min_window_years(months: int) -> float:
    return months / 12 - shortfall_days(months) / 365.25


def tol_days(months: int) -> int:
    return min(10, math.ceil(shortfall_days(months)))


def anchor_end(months: int, cutoff: pd.Timestamp) -> pd.Timestamp:
    """Last month-end anchor whose window can close by D: month-ends <= D - h."""
    return MonthEnd().rollback(cutoff - pd.DateOffset(months=months))


def causal_split(end: pd.Timestamp, months: int, embargo: int = EMBARGO_MONTHS):
    """(test_start, test_end, train_end): test = last 12 anchors, train <= test_start - h - embargo."""
    test_start = end - MonthEnd(HOLDOUT_ANCHORS - 1)
    return test_start, end, test_start - MonthEnd(months + embargo)


def cpcv_blocks(end: pd.Timestamp) -> tuple:
    """2-year calendar blocks from 2014-01, last cut at the anchor end, a final
    block shorter than 12 months merged into the one before it."""
    spans, y = [], MODEL_START.year
    while pd.Timestamp(f"{y}-01-31") <= end:
        spans.append([pd.Timestamp(f"{y}-01-31"), min(pd.Timestamp(f"{y + 1}-12-31"), end)])
        y += 2
    s, e = spans[-1]
    if len(spans) > 1 and (e.year - s.year) * 12 + e.month - s.month + 1 < 12:
        last = spans.pop()
        spans[-1][1] = last[1]
    return tuple((f"B{i}", str(s.date()), str(e.date())) for i, (s, e) in enumerate(spans, 1))


def divisor(key: str) -> int:
    """ceil(max(DEFF12, DEFF12 pre) x A x kappa), kappa = max(1, CS(h) / max(CS(1y), CS(3y)))."""
    deff, deff_pre, cs = MEASURED[key]
    kappa = max(1.0, cs / max(MEASURED["1y"][2], MEASURED["3y"][2]))
    return math.ceil(round(max(deff, deff_pre) * CROSS_FUND_A * kappa, 9))  # round: 8.76 x 30/8.76 is 30, not 30.000...1


def strict_divisor(key: str) -> int:
    """ceil(max(DEFF12, DEFF12 pre) x max(A, CS(h)))."""
    deff, deff_pre, cs = MEASURED[key]
    return math.ceil(round(max(deff, deff_pre) * max(CROSS_FUND_A, cs), 9))


# ==============================================================================
# The registry
# ==============================================================================
@dataclass(frozen=True)
class Horizon:
    key: str
    months: int
    target: str
    model_version: str
    min_window_years: float
    tol_days: int
    embargo_months: int
    anchor_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp
    blocks: tuple  # ((name, start, end), ...) as ISO strings, mf_cv.BLOCKS shape
    divisor: int
    strict_divisor: int
    data_cutoff: pd.Timestamp
    out_dir: Path  # gitignored research outputs
    artifact_path: Path
    psi_reference_path: Path
    stamp_path: Path  # git-tracked holdout stamp + standing

    @property
    def legacy(self) -> bool:
        """3y: frozen labels/ledger, DateOffset arithmetic, OR-label legs, mf_cache stamp."""
        return self.key == "3y"

    @property
    def model_id(self) -> str:
        return self.model_version + "_cohort"

    @property
    def embargo(self):
        return pd.DateOffset(months=self.embargo_months) if self.legacy else MonthEnd(self.embargo_months)

    def window_end(self, t):
        """Label window end for anchor(s) t (Timestamp or Series)."""
        return t + (pd.DateOffset(years=self.months // 12) if self.legacy else MonthEnd(self.months))

    # per-horizon research files (gitignored): 3y's names, one directory per horizon
    labels_path = property(lambda self: self.out_dir / "labels.parquet")
    features_path = property(lambda self: self.out_dir / "features.parquet")
    cpcv_results_path = property(lambda self: self.out_dir / "cpcv_results_cohort.json")
    oos_pred_path = property(lambda self: self.out_dir / "oos_predictions_cohort.parquet")
    holdout_pred_path = property(lambda self: self.out_dir / "holdout_predictions_cohort.parquet")
    coef_path = property(lambda self: self.out_dir / "coefficients_cohort.json")
    report_path = property(lambda self: self.out_dir / "cohort_target_report.md")
    cache_stamp_path = property(lambda self: self.out_dir / "holdout_evaluated_cohort.json")  # 3y's legacy stamp


_B14_21 = (
    ("B1", "2014-01-31", "2015-12-31"),
    ("B2", "2016-01-31", "2017-12-31"),
    ("B3", "2018-01-31", "2019-12-31"),
    ("B4", "2020-01-31", "2021-12-31"),
)
_B5_FULL = ("B5", "2022-01-31", "2023-12-31")


def _h(key, months, version, min_win, tol, end, ts, te, tr, blocks, div, strict, cutoff) -> Horizon:
    T = pd.Timestamp
    sfx = "" if key == "3y" else f"_{key}"
    return Horizon(
        key=key,
        months=months,
        target="cohort_q1" + sfx,
        model_version=version,
        min_window_years=min_win,
        tol_days=tol,
        embargo_months=EMBARGO_MONTHS,
        anchor_end=T(end),
        test_start=T(ts),
        test_end=T(te),
        train_end=T(tr),
        blocks=blocks,
        divisor=div,
        strict_divisor=strict,
        data_cutoff=cutoff,
        out_dir=CACHE / "phase_b" / key if sfx else CACHE / "phase_b",
        artifact_path=MODEL_DIR / f"model_artifact_cohort{sfx}.json",
        psi_reference_path=MODEL_DIR / f"psi_reference{sfx}.json",
        stamp_path=MODEL_DIR / f"holdout_stamp_{key}.json",
    )


# fmt: off
HORIZONS: dict[str, Horizon] = {
    "1m": _h("1m", 1, "phase_b_1m_v1", 0.06417, 7, "2026-09-30", "2025-10-31", "2026-09-30", "2025-08-31",
             _B14_21 + (_B5_FULL, ("B6", "2024-01-31", "2026-09-30")), 6, 8, CUTOFF),
    "3m": _h("3m", 3, "phase_b_3m_v1", 0.22917, 8, "2026-07-31", "2025-08-31", "2026-07-31", "2025-04-30",
             _B14_21 + (_B5_FULL, ("B6", "2024-01-31", "2026-07-31")), 10, 13, CUTOFF),
    "6m": _h("6m", 6, "phase_b_6m_v1", 0.45833, 10, "2026-04-30", "2025-05-31", "2026-04-30", "2024-10-31",
             _B14_21 + (_B5_FULL, ("B6", "2024-01-31", "2026-04-30")), 15, 18, CUTOFF),
    # 1y divisor 23, not the formula's 22: settled 2026-11-10 on a pre-holdout DEFF12 of
    # 6.49 (the cache has since moved to 6.38); the larger divisor is the stricter bar (§3).
    "1y": _h("1y", 12, "phase_b_1y_v1", 0.91667, 10, "2025-10-31", "2024-11-30", "2025-10-31", "2023-10-31",
             _B14_21 + (_B5_FULL, ("B6", "2024-01-31", "2025-10-31")), 23, 29, CUTOFF),
    "3y": _h("3y", 36, "phase_b_v5", 2.75, 10, "2023-07-31", "2022-08-31", "2023-07-31", "2019-07-31",
             _B14_21 + (("B5", "2022-01-31", "2023-07-31"),), 30, 38, CUTOFF_3Y),
}
# fmt: on
DEFAULT = "3y"


# ==============================================================================
# §4 — the bar
# ==============================================================================
def hanley_mcneil_se(auc: float, n_pos: float, n_neg: float) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)
    return math.sqrt(var / (n_pos * n_neg))


def lower_bound(auc: float, n_pos: float, n_neg: float, div: float) -> float:
    """One-sided 95% lower bound of the AUC at the effective counts n/div."""
    return auc - Z_ONE_SIDED_5 * hanley_mcneil_se(auc, n_pos / div, n_neg / div)


def auc_needed(n: int, base_rate: float, div: float) -> float:
    """Smallest AUC whose lower bound clears 0.50 at (n, base rate, divisor)."""
    lo, hi = 0.5, 0.999
    for _ in range(60):  # bisection; lower_bound is increasing in auc on (0.5, 1)
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if lower_bound(mid, n * base_rate, n * (1 - base_rate), div) > 0.5 else (mid, hi)
    return hi


def standing(holdout: dict, cpcv_block_aucs, h: Horizon) -> dict:
    """PASS / WEAK / FAIL per §4 from the causal-holdout enet metrics (block_metrics
    keys: pooled_auc, rank_auc, lift_over_base, n, negatives) and the CPCV pooled
    AUC of each block. PASS that fails (a) at the strict divisor is `marginal`."""
    auc, n, neg = holdout["pooled_auc"], holdout["n"], holdout["negatives"]
    lb = lower_bound(auc, n - neg, neg, h.divisor)
    lb_strict = lower_bound(auc, n - neg, neg, h.strict_divisor)
    aucs = list(cpcv_block_aucs)
    assert len(aucs) == len(h.blocks), f"{h.key}: {len(aucs)} CPCV blocks, registry has {len(h.blocks)}"
    above, needed = sum(a > 0.5 for a in aucs), math.ceil(2 * len(h.blocks) / 3)  # NaN counts as not above
    a, b, c, d = lb > 0.5, holdout["rank_auc"] > 0.5, holdout["lift_over_base"] > 1, above >= needed
    st = "PASS" if a and b and c and d else "WEAK" if b and c and d else "FAIL"
    return dict(
        standing=st,
        marginal=st == "PASS" and not lb_strict > 0.5,
        lower_bound=lb,
        lower_bound_strict=lb_strict,
        divisor=h.divisor,
        strict_divisor=h.strict_divisor,
        effective_n=round(n / h.divisor),
        checks=dict(a_lower_bound=a, b_rank_auc=b, c_lift=c, d_cpcv=d),
        cpcv_blocks_above=above,
        cpcv_blocks_needed=needed,
    )


# ==============================================================================
# Selftest
# ==============================================================================
def run_selftest() -> bool:
    import inspect

    import mf_benchmarks
    import mf_cv
    import mf_infer
    import mf_labels
    import mf_ledger
    import mf_model

    fails: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        if not ok:
            fails.append(label)
            print(f"RED  {label} {detail}")

    def close(label, got, want, tol):
        check(label, got is not None and abs(got - want) <= tol, f"got {got!r} want {want!r}")

    def eq(label, got, want):
        check(label, got == want, f"got {got!r} want {want!r}")

    # (i) the rules reproduce every pinned value; 3y equals today's live constants
    for key, h in HORIZONS.items():
        cut = h.data_cutoff
        close(f"{key} min window", min_window_years(h.months), h.min_window_years, 5e-6)
        eq(f"{key} tol days", tol_days(h.months), h.tol_days)
        end = anchor_end(h.months, cut)
        eq(f"{key} anchor end", end, h.anchor_end)
        eq(f"{key} holdout", causal_split(end, h.months), (h.test_start, h.test_end, h.train_end))
        eq(f"{key} cpcv blocks", cpcv_blocks(end), h.blocks)
        eq(f"{key} holdout is 12 anchors", len(pd.date_range(h.test_start, h.test_end, freq="ME")), HOLDOUT_ANCHORS)
    h3 = HORIZONS["3y"]
    probe = pd.Series(pd.to_datetime(["2016-02-29", "2019-07-31", "2020-02-29", "2023-07-31"]))
    check("3y window == mf_cv.LABEL_HORIZON", (h3.window_end(probe) == probe + mf_cv.LABEL_HORIZON).all())
    check("3y window == mf_ledger.MATURITY", (h3.window_end(probe) == probe + mf_ledger.MATURITY).all())
    check("3y embargo == mf_cv.EMBARGO", (probe + h3.embargo == probe + mf_cv.EMBARGO).all())
    eq("MODEL_START == mf_cv", MODEL_START, mf_cv.MODEL_START)
    eq("3y blocks == mf_cv.BLOCKS", {n: (s, e) for n, s, e in h3.blocks}, mf_cv.BLOCKS)
    eq("3y train end", h3.train_end, mf_cv.CAUSAL_TRAIN_END)
    eq("3y test start", h3.test_start, mf_cv.CAUSAL_TEST_START)
    eq("3y test end", h3.test_end, mf_cv.CAUSAL_TEST_END)
    fr = inspect.signature(mf_benchmarks.forward_return).parameters
    eq("forward_return years", fr["years"].default * 12, h3.months)
    eq("forward_return tol", fr["tol_days"].default, h3.tol_days)
    eq("forward_return min_yrs", fr["min_yrs"].default, h3.min_window_years)
    eq("3y divisor == EFFECTIVE_N_DIVISOR", h3.divisor, mf_model.EFFECTIVE_N_DIVISOR)
    eq("3y version == MODEL_VERSION", h3.model_version, mf_model.MODEL_VERSION)
    eq("3y artifact path == mf_infer", h3.artifact_path, mf_infer.COHORT_ARTIFACT_PATH)
    eq("3y psi path == mf_ledger", h3.psi_reference_path, mf_ledger.PSI_REFERENCE_PATH)
    eq("3y target == DEFAULT_TARGET", h3.target, mf_infer.DEFAULT_TARGET)
    eq("3y labels path == mf_labels", h3.labels_path, mf_labels.LABELS_PATH)
    eq("default horizon", DEFAULT, "3y")

    # (ii) the divisor procedure, fed the §3 measured inputs
    for key, (d, s) in dict(zip(HORIZONS, [(6, 8), (10, 13), (15, 18), (22, 29), (30, 38)])).items():
        eq(f"{key} divisor formula", divisor(key), d)
        eq(f"{key} strict formula", strict_divisor(key), s)
        eq(f"{key} strict pinned", HORIZONS[key].strict_divisor, s)
        if key != "1y":
            eq(f"{key} divisor pinned", HORIZONS[key].divisor, d)
    eq("1y pinned 23 (stricter than formula's 22)", HORIZONS["1y"].divisor, 23)

    # (iii) the bar: frozen 3y stamp scores WEAK, lower bound 0.4452
    # read from the git-tracked stamp, never mf_cache/: this must run on a cold CI cache
    stamp = json.loads(h3.stamp_path.read_text(encoding="utf-8"))
    s3 = standing(stamp["models"]["cohort_q1:enet"], stamp["cpcv_block_aucs"].values(), h3)
    eq("3y stamp standing block is what standing() gives", stamp["standing"], s3)
    eq("3y frozen stamp standing", s3["standing"], "WEAK")
    close("3y frozen lower bound", s3["lower_bound"], 0.4452, 0.0005)
    close("1y AUC needed (n 5872, base 0.277, div 23)", auc_needed(5872, 0.277, 23), 0.5672, 0.0005)

    h1m = HORIZONS["1m"]  # divisor 6 needs ~0.533, strict 8 needs ~0.538 at 6,447 rows

    def synth(auc, rank=0.53, lift=1.2, above=4):
        n, neg = 6447, round(6447 * 0.725)
        m = dict(pooled_auc=auc, rank_auc=rank, lift_over_base=lift, n=n, negatives=neg)
        return standing(m, [0.55] * above + [0.45] * (6 - above), h1m)

    p = synth(0.56)
    check("synthetic PASS", p["standing"] == "PASS" and not p["marginal"], str(p))
    p = synth(0.535)
    check("synthetic marginal PASS", p["standing"] == "PASS" and p["marginal"], str(p))
    p = synth(0.52)
    check("synthetic WEAK", p["standing"] == "WEAK" and not p["marginal"], str(p))
    for label, kw in [("lift<=1", dict(lift=0.95)), ("rank<=0.5", dict(rank=0.5)), ("3 of 6 blocks", dict(above=3))]:
        p = synth(0.56, **kw)
        check(f"synthetic FAIL ({label})", p["standing"] == "FAIL", str(p))

    print(f"mf_horizon selftest: {'PASS' if not fails else f'{len(fails)} RED'}")
    return not fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    for h in HORIZONS.values():
        print(h.key, h.model_id, h.anchor_end.date(), h.test_start.date(), h.train_end.date(), h.divisor, h.strict_divisor)
