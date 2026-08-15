"""
====================================================================================
mf_cv.py — purged/embargoed CPCV harness + strictly-causal holdout (design §3)
====================================================================================
The #1 leakage risk in this dataset is the 3-YEAR overlap of adjacent monthly
label windows (measured adjacent-label agreement 0.935): a random split puts
near-duplicates of test rows into training and produces fraudulent scores.

Countermeasures implemented here (design §3.1-3.2):
  PURGE    a training anchor a is dropped whenever its label window [a, a+3y]
           overlaps ANY test anchor's window [t, t+3y] — i.e. every training
           anchor within 3y BEFORE the test block or inside the test block's
           forward span is gone.
  EMBARGO  a further 1 month after the test block's forward-window end before
           training anchors resume (serial correlation beyond the window).

Splits provided:
  cpcv_folds()      5 blocks (B1..B5); each fold tests one block, trains on the
                    others after purging + embargo. Training on data AFTER the
                    test block is permitted and clearly labeled: it answers "do
                    these factor weights discriminate in an unseen regime", not
                    "what would live deployment have earned".
  causal_holdout()  train anchors <= 2019-07-31 (all training label windows
                    close by 2022-07), test anchors 2022-08-31..2023-07-31.
                    EVALUATE-ONCE: mf_model enforces a stamp file so this split
                    is scored a single time, after all development freezes.

`--selftest` demonstrates on synthetic data that a memorizable feature +
overlapping labels inflates random-split AUC but does NOT survive purged CPCV.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

LABEL_HORIZON = pd.DateOffset(years=3)
EMBARGO = pd.DateOffset(months=1)

# Modeling table starts 2014-01 (2013 anchors are label-side peer-benchmark
# contributors only — <1y trailing feature history, design §1.1).
MODEL_START = pd.Timestamp("2014-01-31")

# CPCV blocks per design §3.2 (B5 end covers through the last real anchor).
BLOCKS: dict[str, tuple[str, str]] = {
    "B1": ("2014-01-31", "2015-12-31"),
    "B2": ("2016-01-31", "2017-12-31"),
    "B3": ("2018-01-31", "2019-12-31"),
    "B4": ("2020-01-31", "2021-12-31"),
    "B5": ("2022-01-31", "2023-07-31"),
}

CAUSAL_TRAIN_END = pd.Timestamp("2019-07-31")
CAUSAL_TEST_START = pd.Timestamp("2022-08-31")
CAUSAL_TEST_END = pd.Timestamp("2023-07-31")


def purge_train_mask(anchors: pd.Series, test_start: pd.Timestamp, test_end: pd.Timestamp) -> np.ndarray:
    """True where a candidate training anchor SURVIVES purging vs the test span
    [test_start, test_end]: its window [a, a+3y] must close before test_start,
    or open after test_end + 3y + 1-month embargo."""
    a = pd.to_datetime(anchors)
    before = (a + LABEL_HORIZON) < test_start
    after = a > (test_end + LABEL_HORIZON + EMBARGO)
    return np.asarray(before | after)


def block_of(anchors: pd.Series) -> pd.Series:
    """Block id (B1..B5) per anchor; NaN outside the modeling span."""
    a = pd.to_datetime(anchors)
    out = pd.Series(pd.NA, index=a.index, dtype="object")
    for name, (s, e) in BLOCKS.items():
        out[(a >= pd.Timestamp(s)) & (a <= pd.Timestamp(e))] = name
    return out


def cpcv_folds(anchors: pd.Series) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yields (block_name, train_mask, test_mask) over positional order of
    `anchors`. Purge + embargo already applied to the train mask."""
    a = pd.to_datetime(anchors).reset_index(drop=True)
    in_model = np.asarray(a >= MODEL_START)
    for name, (s, e) in BLOCKS.items():
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        test = in_model & np.asarray((a >= ts) & (a <= te))
        train = in_model & ~test & purge_train_mask(a, ts, te)
        yield name, train, test


def causal_holdout(anchors: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """(train_mask, test_mask) for the strictly-causal evaluate-once holdout."""
    a = pd.to_datetime(anchors).reset_index(drop=True)
    in_model = np.asarray(a >= MODEL_START)
    test = in_model & np.asarray((a >= CAUSAL_TEST_START) & (a <= CAUSAL_TEST_END))
    train = in_model & np.asarray(a <= CAUSAL_TRAIN_END) & purge_train_mask(a, CAUSAL_TEST_START, CAUSAL_TEST_END)
    return train, test


def purged_val_split(anchors: pd.Series, train_mask: np.ndarray, val_block: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a validation slice (one block) OUT OF a training mask, purging the
    remaining training anchors against the validation span too. Used for
    hyperparameter selection and calibration — never touches any test data."""
    a = pd.to_datetime(anchors).reset_index(drop=True)
    vs, ve = (pd.Timestamp(x) for x in BLOCKS[val_block])
    val = train_mask & np.asarray((a >= vs) & (a <= ve))
    inner_train = train_mask & ~val & purge_train_mask(a, vs, ve)
    return inner_train, val


# ==============================================================================
# Selftest: overlapping labels + memorizable feature => random-split AUC is
# fraudulently high, purged CPCV stays honest (~0.5). Design §9 task 4.
# ==============================================================================
def run_leakage_selftest(seed: int = 7) -> bool:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(seed)
    n_funds, months = 60, 120
    anchor_grid = pd.date_range("2014-01-31", periods=months, freq="ME")

    # Latent future market path drives the label => 36-month overlapping windows
    # shared by every fund at the same anchor (the real dataset's structure).
    horizon = 36
    market = rng.normal(size=months + horizon)
    fwd = np.array([market[t + 1 : t + 1 + horizon].sum() for t in range(months)])
    # As-of-LEGITIMATE feature that merely identifies the anchor date (cumulative
    # past market). It has no causal forward signal — but under a random split
    # the model memorizes anchor -> majority label, because near-duplicate rows
    # of every test anchor sit in training. That is precisely the fraud purging
    # must kill.
    past_state = np.cumsum(market[:months])

    rows = []
    for i in range(n_funds):
        eps = rng.normal(scale=1.0, size=months)
        for t in range(months):
            rows.append(
                dict(
                    anchor=anchor_grid[t],
                    x_state=past_state[t] + rng.normal(scale=0.01),
                    x_noise=rng.normal(),
                    y=int(fwd[t] + eps[t] > 0),
                )
            )
    df = pd.DataFrame(rows)
    X = df[["x_state", "x_noise"]].to_numpy()
    y = df["y"].to_numpy()

    def _fit_auc(tr, te):
        m = HistGradientBoostingClassifier(random_state=0)
        m.fit(X[tr], y[tr])
        return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])

    random_aucs = [_fit_auc(tr, te) for tr, te in KFold(5, shuffle=True, random_state=0).split(X)]
    cpcv_aucs = []
    for name, tr, te in cpcv_folds(df["anchor"]):
        if te.sum() and len(np.unique(y[te])) == 2 and tr.sum():
            cpcv_aucs.append(_fit_auc(tr, te))

    rand_mean, cpcv_mean = float(np.mean(random_aucs)), float(np.mean(cpcv_aucs))
    print(f"random-split AUC (leaky, fraudulent): {rand_mean:.3f}")
    print(f"purged CPCV AUC (honest):             {cpcv_mean:.3f}")
    ok = rand_mean > 0.75 and cpcv_mean < 0.60
    print("PASS — purging kills the overlap leakage" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if run_leakage_selftest() else 1)
    # No default action: this module is a harness, imported by mf_model.py.
    ap.print_help()
