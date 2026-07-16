"""
====================================================================================
mf_labels.py — Phase B label builder (3-year forward OR-label)
====================================================================================
Builds the labeled dataset per docs/phase_b_design.md §1: monthly month-end
anchors, 3-year-forward annualized fund return, and the OR-label

    condA(i,t) = R_fwd(i,t) - Bench_fwd(i,t) >= +0.02   (excess vs benchmark)
    condB(i,t) = R_fwd(i,t) >= 0.10                     (absolute)
    y(i,t)     = condA OR condB

The label is computed TWICE per (fund, anchor) for sensitivity: once against
the bias-aware hybrid benchmark (mf_benchmarks.resolve_hybrid — real index
where the cached history covers the window, else peer-proxy) and once against
the pure peer-proxy benchmark (mf_benchmarks.resolve_pure_peer — the original
design §1.3 default). condB and R_fwd do not depend on the benchmark and are
stored once.

No lookahead: the benchmark's own forward window is computed with the exact
same as-of/staleness rule as the fund's own forward window, over the same
[t, t+3y] span (mf_benchmarks.forward_return, shared by both sides).

Output: mf_cache/phase_b/labels.parquet
"""

from __future__ import annotations

import pandas as pd

from mf_datasources import CACHE, MFAPIAdapter
from mf_benchmarks import (
    PeerProxyResolver, forward_return, resolve_hybrid, resolve_pure_peer,
)

OUT_DIR = CACHE / "phase_b"
LABELS_PATH = OUT_DIR / "labels.parquet"
REPORT_PATH = OUT_DIR / "base_rate_sensitivity.md"

MANIFEST_PATH = CACHE / "universe_manifest.csv"

# Within-cohort relative targets (see add_cohort_targets below): minimum number
# of funds with a valid R_fwd in a fund's (anchor, cohort) cell, else excluded.
COHORT_MIN_SIZE = 4

# Wide anchor grid; forward_return() naturally truncates wherever cached NAV
# history runs out (10-day staleness + 2.75y minimum window, per design §1.2),
# so no anchor cutoff needs to be hardcoded here.
ANCHOR_GRID = pd.date_range("2013-01-31", "2024-12-31", freq="ME")

ABSOLUTE_THRESHOLD = 0.10
EXCESS_THRESHOLD = 0.02


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH, dtype={"amfi_code": str})


def load_nav_panel(manifest: pd.DataFrame) -> dict[str, pd.Series]:
    """Cleaned NAV series per fund, via the shared NAVCleaner (mf_datasources)."""
    adapter = MFAPIAdapter()
    panel = {}
    for code in manifest["amfi_code"]:
        series, _meta, _report = adapter.nav_series(code)
        if series is not None and len(series) >= 5:
            panel[code] = series
    return panel


def build_fwd_panel(manifest: pd.DataFrame, nav_panel: dict[str, pd.Series]
                     ) -> dict[tuple[str, pd.Timestamp], float]:
    """R_fwd(code, t) for every fund/anchor with a valid 3y-forward window."""
    fwd = {}
    for code in manifest["amfi_code"]:
        series = nav_panel.get(code)
        if series is None:
            continue
        for t in ANCHOR_GRID:
            res = forward_return(series, t)
            if res is not None:
                fwd[(code, t)] = res.value
    return fwd


def build_labels(manifest: pd.DataFrame, fwd_panel: dict) -> pd.DataFrame:
    peer_resolver = PeerProxyResolver(manifest)
    manifest_idx = manifest.set_index("amfi_code")
    rows = []
    for code in manifest["amfi_code"]:
        row = manifest_idx.loc[code]
        category, sector = row["category"], row["sector"]
        for t in ANCHOR_GRID:
            r_fwd = fwd_panel.get((code, t))
            if r_fwd is None:
                continue
            cond_b = r_fwd >= ABSOLUTE_THRESHOLD

            hybrid = resolve_hybrid(code, category, sector, t, peer_resolver, fwd_panel)
            cond_a = bool((r_fwd - hybrid["benchmark_fwd"]) >= EXCESS_THRESHOLD) \
                if pd.notna(hybrid["benchmark_fwd"]) else False
            y = cond_a or cond_b

            peer = resolve_pure_peer(code, t, peer_resolver, fwd_panel)
            cond_a_peer = bool((r_fwd - peer["benchmark_fwd"]) >= EXCESS_THRESHOLD) \
                if pd.notna(peer["benchmark_fwd"]) else False
            y_peer = cond_a_peer or cond_b

            rows.append(dict(
                amfi_code=code, scheme_name=row["scheme_name"], category=category,
                sector=sector, anchor=t, anchor_year=t.year,
                R_fwd=r_fwd, condB=cond_b,
                benchmark_fwd=hybrid["benchmark_fwd"], benchmark_source=hybrid["benchmark_source"],
                benchmark_quality=hybrid["benchmark_quality"], condA=cond_a, y=y,
                benchmark_fwd_peer=peer["benchmark_fwd"], benchmark_quality_peer=peer["benchmark_quality"],
                condA_peer=cond_a_peer, y_peer=y_peer,
            ))
    return pd.DataFrame(rows)


# ==============================================================================
# Sensitivity report
# ==============================================================================
def _rate_table(df: pd.DataFrame) -> pd.DataFrame:
    by_year = df.groupby("anchor_year").agg(
        n=("y", "size"),
        y_hybrid=("y", "mean"), y_peer=("y_peer", "mean"),
        condA_hybrid=("condA", "mean"), condA_peer=("condA_peer", "mean"),
        condB=("condB", "mean"),
        negatives_hybrid=("y", lambda s: int((~s).sum())),
        negatives_peer=("y_peer", lambda s: int((~s).sum())),
    )
    overall = pd.DataFrame([dict(
        n=len(df), y_hybrid=df["y"].mean(), y_peer=df["y_peer"].mean(),
        condA_hybrid=df["condA"].mean(), condA_peer=df["condA_peer"].mean(),
        condB=df["condB"].mean(),
        negatives_hybrid=int((~df["y"]).sum()), negatives_peer=int((~df["y_peer"]).sum()),
    )], index=["OVERALL"])
    return pd.concat([by_year, overall])


def _coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    cov = df["benchmark_source"].value_counts().reindex(
        ["fund_nav", "adj_pri", "peer_proxy", "none"], fill_value=0)
    cov_pct = (cov / len(df) * 100).round(1)
    return pd.DataFrame({"n_samples": cov, "pct": cov_pct})


def write_report(df: pd.DataFrame) -> str:
    rates = _rate_table(df)
    cov = _coverage_table(df)

    def fmt_rate_row(idx, r):
        return (f"| {idx} | {int(r.n)} | {r.y_hybrid:.3f} | {r.y_peer:.3f} | "
                f"{r.condA_hybrid:.3f} | {r.condA_peer:.3f} | {r.condB:.3f} | "
                f"{int(r.negatives_hybrid)} | {int(r.negatives_peer)} |")

    lines = [
        "# Phase B base-rate sensitivity: peer-proxy vs bias-aware hybrid benchmark",
        "",
        f"Total labeled samples: **{len(df)}** "
        f"(anchors {df['anchor'].min().date()} .. {df['anchor'].max().date()}, "
        f"{df['amfi_code'].nunique()} funds).",
        "",
        "## Base rates: peer-proxy vs hybrid, by anchor year",
        "",
        "| anchor_year | n | P(y=1) hybrid | P(y=1) peer | P(condA) hybrid | "
        "P(condA) peer | P(condB) | negatives hybrid | negatives peer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for idx, r in rates.iterrows():
        lines.append(fmt_rate_row(idx, r))
    lines += [
        "",
        "## Benchmark-source coverage (hybrid scheme)",
        "",
        "| source | n_samples | pct |",
        "|---|---|---|",
    ]
    for src, r in cov.iterrows():
        lines.append(f"| {src} | {int(r.n_samples)} | {r.pct}% |")

    overall = rates.loc["OVERALL"]
    excess_leg_shift = overall.condA_hybrid - overall.condA_peer
    y_shift = overall.y_hybrid - overall.y_peer
    lines += [
        "",
        "## Notes",
        "",
        f"- condB (absolute, >=10% p.a.) is benchmark-independent: identical under both "
        f"schemes, overall {overall.condB:.3f}.",
        f"- Overall condA (excess) rate moves from {overall.condA_peer:.3f} (pure peer-proxy) "
        f"to {overall.condA_hybrid:.3f} (bias-aware hybrid), a shift of {excess_leg_shift:+.3f}.",
        f"- Overall OR-label base rate moves from {overall.y_peer:.3f} (peer-proxy) to "
        f"{overall.y_hybrid:.3f} (hybrid), a shift of {y_shift:+.3f}.",
        f"- Negatives available: {int(overall.negatives_hybrid)} (hybrid) / "
        f"{int(overall.negatives_peer)} (peer-proxy) out of {int(overall.n)} samples.",
    ]
    text = "\n".join(lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


# ==============================================================================
# Within-cohort relative targets — a second, better-posed modeling attempt.
# Computed per (anchor, cohort) directly from the already-built R_fwd column
# (no NAV re-fetch): does NOT depend on the benchmark resolver at all, only on
# the PeerProxyResolver GROUPING (same manifest `category` for diversified;
# same `sector` for thematic with >=3 funds; pooled small-sector thematic group
# otherwise — identical grouping to the peer-proxy benchmark leg above).
#
#   y_cohort_q1(i,t)        = R_fwd(i,t) >= cohort's 75th percentile of R_fwd at t
#   y_cohort_top_half(i,t)  = R_fwd(i,t) >  cohort's median R_fwd at t
#
# A fund's own R_fwd participates in its cohort's quantile (standard relative-
# rank convention, not leave-one-out — LOO is a benchmark-construction device
# used for condA above, not needed for a within-group rank). Anchors/cohorts
# with fewer than COHORT_MIN_SIZE funds carrying a valid R_fwd are excluded
# (both target columns are pd.NA for that row) rather than guessed.
# ==============================================================================
def add_cohort_targets(df: pd.DataFrame, manifest: pd.DataFrame,
                       min_size: int = COHORT_MIN_SIZE) -> tuple[pd.DataFrame, dict]:
    peer_resolver = PeerProxyResolver(manifest)
    cohort_key = {code: peer_resolver._group_key(code) for code in manifest["amfi_code"]}

    df = df.copy()
    df["cohort_key"] = df["amfi_code"].map(cohort_key).astype(str)

    grp = df.groupby(["anchor", "cohort_key"])["R_fwd"]
    cohort_n = grp.transform("size")
    q75 = grp.transform(lambda s: s.quantile(0.75))
    med = grp.transform("median")
    eligible = cohort_n >= min_size

    df["cohort_n"] = cohort_n
    df["y_cohort_q1"] = (df["R_fwd"] >= q75).astype("boolean")
    df["y_cohort_top_half"] = (df["R_fwd"] > med).astype("boolean")
    df.loc[~eligible, ["y_cohort_q1", "y_cohort_top_half"]] = pd.NA

    stats = dict(
        n_total=int(len(df)), n_dropped=int((~eligible).sum()),
        n_kept=int(eligible.sum()),
        base_rate_q1=float(df.loc[eligible, "y_cohort_q1"].mean()),
        base_rate_top_half=float(df.loc[eligible, "y_cohort_top_half"].mean()),
    )
    return df, stats


def regenerate_with_cohort_targets() -> pd.DataFrame:
    """Adds y_cohort_q1/y_cohort_top_half to the existing labels.parquet in
    place, reusing the already-validated R_fwd panel (no NAV rebuild). All
    existing columns are kept untouched."""
    manifest = load_manifest()
    df = pd.read_parquet(LABELS_PATH)
    df, stats = add_cohort_targets(df, manifest)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LABELS_PATH, index=False)
    print(f"cohort targets added -> {LABELS_PATH}: "
          f"kept={stats['n_kept']} dropped={stats['n_dropped']} "
          f"(min cohort size {COHORT_MIN_SIZE} funds with valid R_fwd); "
          f"base rate y_cohort_q1={stats['base_rate_q1']:.3f} "
          f"y_cohort_top_half={stats['base_rate_top_half']:.3f}")
    return df


def main() -> pd.DataFrame:
    manifest = load_manifest()
    nav_panel = load_nav_panel(manifest)
    fwd_panel = build_fwd_panel(manifest, nav_panel)
    labels = build_labels(manifest, fwd_panel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS_PATH, index=False)

    report = write_report(labels)
    print(report)
    return labels


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-only", action="store_true",
                    help="add within-cohort targets to the existing labels.parquet "
                         "(reuses cached R_fwd; no NAV rebuild)")
    args = ap.parse_args()
    if args.cohort_only:
        regenerate_with_cohort_targets()
    else:
        main()
