#!/usr/bin/env python3
"""
bootstrap.py — ONE COMMAND. Run it, walk away, come back done.

    pip install requests pandas numpy scipy pydantic pyarrow yfinance
    python3 bootstrap.py --funds "Parag Parikh Flexi Cap" "HDFC Small Cap"

Automates everything that CAN be automated:
  1. AMFI scheme master (NAVAll.txt)                  -> mf_cache/amfi_master.parquet
  2. Resolves each fund to its DIRECT-GROWTH code     (never IDCW — see NAVCleaner)
  3. Full NAV history per fund from mfapi.in          -> mf_cache/mfapi_<code>.json
  4. NSE stock prices via yfinance                    -> mf_cache/px_*.parquet
  5. AMFI cap-band list (auto; prints the 60s manual step if AMFI blocks it)
  6. Readiness table + exactly what remains

Idempotent and cached: the pipeline then runs fully offline.
"""
from __future__ import annotations

import argparse
import logging
import sys

from mf_datasources import (
    AMFI_CAPLIST_PAGE, CACHE, AMFIRegistryLive, AnthropicNewsAgent,
    CapBandAdapter, MFAPIAdapter, YFinanceAdapter, readiness,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
LOG = logging.getLogger("bootstrap")


def _candidate_codes(master) -> list:
    """AMFI codes for the canonical funds OUTSIDE the manifest that are worth fetching.

    Two filters, and the second one is the point of this function. `mf_universe`
    decides which funds the model may score at all; on top of that we drop the
    sector-keyed categories with no curated sector, because `score_live` refuses
    those (SECTOR_UNRESOLVED) *before* it would ever call its nav_loader. Fetching
    them would download hundreds of NAV histories that nothing can score until a
    human fills the sector worklist — see blocker #2.
    """
    import mf_overrides
    import mf_universe
    from mf_labels import load_manifest
    # Imported rather than restated: if this list and score_live's ever diverged,
    # we would fetch funds it refuses, or refuse funds we never fetched.
    from mf_live_score import SECTOR_KEYED_CATEGORIES

    manifest = load_manifest()
    cands = mf_universe.canonical_candidates(master, manifest)
    outsiders = cands[~cands["in_manifest"]]

    overrides, _issues = mf_overrides.load(manifest=manifest)
    curated = {str(code) for code, ov in (overrides or {}).items()
               if getattr(ov, "sector", None)}
    sector_keyed = outsiders["category"].isin(SECTOR_KEYED_CATEGORIES)
    blocked = sector_keyed & ~outsiders["amfi_code"].isin(curated)

    keep = outsiders[~blocked]
    LOG.info("[2/5] --candidates: %d canonical funds, %d in the manifest, %d outside it",
             len(cands), int(cands["in_manifest"].sum()), len(outsiders))
    LOG.info("      fetching %d; skipping %d blocked on a curated sector (mf_overrides.py --gaps)",
             len(keep), int(blocked.sum()))
    return [str(c) for c in keep["amfi_code"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funds", nargs="*", default=[], help="Fund names or ISINs (your shortlist)")
    ap.add_argument("--stocks", nargs="*", default=[], help="NSE tickers, e.g. RELIANCE HDFCBANK")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--from-manifest", action="store_true",
                    help="refresh the whole tracked universe by loading fund names from "
                         "overrides/universe_manifest.csv (for the scheduled nightly job). "
                         "Cache-first + 20h TTL means only stale NAV is refetched — the "
                         "throttle is the cache, not a sleep. The manifest is git-tracked, "
                         "so this works on a COLD runner with no cache at all.")
    ap.add_argument("--candidates", action="store_true",
                    help="ALSO fetch NAV for the canonical Direct+Growth funds OUTSIDE the "
                         "manifest that the cohort model may score by insertion. Fetches by "
                         "AMFI CODE, so it never touches name resolution. Sector-keyed "
                         "categories with no curated sector are skipped, because score_live "
                         "refuses them before it would ever ask for their NAV.")
    args = ap.parse_args()

    if args.from_manifest:
        from mf_labels import MANIFEST_PATH, load_manifest  # lazy: only when scheduled
        if not MANIFEST_PATH.exists():
            LOG.error("--from-manifest: %s not found. The manifest is hand-curated and "
                      "git-tracked, so on a clean checkout it should always be present — "
                      "if it is missing, something deleted it rather than a cold cache.",
                      MANIFEST_PATH)
            return 2
        names = list(load_manifest()["scheme_name"])
        LOG.info("--from-manifest: refreshing %d funds from %s", len(names), MANIFEST_PATH.name)
        args.funds = list(args.funds) + names

    print("=" * 78 + "\n  MF PIPELINE BOOTSTRAP\n" + "=" * 78)

    LOG.info("[1/5] AMFI scheme master ...")
    amfi = AMFIRegistryLive()
    master = amfi.master()
    LOG.info("      OK — %d schemes cached", len(master)) if not master.empty \
        else LOG.error("      AMFI unreachable — check outbound HTTPS, then re-run.")

    mfapi, resolved = MFAPIAdapter(), []
    if args.funds and not master.empty:
        LOG.info("[2/5] Resolving %d fund(s) to DIRECT-GROWTH codes ...", len(args.funds))
        for q in args.funds:
            row = amfi.resolve(q)
            if row is None:
                LOG.warning("      NOT FOUND: %r (try a shorter fragment)", q)
                continue
            LOG.info("      %-30s -> %s | %s", q, row["amfi_code"], row["scheme_name"][:50])
            resolved.append(row["amfi_code"])
    elif not args.candidates:
        LOG.info("[2/5] no --funds given, skipped")
    if args.candidates and not master.empty:
        resolved += _candidate_codes(master)
    # A code reached by BOTH paths (a --funds query naming an out-of-manifest fund)
    # would otherwise be fetched twice. Order-preserving so the log reads sensibly.
    resolved = list(dict.fromkeys(resolved))

    if resolved:
        LOG.info("[3/5] Pulling NAV history for %d code(s) ...", len(resolved))
        n_ok = n_fail = 0
        for code in resolved:
            nav, meta, rep = mfapi.nav_series(code)
            if nav is None:
                LOG.warning("      NAV fetch failed: %s", code)
                n_fail += 1
                continue
            n_ok += 1
            msg = f"      {code}: {len(nav)} obs, {nav.index.min().date()} -> {nav.index.max().date()}"
            if rep and (rep.spikes_removed or rep.payout_steps_detected):
                msg += f"  [{rep.summary()}]"
            LOG.info(msg)
        LOG.info("      NAV: %d ok, %d failed", n_ok, n_fail)
    else:
        LOG.info("[3/5] skipped")

    if args.stocks:
        LOG.info("[4/5] Pulling NSE prices for %d tickers ...", len(args.stocks))
        px = YFinanceAdapter().prices(args.stocks, start=args.start)
        LOG.info("      OK — %d tickers x %d days", px.shape[1], px.shape[0]) if px is not None \
            else LOG.warning("      yfinance failed (pip install yfinance / check network)")
    else:
        LOG.info("[4/5] no --stocks given — pass your funds' top-10 holdings here")

    LOG.info("[5/5] AMFI cap-band classification ...")
    cap = CapBandAdapter()
    cap.ensure()
    if cap.health():
        LOG.info("      OK — %d symbols classified", len(cap.band_lookup()))
    else:
        print(f"""
      >>> THE ONLY MANUAL STEP (60 seconds, twice a year) <<<
      1. Open {AMFI_CAPLIST_PAGE}
      2. Download the current list (the one effective 1-Jul-2026)
      3. Save as CSV to: {CACHE / 'amfi_cap_classification.csv'}
         (any column layout — the loader sniffs symbol/company/cap columns)
      Without it the SEBI 80%/65% true-to-label checks cannot run.
""")

    print("\n" + "=" * 78 + "\n  READINESS\n" + "=" * 78)
    print(readiness().to_string(index=False))

    if not AnthropicNewsAgent().health():
        print("\n  Agent D (news) OFF: export ANTHROPIC_API_KEY=sk-ant-...")
        print("  Without it Agent D returns NEUTRAL. It will never invent headlines.")

    print("\n  Historical disclosures (Agent B's core input) CANNOT be automated —")
    print("  no free API exposes a fund's top-10 holdings as of 3 and 5 years ago.")
    print(f"  Drop CSVs here: {CACHE / 'disclosures'}/<code>_<YYYY-MM>.csv")
    print("  Columns: instrument,weight,asset_type,sector,cap_band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
