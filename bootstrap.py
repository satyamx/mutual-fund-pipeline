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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funds", nargs="*", default=[], help="Fund names or ISINs (your shortlist)")
    ap.add_argument("--stocks", nargs="*", default=[], help="NSE tickers, e.g. RELIANCE HDFCBANK")
    ap.add_argument("--start", default="2019-01-01")
    args = ap.parse_args()

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
        LOG.info("[3/5] Pulling NAV history ...")
        for code in resolved:
            nav, meta, rep = mfapi.nav_series(code)
            if nav is None:
                LOG.warning("      NAV fetch failed: %s", code)
                continue
            msg = f"      {code}: {len(nav)} obs, {nav.index.min().date()} -> {nav.index.max().date()}"
            if rep and (rep.spikes_removed or rep.payout_steps_detected):
                msg += f"  [{rep.summary()}]"
            LOG.info(msg)
    else:
        LOG.info("[2/5] no --funds given, skipped")
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
