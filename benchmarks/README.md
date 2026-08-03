# benchmarks/ — real index series, git-tracked because nothing regenerates them

The bias-aware benchmark side of `mf_benchmarks.py`: one availability sheet plus
one parquet per real index. **No script in this repo rebuilds any of it** —
`mf_benchmarks.py` has no builder and no `__main__`, and `bootstrap.py` never
writes here. They were assembled by an earlier ad-hoc process.

That makes this directory unbackfillable, which is why it lives in git and **not**
in `mf_cache/` (gitignored, rebuilt by `bootstrap.py`, and restored in CI from an
`actions/cache` that GitHub evicts on idle or size pressure). Same reasoning that
placed `ledger/predictions.jsonl` and `overrides/universe_manifest.csv` outside the
cache. Moved here 2026-08-03.

## What went wrong while it was inside `mf_cache/`

Every cold CI runner had no benchmark data at all, and `_availability()` was a bare
`read_csv`, so the `FileNotFoundError` propagated out of each fund's evaluation and
discarded it. Run `30774345802` dropped **118 of 136 funds** and still reported
success. Both halves are fixed (`6464ee7`): the read degrades to an empty sheet, and
the nightly carries `--max-error-rate`. Tracking the data here removes the cause.

## Contents

- **`benchmark_availability.csv`** — `benchmark_name, matched_sector_or_segment,
  source, return_type, earliest_date, latest_date, n_obs, covers_2013, notes`.
  Partly **hand-annotated**: `notes` records the specific AMFI code or yfinance
  ticker each series came from, and `source`/`return_type` drive real bias handling
  (`yfinance_PRI` series are price-return and get `DIVIDEND_YIELD_ADDBACK` applied to
  approximate TRI; `mfapi_fund` series are a real index fund's NAV, already net).
  A benchmark with `source == NONE` is deliberately declared unavailable.
- **`<slug>.parquet`** — one cleaned series per index, `_slug(name)` of the
  benchmark name (e.g. `Nifty Bank` → `nifty_bank.parquet`).

## Adding or refreshing an index

There is no builder to run — write the parquet (a `close` or `nav` column on a
DatetimeIndex), then add the matching row to the availability sheet including an
honest `source`/`return_type` and a `notes` entry naming where the series came
from. An index that is present as a parquet but absent from the sheet is treated as
**unavailable**, not guessed: `_benchmark_meta()` is the gate, and the caller falls
back to `PeerProxyResolver` exactly as it does for a sector with no mapped index.

Writing a builder that regenerates all of this from the free sources is open work —
until then, deleting a file here loses data permanently.
