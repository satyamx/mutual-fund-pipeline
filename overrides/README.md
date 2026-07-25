# overrides/ — hand-curated inputs the pipeline cannot fetch

Facts here come from a human reading an AMC factsheet or SID. Nothing in this
directory is regenerable, which is why it lives in git and **not** in `mf_cache/`
(gitignored, rebuilt by `bootstrap.py`, and restored in CI from `actions/cache`,
which GitHub evicts on idle or size pressure — curated work put there is one
eviction from gone). Same reasoning that placed `ledger/predictions.jsonl` outside
the cache.

## `universe_overrides.csv`

```
amfi_code,category,sector,source_url,note
```

Supplies **inputs only**. It can never set a label, probability, score or verdict —
`mf_overrides.validate()` rejects the whole file if such a column appears, because
otherwise this becomes a way to write the answer you wanted past the screen.

### What it unblocks

A `Sectoral/Thematic` fund's cohort is keyed on `("sector", …)`. AMFI's NAVAll has
no sector field at all, and the trained manifest's 52 sector values were typed by
hand. So a thematic fund outside the trained sample has no honest peer group, and
`mf_live_score` refuses it with `SECTOR_UNRESOLVED` rather than guess — a guessed
sector fabricates the very peer set the "top quartile within cohort" label is
defined against. Adding a sourced row here resolves exactly one fund.

This is the binding constraint on coverage: **256 of the 569** canonical scoreable
schemes are Sectoral/Thematic.

### Rules enforced by `mf_overrides.validate()`

| Check | Severity | Why |
|---|---|---|
| unknown / output-setting column | ERROR, whole file rejected | keeps this to inputs only |
| `amfi_code` missing or blank | ERROR, row dropped | nothing to attach the fact to |
| duplicate `amfi_code` | ERROR, **both** dropped | last-write-wins would make meaning depend on row order |
| `category` outside the trained set | ERROR, row dropped | hand-writing a fund into an untrained cohort fabricates model coverage |
| `sector` matching no known sector | WARN, still applied | a novel sector is legitimate; a typo silently creates a one-member cohort that later fails as `THIN_COHORT` for a misleading reason |
| no `source_url` | WARN | same rule as `managers.csv` — every curated row traceable, never inferred |

ERROR rows are dropped and never partially applied.

### Workflow

```bash
./.venv/Scripts/python.exe mf_overrides.py --gaps
```

Lists what is blocked and prints ready-to-paste rows for the next funds to curate.
Fill in `<sector>` and `<source_url>`, then check your edit before it reaches a score:

```bash
./.venv/Scripts/python.exe mf_overrides.py --validate
```

Use a sector string that already appears in the manifest where one fits — matching
an existing cohort is what makes the fund scoreable, since a brand-new sector with
one member is below `COHORT_MIN_SIZE`.
