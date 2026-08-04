# overrides/ — hand-curated inputs the pipeline cannot fetch

Facts here come from a human reading an AMC factsheet or SID. Nothing in this
directory is regenerable, which is why it lives in git and **not** in `mf_cache/`
(gitignored, rebuilt by `bootstrap.py`, and restored in CI from `actions/cache`,
which GitHub evicts on idle or size pressure — curated work put there is one
eviction from gone). Same reasoning that placed `ledger/predictions.jsonl` outside
the cache.

## `universe_manifest.csv`

```
amfi_code,scheme_name,amc,category,sector,first_nav_date,last_nav_date,n_obs
```

The **trained universe** — the 367 funds the `cohort_q1` model was fitted on, and
the definition of what "in the trained universe" means everywhere in the pipeline
(`mf_universe.trained_categories`, `mf_live_score`'s `OUT_OF_TRAINING_UNIVERSE`
gate, `mf_labels`' cohort construction, `mf_artifact`'s batch loop). **52 of its
sector values were typed by hand and no script in this repo regenerates this
file** — it is not derivable from AMFI, which publishes no sector field.

It lived in `mf_cache/` until 2026-07-25, which meant the single most
unbackfillable file in the project sat in the one directory CI restores from an
evictable cache. Moving it here also makes `bootstrap.py --from-manifest` work on
a cold CI runner, since the manifest now arrives with the checkout.

Read it through `mf_labels.load_manifest()` / `mf_labels.MANIFEST_PATH`, which is
the single definition of the path; `mf_realstore.UNIVERSE_MANIFEST` is an alias of
it, not a second literal.

**Editing it is a retrain-level change**, not curation: adding a fund changes the
within-cohort labels and invalidates the holdout metrics. To make a fund scoreable
*without* retraining, add a row to `universe_overrides.csv` below — out-of-manifest
funds are scored by insertion against this frozen panel.

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

## `_sector_worklist.csv`

```
amfi_code,category,sector,source_url,note
```

The open curation queue: every `Sectoral/Thematic` fund that is otherwise
scoreable but has **no sector**, and is therefore refused with `SECTOR_UNRESOLVED`.
Regenerate with `mf_overrides.py --gaps --out overrides/_sector_worklist.csv`;
fill the `sector` column from a real source, append the filled rows to
`universe_overrides.csv`, then run `mf_overrides.py --validate`.

**Git-tracked even though an empty one is regenerable**, because a partly-filled
one is not — it is the same "one eviction from gone" argument as the rest of this
directory, and partial curation is exactly the work worth protecting. The
leading underscore marks it as a worklist, not an input: nothing in the pipeline
reads this file.

**Do not fill it from scheme names.** Many names do carry their theme
("… Banking and Financial Services Fund"), and it is tempting to script it. The
sector defines the *peer set* the within-cohort label is computed against, so a
name-derived guess does not merely mislabel one fund — it fabricates the cohort
every one of its peers is scored relative to. Reuse an existing sector where one
genuinely fits; a new one-member sector is below `COHORT_MIN_SIZE` and will fail
later as `THIN_COHORT` anyway.

As of 2026-08-05: **205 rows**, all blank, and they are the entire gap between the
367 funds shipping today and the 572-fund ceiling.
