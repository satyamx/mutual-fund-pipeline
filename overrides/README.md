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

### ⚠️ `sector` here means PEER GROUP, not industry sector (D2, 2026-08-06)

142 reviewed rows were accepted on 2026-08-06, and **41 of them carry a label that is
not an industry at all**: `Business Cycle` (17), `Quant` (10), `ESG` (8), `MNC` (4),
`PSU` (2). A quant fund holds across every sector; a business-cycle fund rotates
between them by design; MNC and PSU describe *ownership*, not industry.

This was a deliberate decision, not an oversight, and it follows the manifest's own
prior curation — all five labels already existed there by hand. It is defensible
because the cohort is a **peer group**: ranking quant funds against quant funds is a
better-posed "top quartile" than pooling them with unrelated thematics, and the
grouping is applied identically at train and serve time, so nothing is laundered.

**The consequence to remember:** for those five cohorts, `sector_mom_12m`,
`sector_rel_strength` and `sector_vol_1y` are **aggregates over a basket of peer
funds, not a sector index**. Do not read them as industry exposure. Every other
`sector` value in this file is a genuine industry.

Rows were accepted from `_sector_proposals.csv` after a fund-by-fund review; the
`source_url` is AMFI's NAVAll, which is genuinely where the scheme names the
proposals were read from come from. No sector was invented, and every D2 label
already existed in the manifest — that batch minted no cohort.

### Five NEW cohorts, created deliberately (D3, 2026-08-06)

D3 added 47 more rows (189 total) and **does** mint cohorts, applying the same
peer-group reading: **Innovation (12), Momentum (9), Services (5),
Special/Opportunities (5), Ethical (4)** — each at or above `COHORT_MIN_SIZE`=4.
Sub-minimum groups were folded into a natural parent rather than left as
one-member cohorts: Multi-Factor + Quality + Minimum Variance + Quantamental →
`Quant`; Sector Rotation → `Business Cycle`; Best-in-Class Strategy → `ESG` (SEBI's
own naming for that ESG sub-category).

`Ethical` is kept **separate from `ESG` on purpose** — Tata and Taurus Ethical are
Shariah-compliant, which is a different screen from an ESG mandate, and merging
them would assert a peer relationship that does not exist.

> **`--validate` reports 35 WARNs on these rows, and that is correct.** They are
> exactly the 35 whose sector has no manifest member yet (12+9+5+5+4). **Those funds
> are `THIN_COHORT` and NOT scoreable until a retrain widens the manifest** —
> insertion scoring draws cohort peers from the manifest, and a brand-new sector has
> none. The WARN clears itself once they are in.

### What is deliberately still refused

**9 foreign-equity funds** (US Bluechip, Taiwan, Japan, Asian, International, Global
Commodity) have **no sector row and must not get one.** Their gap is not a missing
sector — the model was fitted on Indian equity, so making them scoreable would claim
the holdout AUC transfers to a population it never saw. `SECTOR_UNRESOLVED` already
refuses them; the correct end state is `OUT_OF_TRAINING_UNIVERSE`, which is a code
change in `mf_universe`, not a curation row. Plus 8 with genuinely no viable cohort:
Conglomerate (3), Rural (2), and three singletons.

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

**Never AUTO-FILL it from scheme names.** Many names do carry their theme
("… Banking and Financial Services Fund"), and it is tempting to script it. The
sector defines the *peer set* the within-cohort label is computed against, so a
name-derived guess does not merely mislabel one fund — it fabricates the cohort
every one of its peers is scored relative to. Reuse an existing sector where one
genuinely fits; a new one-member sector is below `COHORT_MIN_SIZE` and will fail
later as `THIN_COHORT` anyway.

> **Amended 2026-08-06.** This rule originally read "do not fill it from scheme
> names" full stop. That is the right instinct pointed at the wrong step: what
> corrupts a cohort is a name-derived sector being *applied*, not one being
> *suggested*. `mf_overrides.py --propose` now reads the names and writes
> suggestions to `_sector_proposals.csv` — a separate file, never
> `universe_overrides.csv` — each carrying the token that fired so the call is
> auditable. A proposal still reaches a score only when a human moves the row
> across with a real `source_url` and `--validate` passes. The prohibition on
> auto-application is unchanged and is enforced structurally, not by convention.
>
> The distinction is not cosmetic: **the first real run of the proposer was
> wrong about 24 funds** — a bare `services` pattern swallowed every "Banking and
> Financial Services" fund, proposing 1 of 25. A regex over fund names is exactly
> as fallible as this section assumed. It is useful because a human reviews it.

As of 2026-08-05: **205 rows**, all blank, and they are the entire gap between the
367 funds shipping today and the 572-fund ceiling.

## `_sector_proposals.csv` (generated — review surface, NOT an input)

```
amfi_code,category,proposed_sector,matched_on,needs_human,reason,note
```

Written by `mf_overrides.py --propose --out overrides/_sector_proposals.csv`.
Nothing in the pipeline reads it; `mf_universe`/`mf_live_score` still see only
`universe_overrides.csv`, so an unreviewed proposal cannot change a single score.

As of 2026-08-06, over the 205 blocked funds: **142 proposed, 63 left for a human.**
Every proposed sector already exists in the manifest — the proposer refuses to mint a
new one, since creating a cohort is a product decision rather than a regex's call.

`matched_on` is the token that fired, so a wrong proposal is diagnosable rather than
mysterious. `reason` explains every blank, and the blanks are the interesting part:

| left blank | why |
|---|---|
| Innovation, Momentum, Quality, Multi-Factor, Minimum Variance, Quantamental | names a **strategy or factor**, not a sector |
| Sector Rotation | rotates *across* sectors — any single sector is wrong by construction |
| Ethical / Shariah | a religious-law screen, **not** the ESG sector |
| International, Japan, Taiwan, US, Asian, Global | non-domestic mandate; a global sleeve does not belong in a domestic cohort even when it names a sector |
| Services, Conglomerate, Rural, Recently-Listed-IPO | real themes with **no existing sector** to join |

Those last four are the live product question: each would need a **new** sector, and
a new sector is only viable at `COHORT_MIN_SIZE` (4) or more members. Services (4) and
Conglomerate (3) are near the line; Rural (2) is not.
