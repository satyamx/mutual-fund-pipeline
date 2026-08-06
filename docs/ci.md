# CI / scheduled automation (roadmap #7)

Two GitHub Actions workflows keep the pipeline fresh at **$0** on GitHub-hosted
runners. No secrets are required — every data source is a free, no-key public API,
and Agent D (news) degrades to NEUTRAL without `ANTHROPIC_API_KEY` rather than failing.

## `.github/workflows/nightly.yml` — nightly refresh
Runs `20:30 UTC` (~02:00 IST, after AMFI's evening NAV publish) and on manual dispatch.

1. Restores `mf_cache/` via `actions/cache` (evolving-cache pattern).
2. Runs the pure selftests (`mf_holdings`, `mf_eval`) as a fast sanity gate.
3. `python bootstrap.py --from-manifest` — refreshes AMFI master + NAV for every fund
   in the tracked universe.
4. Runs the model-layer selftests **only if** the cohort artifact is present.
5. `python mf_artifact.py` — re-emits the versioned batch artifact **and** appends the
   prediction ledger.
6. Commits `ledger/` back to the repo (git-tracked, unbackfillable).
7. Uploads the artifact (`mf_cache/artifacts/*.json.gz`) as a build artifact (30-day
   history, auth-gated; runs even on failure so a bad batch stays diagnosable).
8. Publishes it to the rolling `latest-artifact` Release — the app-facing URL. Only on
   `master`, and only if step 5 exited 0. See *How the app consumes the artifact*.

**Throttle = the cache, not a sleep.** `mf_cache/` is persisted between runs and the
data adapters are cache-first with a 20h TTL, so a warm cache means bootstrap only
refetches NAV that is actually stale. That is the politeness mechanism for the free
APIs — there is no rate-limiting sleep loop to tune.

## `.github/workflows/realize-monthly.yml` — monthly realization
Runs `03:00 UTC` on the 1st. Executes `python mf_ledger.py --realize`, the OFFLINE job
that joins matured predictions (3y horizon elapsed) against fresh NAV using the frozen
cohort peer set logged at prediction time, and commits the realized outcomes. This is
what lets `mf_eval`'s `outcome_skill` eventually move from PENDING to a real number
(~2029 earliest — earlier runs are correct no-ops).

## Cold-cache seeding (one-time)
`mf_cache/` is gitignored and regenerable, so a brand-new runner has no universe
manifest and no trained model. Seed it **once locally**, then let the first Actions run
populate `actions/cache`:

```
python bootstrap.py --funds "Parag Parikh Flexi Cap" "HDFC Small Cap" ...
python mf_model.py --stage cohort          # trains + writes the cohort artifact
python mf_ledger.py --build-reference      # PSI drift reference
```

Until the cache is warm, `--from-manifest` fails fast with a clear message and the
model selftests are skipped — nothing is fabricated.

## How the app consumes the artifact
Every green nightly publishes the batch artifact to a **rolling GitHub Release** on the
tag `latest-artifact`, at a URL that no longer moves from run to run:

```
https://github.com/satyamx/mutual-fund-pipeline/releases/download/latest-artifact/mf_artifact_latest.json.gz
https://github.com/satyamx/mutual-fund-pipeline/releases/download/latest-artifact/latest.json
```

> ### ⚠️ This URL is STABLE but not yet PUBLIC — one decision still stands between it and the app
> **This repository is private**, and a private repo's release assets are not
> anonymously downloadable. Those `releases/download/...` links 404 without
> credentials; unauthenticated fetching only starts working the moment the repo goes
> public. Fetching one today means an authenticated API call
> (`GET /repos/:owner/:repo/releases/assets/:id` with `Accept: application/octet-stream`),
> which is fine for a developer or a server but **not** for the Flutter app — a token
> shipped inside a mobile binary is extractable, so that is not a route to take.
>
> The publishing step is correct either way and needs no change; what is undecided is
> where the app reads from. Three honest options:
> 1. **Make this repo public.** The URLs above go live as-is, zero code change. It also
>    publishes the code, the prediction ledger, and the hand-curated manifest.
> 2. **Mirror to a small public artifacts repo.** Keep this one private; CI pushes the
>    release to e.g. `mutual-fund-artifacts` using a PAT held as an Actions secret. The
>    token stays server-side. Costs one secret — the first this pipeline would have.
> 3. **Publish to object storage** (R2/S3/Pages) instead of Releases. Most control, most
>    moving parts, and a bill.
>
> Until one is chosen, treat the Release as the canonical artifact **store** and the app
> integration as still blocked on this single call.

The asset name is fixed on purpose — `mf_artifact.py` timestamps its output filename,
which would move the URL every night, so the run identity lives *inside* the payload
(`generated_at`, `model_id`, `pipeline_sha`) where the app already reads it.

`latest.json` is a small sidecar the app can poll to decide whether to download the
~58 KB payload at all. It is **derived from the artifact** at publish time, never
hand-maintained, and carries the `sha256` of the exact asset published beside it:

```jsonc
{ "artifact_version": "mf_artifact_v1", "generated_at": "...", "model_id": "phase_b_v3_cohort",
  "pipeline_sha": "...", "n_funds": 367, "n_errors": 0,
  "asset": "mf_artifact_latest.json.gz", "sha256": "...", "size_bytes": 59392 }
```

Three properties the app can rely on once the access question above is settled:

- **A failed run never reaches the URL.** The publish step is gated on `success()`, so a
  batch that tripped `--max-error-rate` or `--max-anchor-age-days` cannot overwrite a
  good artifact. The app may therefore see a *stale* artifact, never a broken one —
  check `generated_at` and degrade on staleness rather than assuming freshness.
- **Default branch only.** A `workflow_dispatch` from a feature branch cannot publish.
- **No history.** The tag is a rolling pointer; yesterday's bytes are replaced. Per-run
  history is the 30-day build artifact on each run, which is also where a *failed* run's
  output goes for diagnosis.

The artifact schema — including the `evaluation{}` model-health block — is documented in
`docs/app_evaluation_contract.md`.

## What is deliberately NOT here
- **Retraining is not nightly.** The `cohort_q1` model uses 3y-forward windows and does
  not change day to day; retrain deliberately (`python mf_model.py --stage cohort`) and
  rebuild the PSI reference afterwards. A separate manual/periodic workflow is the right
  home for it if/when desired.
- **True NAVAll incremental append** (parsing AMFI's daily `NAVAll.txt` and appending
  only the day's new rows per fund) is a future optimization; today the 20h cache TTL
  already prevents redundant full refetches.
