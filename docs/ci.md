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
7. Uploads the artifact (`mf_cache/artifacts/*.json.gz`) as a build artifact.

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
Today the artifact ships as a **workflow build artifact** (downloadable from the run,
requires auth). For a stable URL the Flutter app can fetch unattended, publish it to a
**GitHub Release** instead (add a step using `softprops/action-gh-release` or `gh
release upload` against a rolling tag such as `latest-artifact`). The artifact schema —
including the `evaluation{}` model-health block — is documented in
`docs/app_evaluation_contract.md`.

## What is deliberately NOT here
- **Retraining is not nightly.** The `cohort_q1` model uses 3y-forward windows and does
  not change day to day; retrain deliberately (`python mf_model.py --stage cohort`) and
  rebuild the PSI reference afterwards. A separate manual/periodic workflow is the right
  home for it if/when desired.
- **True NAVAll incremental append** (parsing AMFI's daily `NAVAll.txt` and appending
  only the day's new rows per fund) is a future optimization; today the 20h cache TTL
  already prevents redundant full refetches.
