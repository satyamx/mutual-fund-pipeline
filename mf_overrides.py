"""
====================================================================================
mf_overrides.py — HAND-CURATED UNIVERSE OVERRIDES (git-tracked, validated)
====================================================================================
WHY THIS EXISTS
---------------
Some facts the pipeline needs are published by NO free source and can only come from
a human reading an AMC factsheet or SID. The binding case today is SECTOR: a
Sectoral/Thematic fund's cohort is keyed on `("sector", ...)`, the trained manifest's
52 sector values are hand-typed, and AMFI's NAVAll carries no sector field at all.
Without a sector, 256 of the 569 canonical scoreable schemes have no honest cohort
and `mf_live_score` correctly refuses them with SECTOR_UNRESOLVED.

This module is where that human knowledge lives, and the guard rail around it.

WHY IT IS GIT-TRACKED AND NOT IN mf_cache/
------------------------------------------
`mf_cache/` is gitignored and documented as regenerable from `bootstrap.py`. Curated
labels are the exact opposite: unbackfillable once lost, because no fetch reproduces
them. The prediction ledger was moved out of `mf_cache/` for precisely this reason
(see `ledger/predictions.jsonl`), and this file follows it. Note the nightly CI
restores `mf_cache/` from `actions/cache`, which GitHub evicts on idle or size
pressure — anything curated in there is one eviction from gone.

WHAT AN OVERRIDE MAY AND MAY NOT DO (the honesty boundary)
----------------------------------------------------------
An override supplies INPUTS ONLY — `category` and `sector`. It can never set a
label, a probability, or a verdict. If it could, the file would become a way to
launder a desired answer past the screen, which is the one thing the whole project
is built to prevent. `validate()` enforces this structurally: any unknown column is
an error, not a warning.

Further checks, each guarding a real failure mode rather than tidiness:
  * `category` must already exist in the TRAINED manifest. Hand-writing a fund into
    a cohort the model never trained on would fabricate exactly the transfer of
    skill the OUT_OF_TRAINING_UNIVERSE gate exists to refuse.
  * `sector` should match a sector the manifest already uses. A typo silently
    creates a one-member cohort, which then fails downstream as THIN_COHORT — a
    misleading reason that sends you looking in the wrong place.
  * `source_url` is required, mirroring `managers.csv`'s "every row traceable to a
    real AMC factsheet/SID — never inferred" rule.
  * Duplicate `amfi_code` rows are an error: last-write-wins would make the file's
    effect depend on row order.
====================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

OVERRIDES_PATH = Path("overrides/universe_overrides.csv")

# INPUTS only. `amfi_code` identifies the row; the rest are documentation.
REQUIRED_COLUMNS = ("amfi_code",)
VALUE_COLUMNS = ("category", "sector")
META_COLUMNS = ("source_url", "note")
ALLOWED_COLUMNS = REQUIRED_COLUMNS + VALUE_COLUMNS + META_COLUMNS

# Columns that would let the file dictate an OUTPUT rather than supply an input.
# Named explicitly so the error message can say why, instead of "unknown column".
_FORBIDDEN_COLUMNS = {
    "label", "y_cohort_q1", "y", "target", "probability", "prob", "score",
    "verdict", "verdict_color", "cohort_percentile", "rank",
}

SEVERITY_ERROR = "ERROR"      # override is DROPPED — never applied
SEVERITY_WARN = "WARN"        # applied, but surfaced


@dataclass(frozen=True)
class Issue:
    amfi_code: Optional[str]
    severity: str
    message: str

    def __str__(self) -> str:
        who = f"[{self.amfi_code}] " if self.amfi_code else ""
        return f"{self.severity}: {who}{self.message}"


@dataclass(frozen=True)
class Override:
    amfi_code: str
    category: Optional[str] = None
    sector: Optional[str] = None
    source_url: Optional[str] = None
    note: Optional[str] = None


def _clean(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def validate(frame: pd.DataFrame, trained_categories: Set[str],
             known_sectors: Set[str]) -> tuple[Dict[str, Override], List[Issue]]:
    """Pure validation. Returns (accepted overrides, issues).

    Rows with an ERROR are dropped, never partially applied — a half-accepted row
    is how a typo'd sector reaches a cohort assignment.
    """
    issues: List[Issue] = []
    accepted: Dict[str, Override] = {}

    if frame.empty:
        return accepted, issues

    cols = {str(c).strip().lower() for c in frame.columns}
    forbidden = cols & _FORBIDDEN_COLUMNS
    if forbidden:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"columns {sorted(forbidden)} would let this file set an OUTPUT "
                            "(a label/score/verdict). Overrides supply inputs only — "
                            "category and sector. Whole file rejected."))
        return {}, issues
    unknown = cols - set(ALLOWED_COLUMNS)
    if unknown:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"unknown columns {sorted(unknown)}; allowed: {list(ALLOWED_COLUMNS)}. "
                            "Whole file rejected rather than silently ignoring them."))
        return {}, issues
    if "amfi_code" not in cols:
        issues.append(Issue(None, SEVERITY_ERROR, "missing required column 'amfi_code'"))
        return {}, issues

    seen: Set[str] = set()
    for _, raw in frame.iterrows():
        code = _clean(raw.get("amfi_code"))
        if not code:
            issues.append(Issue(None, SEVERITY_ERROR, "row with blank amfi_code — dropped"))
            continue
        if code in seen:
            # last-write-wins would make the file's meaning depend on row order
            issues.append(Issue(code, SEVERITY_ERROR, "duplicate amfi_code — both rows dropped"))
            accepted.pop(code, None)
            continue
        seen.add(code)

        category = _clean(raw.get("category"))
        sector = _clean(raw.get("sector"))
        source_url = _clean(raw.get("source_url"))

        if category is None and sector is None:
            issues.append(Issue(code, SEVERITY_ERROR, "row sets neither category nor sector — dropped"))
            continue
        if category is not None and category not in trained_categories:
            issues.append(Issue(code, SEVERITY_ERROR,
                                f"category {category!r} is not in the trained universe "
                                f"({len(trained_categories)} categories). Hand-writing a fund into "
                                "an untrained cohort would fabricate model coverage — dropped."))
            continue
        if sector is not None and known_sectors and sector not in known_sectors:
            # WARN not ERROR: a genuinely new sector is legitimate, but a typo here
            # silently produces a one-member cohort that later fails as THIN_COHORT.
            issues.append(Issue(code, SEVERITY_WARN,
                                f"sector {sector!r} matches no sector already in the manifest. "
                                "If this is a typo it will create a phantom one-member cohort; "
                                f"known: {sorted(known_sectors)[:6]}..."))
        if not source_url:
            issues.append(Issue(code, SEVERITY_WARN,
                                "no source_url — every curated row should be traceable to a real "
                                "AMC factsheet/SID (same rule as managers.csv)"))

        accepted[code] = Override(amfi_code=code, category=category, sector=sector,
                                  source_url=source_url, note=_clean(raw.get("note")))
    return accepted, issues


def load(path: Path = OVERRIDES_PATH, *, trained_categories: Optional[Set[str]] = None,
         known_sectors: Optional[Set[str]] = None,
         manifest: Optional[pd.DataFrame] = None) -> tuple[Dict[str, Override], List[Issue]]:
    """Read + validate the overrides file. Absent file is normal, not an error."""
    if manifest is not None:
        trained_categories = trained_categories or {str(c) for c in manifest["category"].dropna().unique()}
        known_sectors = known_sectors if known_sectors is not None else {
            str(s) for s in manifest["sector"].dropna().unique()}
    if not Path(path).exists():
        return {}, []
    try:
        frame = pd.read_csv(path, dtype=str)
    except Exception as exc:  # noqa: BLE001 — a malformed hand-edited CSV must not crash a run
        return {}, [Issue(None, SEVERITY_ERROR, f"could not read {path}: {exc}")]
    return validate(frame, set(trained_categories or ()), set(known_sectors or ()))


def _selftest() -> None:
    trained = {"Flexi Cap", "Large Cap", "Sectoral/Thematic"}
    sectors = {"Banking/Financials", "Technology", "Pharma"}

    def frame(rows): return pd.DataFrame(rows)

    # 1. A good row is accepted.
    ok, issues = validate(frame([dict(amfi_code="100001", sector="Technology",
                                      source_url="https://amc.example/sid.pdf")]), trained, sectors)
    assert list(ok) == ["100001"] and ok["100001"].sector == "Technology"
    assert not [i for i in issues if i.severity == SEVERITY_ERROR], issues
    print("[selftest] a sourced sector override is accepted — PASS")

    # 2. THE HONESTY BOUNDARY: a file that tries to set an OUTPUT is rejected whole.
    for bad in ("label", "y_cohort_q1", "probability", "verdict", "score"):
        ok, issues = validate(frame([{"amfi_code": "100001", bad: "1"}]), trained, sectors)
        assert ok == {} and any(i.severity == SEVERITY_ERROR for i in issues), bad
        assert "inputs only" in " ".join(i.message for i in issues), bad
    print("[selftest] overrides that set a label/score/verdict are rejected wholesale — PASS")

    # 3. Cannot hand-write a fund into an UNTRAINED cohort.
    ok, issues = validate(frame([dict(amfi_code="100002", category="Dividend Yield",
                                      source_url="u")]), trained, sectors)
    assert ok == {} and any("not in the trained universe" in i.message for i in issues)
    print("[selftest] category outside the trained set is refused — PASS")

    # 4. A typo'd sector WARNs (would otherwise become a silent one-member cohort).
    ok, issues = validate(frame([dict(amfi_code="100003", sector="Tecnology", source_url="u")]),
                          trained, sectors)
    assert "100003" in ok, "a novel sector is legitimate — must not be dropped"
    assert any(i.severity == SEVERITY_WARN and "phantom" in i.message for i in issues)
    print("[selftest] unrecognised sector warns about phantom cohorts, still applies — PASS")

    # 5. Unsourced row warns; duplicate codes drop BOTH (order-independence).
    ok, issues = validate(frame([dict(amfi_code="100004", sector="Pharma")]), trained, sectors)
    assert "100004" in ok and any("source_url" in i.message for i in issues)
    ok, issues = validate(frame([dict(amfi_code="1", sector="Pharma", source_url="u"),
                                 dict(amfi_code="1", sector="Technology", source_url="u")]),
                          trained, sectors)
    assert ok == {} and any("duplicate" in i.message for i in issues), ok
    print("[selftest] unsourced row warns; duplicate amfi_code drops both — PASS")

    # 6. Unknown columns are rejected rather than silently ignored, and a row that
    #    sets nothing is dropped.
    ok, issues = validate(frame([dict(amfi_code="1", sectr="Pharma")]), trained, sectors)
    assert ok == {} and any("unknown columns" in i.message for i in issues)
    ok, issues = validate(frame([dict(amfi_code="1", source_url="u")]), trained, sectors)
    assert ok == {} and any("neither category nor sector" in i.message for i in issues)
    # A missing file is normal, never an error.
    assert load(Path("overrides/__definitely_absent__.csv"), trained_categories=trained,
                known_sectors=sectors) == ({}, [])
    print("[selftest] unknown column rejected, empty row dropped, absent file is not an error — PASS")

    print("[selftest] PASS — mf_overrides accepts curated inputs and refuses curated ANSWERS")


def _gaps() -> None:
    """Worklist: which real funds are blocked, and what filling them in would unlock."""
    import mf_labels, mf_universe as U
    manifest = mf_labels.load_manifest()
    trained = U.trained_categories(manifest)
    master = pd.read_parquet("mf_cache/amfi_master.parquet")
    have, issues = load(manifest=manifest)
    for i in issues:
        print(f"  {i}")

    in_manifest = set(manifest["amfi_code"].astype(str))
    rows = []
    for code, name, raw in zip(master["amfi_code"], master["scheme_name"], master["category_raw"]):
        v = U.classify(raw, trained)
        if not v.scoreable or str(code) in in_manifest:
            continue
        plan, option = U.plan_option(name)
        if (plan, option) != ("DIRECT", "GROWTH"):
            continue
        rows.append((str(code), name, v.category))

    blocked = [r for r in rows if r[2] in ("Sectoral/Thematic",)]
    ready = [r for r in rows if r[2] not in ("Sectoral/Thematic",)]
    resolved = [r for r in blocked if r[0] in have and have[r[0]].sector]

    print(f"\ncanonical scoreable funds OUTSIDE the trained manifest: {len(rows)}")
    print(f"  scoreable now (category-keyed cohort)      : {len(ready)}")
    print(f"  BLOCKED on sector (Sectoral/Thematic)      : {len(blocked)}")
    print(f"  of those, resolved by overrides            : {len(resolved)}")
    print(f"  still needing a curated sector             : {len(blocked) - len(resolved)}")
    print(f"\noverrides file: {OVERRIDES_PATH} ({len(have)} valid row(s))")
    if blocked and len(resolved) < len(blocked):
        print("\nnext rows to curate (amfi_code,category,sector,source_url,note):")
        for code, name, _ in [b for b in blocked if b[0] not in have][:10]:
            print(f"  {code},Sectoral/Thematic,<sector>,<source_url>,{name[:52]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gaps", action="store_true", help="show what is blocked and what to curate")
    ap.add_argument("--validate", action="store_true", help="validate the real overrides file")
    a = ap.parse_args()
    if a.gaps:
        _gaps()
    elif a.validate:
        import mf_labels
        _have, _issues = load(manifest=mf_labels.load_manifest())
        for i in _issues:
            print(f"  {i}")
        print(f"{len(_have)} valid override(s); "
              f"{sum(1 for i in _issues if i.severity == SEVERITY_ERROR)} error(s), "
              f"{sum(1 for i in _issues if i.severity == SEVERITY_WARN)} warning(s)")
    else:
        _selftest()
