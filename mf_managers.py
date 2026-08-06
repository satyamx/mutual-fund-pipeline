"""
====================================================================================
mf_managers.py — HAND-SOURCED MANAGER <-> SCHEME HISTORY (validated, dormant-safe)
====================================================================================
WHY THIS EXISTS
---------------
No free source publishes fund-manager identity or tenure. Phase C's entire premise —
assessing an NFO with no track record by looking at what its MANAGER has run before —
depends on a mapping that a human must type from AMC factsheets and SIDs. That file
(`mf_cache/managers.csv`) does not exist yet, which is why every cross-fund and
tenure rule in `mf_sentinel` is dormant.

This module is the scaffold around it: the loader, the validator, the two lookups
Sentinel needs, and a template generator so the typing has a work surface. It ships
BEFORE the data so that the day rows are sourced, nothing else has to be written.

WHY IT IS A MODULE AND NOT A pd.read_csv CALL
----------------------------------------------
The call it replaces was three lines and had two real defects, both of the kind this
project keeps rediscovering:

  1. A bare `except: return None` meant a MALFORMED file was indistinguishable from
     an ABSENT one. Both produced "manager rules dormant", so a typo in hand-typed
     data would present as "no data yet" — forever, silently. Absence of evidence
     must never be laundered into a clean state; here it now returns issues.
  2. `rows.iloc[-1]` treated the LAST ROW IN FILE ORDER as the current manager. That
     is the same defect class as the resolver collision (`hit.iloc[0]`): correctness
     resting on the order a human happened to type things. Current manager is now
     resolved by latest `start_date`, with an open `end_date` winning ties.

THE HONESTY BOUNDARY (same as mf_overrides.py)
-----------------------------------------------
This file supplies INPUTS ONLY — who ran what, when. It can never carry a rating, a
score, or a verdict about a manager; `_FORBIDDEN_COLUMNS` rejects the whole file if
it tries, because a hand-editable "manager_score" column is a way to write a desired
answer straight past the screen. Skill is something the pipeline MEASURES (MACS, from
NAV), never something the input file asserts.

`source_url` is required on every row, mirroring the rule the project already applies
to curated sectors: a manager attribution that cannot be traced to a factsheet or SID
is a rumour, and a rumour that reaches an NFO dossier is indistinguishable from a
fabricated one.
====================================================================================
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

MANAGERS_PATH = Path("mf_cache/managers.csv")

REQUIRED_COLUMNS = ("manager_name", "scheme_name", "start_date")
OPTIONAL_COLUMNS = ("amfi_code", "end_date", "source_url", "note")
ALLOWED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Columns that would let a hand-edited file assert SKILL rather than supply history.
# MACS is computed from NAV; nothing here may shortcut it.
_FORBIDDEN_COLUMNS = {
    "macs", "manager_alpha_consistency_score", "score", "rating", "rank", "skill",
    "alpha", "verdict", "label", "probability", "recommendation",
}

SEVERITY_ERROR = "ERROR"      # row (or file) is DROPPED
SEVERITY_WARN = "WARN"        # applied, but surfaced


@dataclass(frozen=True)
class Issue:
    row: Optional[int]
    severity: str
    message: str

    def __str__(self) -> str:
        where = f"[row {self.row}] " if self.row is not None else ""
        return f"{self.severity}: {where}{self.message}"


def validate(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[Issue]]:
    """Pure validation. Returns (accepted rows, issues).

    A row with an ERROR is dropped whole, never partially applied — a half-accepted
    row is how a bad date reaches a tenure calculation."""
    issues: List[Issue] = []
    empty = pd.DataFrame(columns=list(ALLOWED_COLUMNS))
    if frame is None or frame.empty:
        return empty, issues

    cols = {str(c).strip().lower() for c in frame.columns}
    forbidden = cols & _FORBIDDEN_COLUMNS
    if forbidden:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"columns {sorted(forbidden)} would let this file ASSERT manager "
                            "skill. It supplies history only (who ran what, when); skill is "
                            "measured from NAV. Whole file rejected."))
        return empty, issues
    unknown = cols - set(ALLOWED_COLUMNS)
    if unknown:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"unknown columns {sorted(unknown)}; allowed: "
                            f"{list(ALLOWED_COLUMNS)}. Whole file rejected rather than "
                            "silently ignoring them."))
        return empty, issues
    missing = set(REQUIRED_COLUMNS) - cols
    if missing:
        issues.append(Issue(None, SEVERITY_ERROR,
                            f"missing required column(s) {sorted(missing)}. Whole file rejected."))
        return empty, issues

    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("start_date", "end_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    keep: List[int] = []
    seen: Dict[Tuple[str, str, object], int] = {}
    for i, row in df.iterrows():
        n = int(i) + 2                      # +2: 1-indexed, plus the header line
        mgr = str(row.get("manager_name") or "").strip()
        sch = str(row.get("scheme_name") or "").strip()
        if not mgr or not sch:
            issues.append(Issue(n, SEVERITY_ERROR, "manager_name and scheme_name are both required"))
            continue
        if pd.isna(row.get("start_date")):
            # Tenure and prior-fund notes are both computed FROM start_date; without
            # it the row cannot produce either, so it would sit in the file looking
            # like coverage while contributing nothing.
            issues.append(Issue(n, SEVERITY_ERROR,
                                f"{mgr} / {sch}: start_date missing or unparseable — the row "
                                "cannot produce a tenure or a prior-fund note"))
            continue
        end = row.get("end_date")
        if pd.notna(end) and end < row["start_date"]:
            issues.append(Issue(n, SEVERITY_ERROR,
                                f"{mgr} / {sch}: end_date {end.date()} precedes start_date "
                                f"{row['start_date'].date()}"))
            continue
        if not str(row.get("source_url") or "").strip():
            issues.append(Issue(n, SEVERITY_WARN,
                                f"{mgr} / {sch}: no source_url — every row should trace to a "
                                "real AMC factsheet/SID, never be inferred"))
        key = (mgr.lower(), sch.lower(), row["start_date"])
        if key in seen:
            issues.append(Issue(n, SEVERITY_WARN,
                                f"{mgr} / {sch}: duplicate of row {seen[key]} (same start_date)"))
        else:
            seen[key] = n
        keep.append(int(i))

    return df.loc[keep].reset_index(drop=True), issues


def load(path: Path = MANAGERS_PATH) -> Tuple[Optional[pd.DataFrame], List[Issue]]:
    """(rows, issues). None means genuinely ABSENT — rules stay dormant, correctly.

    An unreadable or invalid file returns an EMPTY frame plus issues, deliberately
    distinct from None: 'you have not written this yet' and 'what you wrote is
    broken' are different states, and the old loader collapsed them into silence."""
    if not Path(path).exists():
        return None, []
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 — a hand-edited CSV must not crash a batch run
        return pd.DataFrame(columns=list(ALLOWED_COLUMNS)), [
            Issue(None, SEVERITY_ERROR, f"could not be read ({exc.__class__.__name__}: {exc}). "
                                        "Manager rules are OFF — this is a broken file, not an absent one.")]
    return validate(raw)


def current_manager(df: Optional[pd.DataFrame], scheme_name: str) -> Optional[pd.Series]:
    """The manager running `scheme_name` today, by LATEST start_date — not by file order.

    An open end_date (still running) beats a closed one starting the same day, which is
    the only tie that occurs in practice: a handover typed as two rows on one date."""
    if df is None or df.empty:
        return None
    rows = df[df["scheme_name"].astype(str).str.strip().str.lower() == scheme_name.strip().lower()]
    if rows.empty:
        return None
    rows = rows.assign(_open=rows["end_date"].isna() if "end_date" in rows.columns else True)
    rows = rows.sort_values(["start_date", "_open"], ascending=[True, True])
    return rows.iloc[-1].drop(labels="_open")


def prior_funds(df: Optional[pd.DataFrame], manager: str, exclude_scheme: str) -> Dict[str, str]:
    """Other schemes this manager has run — skill attaches to the manager, not to an
    as-yet-unpriced scheme code. Rows without a start_date never reach here (validate
    drops them), so the note is always renderable."""
    if df is None or df.empty:
        return {}
    m = df["manager_name"].astype(str).str.strip().str.lower() == manager.strip().lower()
    s = df["scheme_name"].astype(str).str.strip().str.lower() != exclude_scheme.strip().lower()
    out: Dict[str, str] = {}
    for _, row in df[m & s].iterrows():
        if pd.isna(row.get("start_date")):
            continue
        end = row.get("end_date")
        end_s = end.date() if pd.notna(end) else "present"
        out[str(row["scheme_name"])] = f"managed {row['start_date'].date()}–{end_s}"
    return out


def _template(out_path: Path, limit: Optional[int] = None) -> None:
    """Seed a ready-to-fill skeleton from the tracked universe manifest.

    Seeds scheme_name + amfi_code — the two fields that are already known and the two
    most tedious to retype correctly — and leaves every HUMAN field blank. It does not
    guess a manager: unlike a sector, a manager's name appears nowhere in a scheme
    name, so there is nothing to read off and any fill would be invention."""
    import mf_labels
    manifest = mf_labels.load_manifest()
    if limit:
        manifest = manifest.head(limit)
    if out_path.resolve() == MANAGERS_PATH.resolve():
        raise SystemExit(f"refusing to overwrite {MANAGERS_PATH} with a blank skeleton — "
                         "that is hand-sourced, unbackfillable work. Pick another --out path.")
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(ALLOWED_COLUMNS))
        for _, r in manifest.iterrows():
            w.writerow(["", r["scheme_name"], "", str(r["amfi_code"]), "", "", ""])
    print(f"wrote {len(manifest)} blank row(s) to {out_path}")
    print("Fill manager_name / start_date / source_url from the AMC factsheet or SID.")
    print("end_date blank = still running. Delete rows you have not sourced — a blank")
    print("row is dropped with an ERROR, so an unfilled template validates as noise.")
    print(f"\nWhen ready, move it to {MANAGERS_PATH} and run: python mf_managers.py --validate")


def _selftest() -> None:
    def frame(rows):
        return pd.DataFrame(rows)

    base = dict(manager_name="A Sharma", scheme_name="Some Equity Fund",
                start_date="2020-01-01", source_url="https://amc.example/sid.pdf")

    ok, issues = validate(frame([base]))
    assert len(ok) == 1 and not [i for i in issues if i.severity == SEVERITY_ERROR]

    # A file that asserts SKILL is rejected whole — that is the honesty boundary.
    bad, issues = validate(frame([dict(base, macs=91)]))
    assert bad.empty and any("ASSERT manager skill" in i.message for i in issues)
    for col in ("rating", "alpha", "verdict"):
        b2, i2 = validate(frame([{**base, col: 1}]))
        assert b2.empty, col

    # Unknown column: rejected, not silently ignored.
    b3, i3 = validate(frame([dict(base, mgr_notes="x")]))
    assert b3.empty and any("unknown columns" in i.message for i in i3)

    # Missing required column.
    b4, i4 = validate(frame([{"manager_name": "A", "scheme_name": "B"}]))
    assert b4.empty and any("missing required column" in i.message for i in i4)

    # Unparseable start_date is an ERROR, not a silently-null tenure.
    b5, i5 = validate(frame([dict(base, start_date="not a date")]))
    assert b5.empty and any("cannot produce a tenure" in i.message for i in i5)

    # end before start is refused.
    b6, i6 = validate(frame([dict(base, end_date="2019-01-01")]))
    assert b6.empty and any("precedes start_date" in i.message for i in i6)

    # Missing source_url warns but still applies — same posture as mf_overrides.
    b7, i7 = validate(frame([{k: v for k, v in base.items() if k != "source_url"}]))
    assert len(b7) == 1 and any(i.severity == SEVERITY_WARN and "source_url" in i.message for i in i7)
    print("[selftest] manager file supplies HISTORY only; skill columns reject the file — PASS")

    # THE ORDERING BUG THIS MODULE EXISTS TO FIX. The predecessor took the last row in
    # FILE ORDER as the current manager, so typing history oldest-last inverted it.
    df, _ = validate(frame([
        dict(manager_name="New Mgr", scheme_name="X Fund", start_date="2024-06-01",
             end_date="", source_url="u"),
        dict(manager_name="Old Mgr", scheme_name="X Fund", start_date="2016-01-01",
             end_date="2024-05-31", source_url="u"),
    ]))
    cur = current_manager(df, "X Fund")
    assert cur is not None and cur["manager_name"] == "New Mgr", "must pick by date, not file order"
    # ...and the same file typed the other way round gives the same answer.
    df2, _ = validate(frame(list(reversed(df.to_dict("records")))))
    assert current_manager(df2, "X Fund")["manager_name"] == "New Mgr"
    # Scheme lookup is case/whitespace tolerant, and an unknown scheme is None.
    assert current_manager(df, "  x fund ")["manager_name"] == "New Mgr"
    assert current_manager(df, "Nonexistent Fund") is None

    # prior_funds excludes the scheme being assessed and renders an open end honestly.
    df3, _ = validate(frame([
        dict(manager_name="A Sharma", scheme_name="Old Fund", start_date="2015-01-01",
             end_date="2019-12-31", source_url="u"),
        dict(manager_name="A Sharma", scheme_name="Live Fund", start_date="2020-01-01",
             end_date="", source_url="u"),
        dict(manager_name="A Sharma", scheme_name="The NFO", start_date="2026-01-01",
             end_date="", source_url="u"),
    ]))
    pf = prior_funds(df3, "A Sharma", "The NFO")
    assert set(pf) == {"Old Fund", "Live Fund"}
    assert pf["Live Fund"].endswith("present") and pf["Old Fund"].endswith("2019-12-31")
    assert prior_funds(df3, "Someone Else", "The NFO") == {}
    print("[selftest] current manager resolved by DATE not file order; prior funds exclude self — PASS")

    # ABSENT vs BROKEN are different states, and that distinction is the point.
    absent, issues = load(Path("mf_cache/__definitely_absent__.csv"))
    assert absent is None and issues == []
    assert current_manager(None, "X") is None and prior_funds(None, "A", "B") == {}
    print("[selftest] absent file stays dormant; a BROKEN file reports issues instead of "
          "impersonating an absent one — PASS")

    print("[selftest] PASS — mf_managers validates hand-sourced history and refuses asserted skill")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate", action="store_true", help="validate the real managers.csv")
    ap.add_argument("--template", action="store_true",
                    help="write a blank, ready-to-fill skeleton seeded with the manifest's "
                         "scheme names + AMFI codes")
    ap.add_argument("--out", type=Path, default=Path("mf_cache/managers_template.csv"))
    ap.add_argument("--limit", type=int, default=None, help="with --template: first N funds only")
    a = ap.parse_args()
    if a.template:
        _template(a.out, a.limit)
    elif a.validate:
        rows, issues = load()
        if rows is None:
            print(f"{MANAGERS_PATH} does not exist — manager/tenure/NFO-proxy rules are DORMANT "
                  f"(correctly: no free source publishes this).")
            print("Start one with: python mf_managers.py --template --out mf_cache/managers_template.csv")
        else:
            for i in issues:
                print(f"  {i}")
            print(f"{len(rows)} valid row(s); "
                  f"{sum(1 for i in issues if i.severity == SEVERITY_ERROR)} error(s), "
                  f"{sum(1 for i in issues if i.severity == SEVERITY_WARN)} warning(s)")
    else:
        _selftest()
