"""Run the eligibility gate on the four funds requested, using verified real metadata."""

from datetime import date

import numpy as np
import pandas as pd

from mf_nfo_gate import EligibilityGate, NFOAssessor

TODAY = date(2026, 7, 13)
gate, assessor = EligibilityGate(), NFOAssessor()

# --- verified facts (sources: AMC releases / NFO listings, July 2026) ----------
FUNDS = [
    dict(
        name="Parag Parikh Flexi Cap Fund",
        amc="PPFAS AMC",
        category="Flexi Cap",
        benchmark="NIFTY 500 TRI",
        allotment=date(2013, 5, 24),
        managers=["Rajeev Thakkar", "Raunak Onkar", "Rukun Tarachandani"],
        disclosure=True,
    ),
    dict(
        name="HDFC Flexi Cap Fund",
        amc="HDFC AMC",
        category="Flexi Cap",
        benchmark="NIFTY 500 TRI",
        allotment=date(1995, 1, 1),
        managers=["Roshi Jain"],
        disclosure=True,
    ),
    dict(
        name="JM Multi Asset Allocation Fund",
        amc="JM Financial AMC",
        category="Multi Asset",
        benchmark="Nifty 500 (55%) + CRISIL Short Term Bond (30%) + Gold (10%) + Silver (5%)",
        allotment=date(2026, 7, 8),
        managers=["Asit Bhandarkar", "Killol Pandya", "Deepak Gupta"],
        disclosure=False,
    ),
    dict(
        name="TRUSTMF Large & Mid Cap Fund",
        amc="TRUST AMC",
        category="Large & Mid Cap",
        benchmark="NIFTY LargeMidcap 250 TRI",
        allotment=date(2026, 7, 24),
        managers=["Mihir Vora (CIO)", "Aakash Manghani"],
        disclosure=False,
    ),
]


def synth_nav(allot):
    """Stand-in NAV series of the correct LENGTH. Length is all the gate reads."""
    if allot >= TODAY:
        return None
    days = pd.bdate_range(allot, TODAY)
    return pd.Series(np.linspace(10, 20, len(days)), index=days)


print("\n" + "#" * 84)
print("#  ELIGIBILITY GATE — the first thing that runs, before any agent")
print("#" * 84)
rows, nfos = [], []
for f in FUNDS:
    nav = synth_nav(f["allotment"])
    v = gate.assess(f["name"], nav, f["allotment"], f["disclosure"], today=TODAY)
    rows.append(
        dict(
            Fund=f["name"][:34],
            Status=v.status.value,
            NAV_obs=v.nav_observations,
            Age_days=v.age_days if v.age_days is not None else "—",
            Scoreable="YES" if v.can_score else "NO",
        )
    )
    if not v.can_score:
        nfos.append((f, v))
print()
print(pd.DataFrame(rows).to_string(index=False))

for f, v in nfos:
    prior = {}
    if "JM" in f["amc"]:
        prior = {
            "JM Flexicap Fund": "same equity desk (Satish Ramanathan/Asit Bhandarkar) — RUN THE PIPELINE ON THIS",
            "JM Aggressive Hybrid Fund": "Asit Bhandarkar, inception 2013 — closest existing analogue to a multi-asset mandate",
        }
    if "TRUST" in f["amc"]:
        prior = {
            "TRUSTMF Mid Cap Fund": "Mihir Vora / Aakash Manghani, live since Feb-2026 — itself too young to score",
            "TRUSTMF Flexi Cap Fund": "same desk — the only Trust equity scheme with usable history",
        }
    d = assessor.assess(
        f["name"],
        f["amc"],
        f["managers"],
        f["category"],
        f["benchmark"],
        v,
        manager_prior_funds=prior,
        structural={
            "Category TER cap": "2.25% (SEBI slab, equity)",
            "First portfolio disclosure due": "~10 days after first month-end",
        },
        considerations=[
            "Multi Asset mandate: SEBI requires >=10% in each of 3 asset classes. "
            "The gold/silver sleeve is what differentiates it — and is also what "
            "makes its benchmark non-standard and peer comparison hard."
        ]
        if "Multi Asset" in f["category"]
        else [
            "Large & Mid Cap mandate: SEBI floor is 35% large + 35% mid. "
            "The remaining 30% is discretionary and is where drift risk lives."
        ],
    )
    print("\n" + d.render())
