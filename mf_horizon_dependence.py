# Produced the effective-n measurement table in docs/horizons.md §3 (rho_1, DEFF12, CS per horizon).
# Kept for reproduction only: NOT run by CI; re-running it does not move a pinned divisor (mf_horizon).
# Run from the repo root on the NAV cache as it stands: ./.venv/Scripts/python.exe mf_horizon_dependence.py
import json, math, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from mf_datasources import NAVCleaner
from mf_benchmarks import PeerProxyResolver
ROOT = "."
man = pd.read_csv(ROOT + "/overrides/universe_manifest.csv", dtype={"amfi_code": str})
pr = PeerProxyResolver(man)
ckey = {c: str(pr._group_key(c)) for c in man["amfi_code"]}
cl = NAVCleaner(); nav = {}
for c in man["amfi_code"]:
    try:
        p = json.load(open(f"{ROOT}/mf_cache/mfapi_{c}.json"))
    except Exception:
        continue
    df = pd.DataFrame(p["data"]); df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    s = df.dropna(subset=["nav"]).set_index("date")["nav"].sort_index(); s = s[s > 0]; s.name = c
    nm = p["meta"].get("scheme_name", "").upper()
    s, _ = cl.clean(s, is_idcw_plan=("IDCW" in nm or "DIVIDEND" in nm))
    if len(s) >= 5:
        nav[c] = s
print("funds with nav", len(nav))
D = pd.Timestamp("2026-10-31")
allanch = pd.date_range("2013-01-31", D, freq="ME")
A0 = pd.Timestamp("2014-01-31")


def asof(idx, vals, target, tol):
    pos = idx.searchsorted(target, side="right") - 1
    if pos < 0 or (target - idx[pos]).days > tol:
        return None
    return idx[pos], vals[pos]


def fwd(s, t, h):
    sf = max(h * 30.4375 / 12, 7); miny = (h * 30.4375 - sf) / 365.25; tol = min(10, math.ceil(sf))
    idx, v = s.index, s.to_numpy()
    a = asof(idx, v, t, tol)
    if a is None or a[1] <= 0:
        return None
    b = asof(idx, v, t + pd.DateOffset(months=h), tol)
    if b is None:
        return None
    yrs = (b[0] - a[0]).days / 365.25
    if yrs < miny:
        return None
    return (b[1] / a[1]) ** (1 / yrs) - 1


def anchors_for(h):
    if h == 36:
        end = pd.Timestamp("2023-07-31")
    else:
        end = max(t for t in allanch if t + pd.DateOffset(months=h) <= D)
    return [t for t in allanch if t <= end]


def feats(s, t):
    idx, v = s.index, s.to_numpy()
    a1 = asof(idx, v, t - pd.DateOffset(months=1), 10); a12 = asof(idx, v, t - pd.DateOffset(months=12), 10)
    a0 = asof(idx, v, t, 10); a36 = asof(idx, v, t - pd.DateOffset(months=36), 10)
    S1 = a1[1] / a12[1] - 1 if a1 and a12 else np.nan
    S2 = (a0[1] / a36[1]) ** (365.25 / max((a0[0] - a36[0]).days, 1)) - 1 if a0 and a36 else np.nan
    w = s[(s.index > t - pd.DateOffset(months=12)) & (s.index <= t)]
    S3 = np.nan
    if len(w) >= 150 and a12 and a0:
        S3 = -float(w.pct_change().dropna().std())
    return S1, S2, S3


FE = {}
for c, s in nav.items():
    for t in allanch:
        if t >= A0:
            FE[(c, t)] = feats(s, t)
print("feats done", flush=True)


def hm(a, npos, nneg):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return (a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg)


def auc(score, y):
    r = pd.Series(score).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), int(n1), int(n0)


def rho_lag(M, k, demean_fund=False):
    if k >= len(M):
        return np.nan
    A = M[:-k]; B = M[k:]; ok = ~np.isnan(A) & ~np.isnan(B)
    if ok.sum() < 10:
        return np.nan
    if demean_fund:
        with np.errstate(all="ignore"):
            mu = np.nanmean(M, axis=0)
        a = (A - mu)[ok]; b = (B - mu)[ok]
    else:
        m = np.nanmean(M); a = (A - m)[ok]; b = (B - m)[ok]
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def deff(rho, L):
    return 1 + 2 * sum((1 - k / L) * rho[k] for k in range(1, L))


def block(M, h, npre, demean):
    rho = {k: rho_lag(M, k, demean) for k in range(1, 151)}
    d = {"rho1": rho[1], "D12": deff(rho, 12), "D24": deff(rho, 24)}
    tot = 1; K = 0
    for k in range(1, 151):
        if not rho[k] > 0:
            break
        tot += 2 * rho[k]; K = k
    d["Dinf"] = tot; d["K"] = K
    Mp = M[:npre]
    rp = {k: rho_lag(Mp, k, demean) for k in range(1, 12)}
    d["D12pre"] = deff(rp, 12)
    for j in (1, 2, 3):
        d[f"rho_h+{j}"] = rho.get(h + j, np.nan)
    return d


res = {}; o = ["# dependence_results\n"]
HS = [1, 3, 6, 12, 36]
for h in HS:
    anc = anchors_for(h)
    rows = []
    for c, s in nav.items():
        for t in anc:
            if t < A0:
                continue
            r = fwd(s, t, h)
            if r is not None:
                rows.append((c, t, r))
    df = pd.DataFrame(rows, columns=["amfi_code", "anchor", "R"])
    df["ck"] = df["amfi_code"].map(ckey)
    g = df.groupby(["anchor", "ck"])["R"]
    df["n"] = g.transform("size"); df["q75"] = g.transform(lambda x: x.quantile(0.75))
    df = df[df.n >= 4].copy(); df["y"] = (df.R >= df.q75).astype(float)
    anc_l = [t for t in anc if t >= A0]
    hold = anc_l[-12:]; npre = len(anc_l) - 12
    M = df.pivot(index="anchor", columns="amfi_code", values="y").reindex(anc_l).to_numpy()
    hs = df[df.anchor.isin(hold)]
    res[h] = dict(c1=dict(funds=df.amfi_code.nunique(), rows=len(df), anchors=df.anchor.nunique(), base=df.y.mean(),
                          hold_rows=len(hs), hpa=hs.groupby("anchor").size().mean()))
    res[h]["A"] = block(M, h, npre, False); res[h]["B"] = block(M, h, npre, True)
    if h == 36:
        lp = pd.read_parquet(ROOT + "/mf_cache/phase_b/labels.parquet")
        lp = lp[lp.y_cohort_q1.notna()].copy(); lp["anchor"] = pd.to_datetime(lp["anchor"])
        lp = lp[lp.anchor >= A0]
        Mq = lp.assign(y=lp.y_cohort_q1.astype(float)).pivot(index="anchor", columns="amfi_code", values="y")
        Mq = Mq.reindex(anc_l).to_numpy()
        res[h]["C"] = block(Mq, h, npre, False)
        res[h]["C"]["info"] = f"parquet rows {len(lp)} funds {lp.amfi_code.nunique()} anchors {lp.anchor.nunique()} base {lp.y_cohort_q1.astype(float).mean():.3f}"
    CS = {}
    for si, sn in enumerate(["S1", "S2", "S3"]):
        ats, hms = [], []
        for t, gdf in df[~df.anchor.isin(hold)].groupby("anchor"):
            sc = np.array([FE[(c, t)][si] for c in gdf.amfi_code]); ok = ~np.isnan(sc)
            gg = gdf[ok].copy(); gg["sc"] = sc[ok]
            if len(gg) == 0:
                continue
            gg["pr"] = gg.groupby("ck")["sc"].rank(pct=True)
            y = gg.y.to_numpy()
            if y.sum() < 2 or (1 - y).sum() < 2:
                continue
            a, npos, nneg = auc(gg.pr.to_numpy(), y)
            ats.append(a); hms.append(hm(a, npos, nneg))
        ats = np.array(ats)
        CS[sn] = dict(mean_a=ats.mean(), var=ats.var(ddof=1), hm=np.mean(hms), CS=ats.var(ddof=1) / np.mean(hms), n=len(ats))
    res[h]["CS"] = CS
    print(h, "done", flush=True)


def f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


o.append("## 1 Counts\n| h | funds | rows | anchors | base | hold rows | funds/anchor hold |\n|--|--|--|--|--|--|--|")
for h in HS:
    c = res[h]["c1"]
    o.append(f"| {h} | {c['funds']} | {c['rows']} | {c['anchors']} | {f(c['base'])} | {c['hold_rows']} | {f(c['hpa'],1)} |")
for v, nm in [("A", "A pooled-mean Pearson"), ("B", "B per-fund-demeaned")]:
    o.append(f"\n## 2{v} Serial dependence ({nm})\n| h | rho1 | DEFF12 | DEFF12 pre | DEFF24 | DEFFinf (K) | rho h+1 | h+2 | h+3 |\n|--|--|--|--|--|--|--|--|--|")
    for h in HS:
        d = res[h][v]
        o.append(f"| {h} | {f(d['rho1'])} | {f(d['D12'],2)} | {f(d['D12pre'],2)} | {f(d['D24'],2)} | {f(d['Dinf'],1)} ({d['K']}) | {f(d['rho_h+1'],2)} | {f(d['rho_h+2'],2)} | {f(d['rho_h+3'],2)} |")
d = res[36]["C"]
o.append(f"\n## 2C h=36 frozen labels.parquet (pooled-mean)\n{d['info']}\nrho1 {f(d['rho1'])} DEFF12 {f(d['D12'],2)} pre {f(d['D12pre'],2)} DEFF24 {f(d['D24'],2)} inf {f(d['Dinf'],1)} (K={d['K']}) rho h+1 {f(d['rho_h+1'],2)}")
o.append("\n## 3 Cross-fund (non-holdout anchors): mean a / Var_emp / mean HM / CS\n| h | S1 | S2 | S3 | median CS |\n|--|--|--|--|--|")
for h in HS:
    cs = res[h]["CS"]
    cells = [f"{f(cs[k]['mean_a'])} / {cs[k]['var']:.5f} / {cs[k]['hm']:.5f} / {f(cs[k]['CS'],2)} (n={cs[k]['n']})" for k in ("S1", "S2", "S3")]
    o.append(f"| {h} | " + " | ".join(cells) + f" | {f(float(np.median([cs[k]['CS'] for k in cs])),2)} |")
spans = {c: (s.index[-1] - s.index[0]).days / 365.25 for c, s in nav.items()}
n6 = sum(v >= 6 for v in spans.values())
rows = []
for c, s in nav.items():
    for t in allanch:
        if A0 <= t <= pd.Timestamp("2021-10-31"):
            r = fwd(s, t, 60)
            if r is not None:
                rows.append((c, t, r))
d5 = pd.DataFrame(rows, columns=["amfi_code", "anchor", "R"]); d5["ck"] = d5.amfi_code.map(ckey)
d5["n"] = d5.groupby(["anchor", "ck"])["R"].transform("size")
o.append(f"\n## 4 5y feasibility\nfunds with >=6y span: {n6} of {len(nav)}; (fund,anchor) rows with valid 60m fwd return (anchors 2014-01..2021-10): {len(d5)}; after cohort>=4 filter: {int((d5.n>=4).sum())}; funds with >=1 such label: {d5[d5.n>=4].amfi_code.nunique()}")
print("\n".join(o))
