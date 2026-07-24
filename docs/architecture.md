# Pipeline architecture

A NAV-only Indian mutual-fund **screening** tool (not a skill-verified buy engine). It
fetches free public data, computes honest quant/compliance facts per fund, and hands a
versioned JSON artifact to the Hisaab Kitaab Flutter app. Solid = built; **dashed =
planned**. The load-bearing rule throughout is the **honesty invariant**: missing data
is a coverage flag (never a pass), no fabricated accuracy, no invented facts.

```mermaid
flowchart TB
    %% ================= DATA LAYER =================
    subgraph DATA["DATA LAYER — free, no-key public sources (cache-first, 20h TTL, self-healing)"]
        direction TB
        SRC["AMFI NAVAll.txt · mfapi.in NAV · yfinance prices · AMFI cap-band XLSX"]
        DISC["Manual holdings CSVs<br/>disclosures/&lt;code&gt;_YYYY-MM.csv"]
        DS["mf_datasources.py<br/>cache-first adapters"]
        RS["mf_realstore.py — RealNAVStore<br/>(raises, never fabricates)"]
        SRC --> DS --> RS
        DISC --> RS
    end

    %% ================= MODEL LAYER =================
    subgraph MODEL["MODEL LAYER — Phase B · within-cohort cohort_q1 (holdout AUC ~0.578, weak but real)"]
        direction TB
        LAB["mf_labels.py<br/>3y-fwd OR-label + cohort target"]
        BENCH["mf_benchmarks.py<br/>peer / benchmark resolver"]
        FEAT["mf_features.py<br/>40 as-of features (anti-lookahead)"]
        CV["mf_cv.py<br/>purged/embargoed CPCV + causal holdout"]
        MOD["mf_model.py<br/>elastic-net + HGBT challenger"]
        INF["mf_infer.py<br/>pure-numpy live inference"]
        LAB --> FEAT
        BENCH --> FEAT
        FEAT --> CV --> MOD --> INF
        LAB --> CV
    end

    RS --> FEAT
    RS --> ORCH

    %% ================= ORCHESTRATOR =================
    subgraph ORCH["LIVE ORCHESTRATOR — mf_agent_orchestrator.py"]
        direction TB
        A["Agent A — ingest + SEBI categorize"]
        B["Agent B — backtest + manager alpha (MACS)"]
        C["Agent C — profile risk scorer + raw facts"]
        D["Agent D — news / sentiment"]
        HOLD["mf_holdings.py<br/>concentration screen + SEBI single-issuer rule"]
        REC["RecommendationEngine — SYSTEM A<br/>honest SCREEN + transparent verdict"]
        SENT["mf_sentinel.py — SYSTEM B<br/>typed ALERTS + NFO dossier"]
        A --> B --> C --> D
        A --> HOLD
        C --> REC
        D --> REC
        HOLD --> REC
        A --> SENT
        HOLD --> SENT
        B --> SENT
    end

    INF --> LIVE["mf_live_score.py<br/>latest-anchor cohort signal (weak, labelled)"]
    LIVE --> REC

    %% ================= HANDOFF + MONITORING =================
    subgraph OUT["HANDOFF + MONITORING"]
        direction TB
        LED["mf_ledger.py<br/>append-only prediction ledger (git-tracked)"]
        EVAL["mf_eval.py<br/>model-health panel · GREEN/AMBER/RED"]
        ART["mf_artifact.py<br/>versioned gzip JSON (facts + verdict + alerts + evaluation)"]
        LIVE --> LED --> EVAL --> ART
        REC --> ART
        SENT --> ART
    end

    ART --> APP["Hisaab Kitaab (Flutter)<br/>Drift/SQLite — reads JSON, runs no Python"]

    %% ================= AUTOMATION =================
    subgraph CI["AUTOMATION — GitHub Actions ($0, schedule paused for testing)"]
        direction TB
        NIGHT["nightly.yml<br/>refresh → emit artifact → append ledger"]
        MONTH["realize-monthly.yml<br/>mature predictions → feeds outcome_skill"]
    end
    NIGHT -. drives .-> ART
    MONTH -. realizes .-> LED

    %% ================= PLANNED EXTENSIONS =================
    subgraph PLAN["PLANNED EXTENSIONS"]
        direction TB
        P1["Holdings ingester<br/>XLSX → digital-PDF → OCR fallback + validation gate"]
        P2["Whole-market scoring (#8)<br/>136 → thousands of funds"]
        P3["Holdings → ML features<br/>gated: needs point-in-time history + #8"]
        P4["1y-forward cohort head (#9)<br/>gated on CPCV > 0.5"]
        P5["NFO evaluation via System B dossier"]
    end
    P1 -. writes .-> DISC
    P2 -. raises sample size .-> P3
    HOLD -. today: rules only .-> P3
    P2 -. expands universe .-> MODEL
    P4 -. new head .-> MOD
    P5 -. consumes .-> SENT

    classDef planned stroke-dasharray:6 4,fill:#f6f6f6,stroke:#888,color:#444;
    class P1,P2,P3,P4,P5 planned;
```

## How to read it

- **Data → Model → Orchestrator → Handoff → App** is the main spine. The model layer
  (Phase B) trains offline; the orchestrator scores live and fuses everything into the
  two products below.
- **System A** (`RecommendationEngine`) = the honest SCREEN: raw NAV facts + a
  transparent 🟢BUY / 🔵HOLD / 🔴SELL verdict (a rule over colour-coded metrics + hard
  compliance gates), now including the **holdings concentration** metric and the
  **SEBI single-issuer** breach gate from `mf_holdings.py`.
- **System B** (`mf_sentinel.py`) = typed compliance/factor/manager ALERTS + NFO
  dossier — flags with evidence, never a fund score.
- **Monitoring**: every live cohort prediction is logged to the git-tracked
  `mf_ledger`; `mf_eval` grades drift/stability/coverage/outcome into an app-facing
  `evaluation{}` health panel. `outcome_skill` stays **PENDING** until predictions
  mature (~2029) — the framework refuses to fake accuracy before then.
- **Automation**: `nightly.yml` refreshes data + re-emits the artifact + appends the
  ledger; `realize-monthly.yml` matures predictions. Both schedules are **paused**
  (manual dispatch only) pending first-run testing.

## Planned extensions (dashed)

| # | Extension | Blocker / gate |
|---|---|---|
| P1 | Holdings ingester (XLSX→PDF→OCR + validation gate) | brittle per-AMC parsing; value is the validation gate, not the model |
| P2 | Whole-market scoring (#8) | category normalization + `OUT_OF_TRAINING_UNIVERSE` flag |
| P3 | Holdings **as ML features** | needs point-in-time holdings history + P2 for sample size; can't be backfilled → deliberately kept out of `cohort_q1` today (holdings drive *rules* only) |
| P4 | 1y-forward cohort head (#9) | ships only if it clears CPCV > 0.5 post-P2 |
| P5 | NFO evaluation via System B dossier | manager-proxy from cross-fund holdings |
