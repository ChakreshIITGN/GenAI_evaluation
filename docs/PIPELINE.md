# Pipeline

The evaluation in one picture, at the level a reader with no code in front
of them needs. Blue steps ask an AI judge; green steps are plain statistics
or local language models; white boxes are the scorecard.

```mermaid
flowchart TB
  classDef llm fill:#E3EEF5,stroke:#156C9C,color:#14181C
  classDef det fill:#EEF6E9,stroke:#72B656,color:#14181C
  classDef kpi fill:#FFFFFF,stroke:#14181C,color:#14181C
  classDef data fill:#E6F6FA,stroke:#00AAC9,color:#14181C

  DATA[("3 consumers x 10 interview questions<br/>each question: the person's real answer<br/>and the answer of their AI twin = 30 pairs")]:::data

  subgraph EDA["1 · First look: three yes/no questions on every pair"]
    direction TB
    E1["Can a judge tell which answer is AI?<br/>every answer shown alone, no hints"]:::llm
    E2["Does the AI answer sound like this person?<br/>judge first reads the person's other real answers"]:::llm
    E3["Does the AI answer say what the person said?<br/>real and AI answer compared side by side"]:::llm
    E4["How is it written?<br/>length, vocabulary, hedging, sentiment, stance<br/>measured with plain statistics, real vs AI"]:::det
  end
  DATA --> EDA

  subgraph L["2 · Full evaluation, three lenses"]
    direction TB
    subgraph L1["Lens 1 · Does it sound the same?"]
      direction TB
      S1["style fingerprint of each answer<br/>compared pairwise, real vs AI, with effect sizes"]:::det
      S2["overall meaning similarity<br/>baseline only, easy to score high by staying on topic"]:::det
    end
    subgraph L2["Lens 2 · Does it say the same things?"]
      direction TB
      C1["split every answer into sentences"]:::det
      C2["for each real sentence: does the AI cover it?<br/>for each AI sentence: is it faithful, invented, or contradicted?"]:::det
      C3["group all sentences into themes<br/>covered · blind spot (only the person says it)<br/>· phantom (only the AI says it)"]:::det
    end
    subgraph L3["Lens 3 · Would an expert believe it?"]
      direction TB
      J0["three independent AI judges from different model families<br/>same items for all, so verdicts are comparable"]:::llm
      J1["Judge A · individuation<br/>given the person's other answers,<br/>is this AI answer recognisably them? 1-5"]:::llm
      J2["Judge B · believability, blind<br/>all 60 answers shuffled, real and AI unlabelled,<br/>how human does each one read? 1-5"]:::llm
      J0 --> J1
      J0 --> J2
    end
    C1 --> C2
    C1 --> C3
  end
  EDA --> L

  subgraph K["3 · Scorecard"]
    direction TB
    K1["KPI 1 · Content fidelity<br/>coverage and faithfulness per person<br/>red / amber / green"]:::kpi
    K2["KPI 2 · Theme coverage<br/>count of blind spots and phantoms<br/>reported as not computable if themes do not separate at this sample size"]:::kpi
    K3["KPI 3 · Believability<br/>can judges tell real from AI?<br/>individuation score, and agreement between judges"]:::kpi
  end
  C2 --> K1
  C3 --> K2
  J1 --> K3
  J2 --> K3

  REPORT["Two-page report and slides<br/>every figure traceable to a run"]:::data
  S1 --> REPORT
  K --> REPORT

  GUARD["Safeguards on every judge call<br/>fixed seed and zero temperature · answers lowercased so typos cannot give the game away<br/>every reply cached, so a re-run reproduces exactly"]:::llm
  GUARD -.-> EDA
  GUARD -.-> L3
```

Also run, not on the chart: the first-look step was re-executed by
autonomous judge agents with full call logs, as an independent check that
the verdicts hold. For the file-level view of each step see the module
docstrings in `src/` and the entrypoints `run_eda.py`, `run_all.py`,
`run_plots.py`.
