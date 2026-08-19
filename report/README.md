# Report — RB GenAI evaluation (2-page, two-column)

`report.tex` is the final 2-page evaluation report (EDA run 2 with three
judge families and lowercased inputs, Layer-2 NLI coverage/fidelity,
Layer-3 judge panel; all numbers from the verified empty-cache rerun,
cross-checked against `output/*.csv`). Figures are read from
`../output/plots/`.

## Compile

From this directory:

```sh
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

Output: `report.pdf` (must stay at exactly 2 pages).

## Conventions

- Q1/Q2 exploratory results are raw counts ("x of 36 verdicts yes"), never
  percentages (author decision). Q0 stays a rate.
- KPI 2 (topic-level segment coverage) is reported as not computable at
  this n; the Caveats section carries the diagnosis.
- Deterministic layers are exactly reproducible; judge layers are
  directionally stable but not bit-stable (stated in Methodology and
  Caveats).
- House style: no long dashes anywhere (text, captions, bibliography);
  number ranges use "to".
