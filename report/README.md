# Report — RB GenAI evaluation (2-page, two-column)

`report.tex` is the 2-page evaluation report. Figures are read from
`../output/plots/` (run `run_plots.py` at the repo root first if they are
missing).

## Compile

From this directory:

```sh
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

Output: `report.pdf` (must stay at 2 pages).

## Placeholders

Every number in the text and in Table 1 is wrapped in `\PLACEHOLDER{...}`
(rendered as bold brackets, e.g. **[58%]**) because they come from run 1.
After the run-2 rerun (lowercased judge inputs; judges gpt-oss:120b,
deepseek-v4-flash, glm-5.2), replace each placeholder value and finally
remove the "All numbers below are from run 1..." sentence in Results.
Search for `PLACEHOLDER` to find them all.
