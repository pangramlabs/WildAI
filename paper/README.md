# Paper figures and tables

Every figure and table of the paper, drawn from the result files in `../results/`:

```bash
python -m paper.make                                   # conference palette, into paper/out/{figures,tables}
python -m paper.make --palette pangram --out paper/out/preprint
python -m paper.make --only runs_table web_filter_sankey
```

It needs only the `paper` extra (`pip install -e ".[paper]"`) and runs on a CPU in about a minute. Output files carry the
names the LaTeX source includes (`figures/<name>.pdf`, `tables/<name>.tex`).

| Module | Outputs |
|---|---|
| `figures/law_overview.py` | Figure 1 |
| `figures/law_ratio.py` | change in loss against the AI ratio $r$, per size and budget |
| `figures/law_validation.py` | the law against the trained models, the value of an AI token |
| `figures/law_anatomy.py` | the law's credit and harm terms; the related-law style panels |
| `figures/law_benchmark.py` | the benchmark criteria and errors; extrapolation to 8B |
| `figures/law_filtering.py` | filtering, repetition, the compute-equivalent gain ladder |
| `figures/evaluation_masking.py` | when AI text helps; evaluating on AI and mixed text |
| `figures/web_ai_share.py` | AI share of the web and its forecast (figure and table) |
| `figures/web_filters.py` | Figure 2 (formats and FineWeb filters), filter survival Sankey |
| `figures/web_topics.py` | topic × format heatmap of the pool, topics and formats over time |
| `figures/downstream.py` | CORE against added AI or fresh human text |
| `figures/generation.py` | AI-typical phrases in generations (figure), generation table |
| `tables/runs.py` | budget convention, training grid, model inventory, architecture |
| `tables/laws.py` | the law benchmark (Table 1 and its appendix companions), coefficients, ablation, significance |

Shared pieces: `style.py` (palettes and matplotlib style), `results.py` (typed readers for `results/`), `groups.py`
(control groups of the training grid), `figures/law_common.py` (law curves and readers for `results/laws/`),
`tables/layout.py` (captions above tables). Law curves are drawn with the fitted laws of `wildai.laws`.
