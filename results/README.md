# Results

Every number in the paper's figures and tables comes from the files here. `python -m paper.make` draws them;
the `wildai` package regenerates them from the released models and data.

| File | Rows | Produced by |
|---|---|---|
| `models.csv` | one per released model (844) | the training grid |
| `losses.csv` | model × evaluation set | `wildai.evaluation.bpb` |
| `downstream.csv` | model | `wildai.evaluation` (CORE) |
| `laws/` | fitted laws and their held-out scores | `wildai.laws.run` |
| `web/monthly_ai_share.csv` | crawl month | `wildai.data` (monthly sample + Pangram API labels) |
| `web/ai_share_forecast.csv` | month | `wildai.data.forecast` |
| `web/monthly_topic_format.csv` | month × topic × format × label | `wildai.data` (WebOrganizer labels) |
| `web/pool_topic_format.csv` | crawl year × topic × format × label | `wildai.data` |
| `web/filter_audit.json` | pipeline × filter stage | `wildai.data` (filter audit) |
| `generation/ai_phrase_rates.csv` | model × prompt set | `wildai.evaluation.generate` |
| `generation/pangram_labels.csv` | model × prompt set | `wildai.evaluation.generate` + Pangram API |

## `models.csv`

| Column | Meaning |
|---|---|
| `name` | public name, also the Hugging Face model name: `<size>-<group>-<arm>[-r<ratio>]` |
| `group` | `g001`–`g099`: one human-only control and every run trained on exactly its human documents plus added text; `f01`–`f19`: a filtering pair |
| `arm` | `control` (human text only), `ai` (control's human documents + AI documents), `human` (+ fresh human documents), `repeat` (control's human documents repeated), `natural` (a 2026 web mix, 22.3 % AI tokens), `filtered` (the same mix with its AI documents removed) |
| `size`, `depth` | model size label and number of layers |
| `n_params` | the paper's $N$: parameters including the input embedding, excluding value embeddings |
| `seed` | training seed |
| `added_ratio` | design ratio $r$ of added to human tokens (for `repeat`, the AI ratio whose token count the repetition matches) |
| `human_tokens`, `ai_tokens` | unique human and AI tokens seen in training |
| `total_tokens`, `steps` | tokens trained (including repeats) and optimizer steps |
| `split` | `fit` (726 models, 19.9M–268M), `held_out` (74 models, 477M and 973M), `filtering`, `repetition` |

## `losses.csv`

Bits per byte of the final checkpoint on each evaluation set: `c4`, `fw22` (FineWeb 2021 crawls), `fw26` (a held-out
2026 crawl sample, 22.3 % AI tokens), `fw26_human` and `fw26_ai` (its Pangram-labeled human and AI documents),
`cosmopedia`, `paloma` (macro average over Paloma's 16 sources) and `paloma_<source>`. Every document is scored at
most once: 15.7M tokens each of C4 and FW22, 12.1M of FW26, 9.6M of FW26-human, 2.2M of FW26-AI and 18.5M of Cosmopedia;
Paloma's validation split in full.
