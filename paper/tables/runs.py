"""Appendix tables on the trained models: the budget convention, the training grid, the model inventory and the model shapes."""

from __future__ import annotations

from pathlib import Path

from paper.groups import FITTED_DEPTHS, HELD_OUT_DEPTHS, MATCHED_BUDGETS, groups, matched_budget
from paper.results import Model, controls, models
from paper.style import SIZE_LABEL, budget_label
from paper.tables.layout import write_table

PRIMARY_SEED = 1337  # every run without a replicate uses it
ARCHITECTURE = {4: (256, 2), 6: (384, 3), 9: (640, 5), 12: (768, 6), 16: (1024, 8), 20: (1280, 10), 26: (1664, 13)}  # depth -> (width, heads)
VOCABULARY = 32768
DEPTHS = (*FITTED_DEPTHS, *HELD_OUT_DEPTHS)


def _cohort() -> list[Model]:
    cohort = [m for m in models().values() if m.split in ("fit", "held_out")]
    assert len(cohort) == 800 and sum(m.split == "fit" for m in cohort) == 726
    return cohort


def _label(budget: float) -> str:
    return budget_label(budget).split()[0]


def _dagger(depth: int) -> str:
    return r"$^{\dagger}$" if depth in HELD_OUT_DEPTHS else ""


def budget_table(out: Path) -> Path:
    """The matched budgets behind the figure labels, one row of labels and one of actual values."""

    matched = [c for c in controls().values() if c.split in ("fit", "held_out") and matched_budget(c) is not None]
    values = {b: sorted(c.human_tpp for c in matched if matched_budget(c) == b) for b in MATCHED_BUDGETS}
    sizes = {b: {c.depth for c in matched if matched_budget(c) == b} for b in MATCHED_BUDGETS}

    def listed(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]

    missing = [f"{SIZE_LABEL[d]} has no " + listed([_label(b) for b in MATCHED_BUDGETS if d not in sizes[b]]) + " budget"
               for d in DEPTHS if any(d not in sizes[b] for b in MATCHED_BUDGETS)]
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{l" + "r" * len(MATCHED_BUDGETS) + "}", r"\toprule",
        "Label in figures ($TPP_h$) & " + " & ".join(_label(b) for b in MATCHED_BUDGETS) + r" \\",
        r"Actual $D_{\mathrm H}/N$ & " + " & ".join(f"{v[len(v) // 2]:.1f}" for v in values.values()) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{$TPP_h$ of the aligned budgets. " + "; ".join(missing) + ".}",
        r"\label{tab:budget-convention}", r"\end{table}", ""])
    return write_table(out, "budget_convention_table", tex)


def runs_table(out: Path) -> Path:
    """All 800 runs as a size-by-budget grid: the matched budgets, then one column for the groups at other budgets."""

    cohort = _cohort()
    cells: dict[tuple[int, float], tuple[int, int]] = {}
    for g in groups():
        if g.budget is not None and g.control.split in ("fit", "held_out"):
            assert (g.depth, g.budget) not in cells, "two matched groups share a cell; the grid would hide one"
            cells[(g.depth, g.budget)] = (len(g.ai), len(g.human))
    for c in controls().values():
        if matched_budget(c) is not None:
            cells.setdefault((c.depth, matched_budget(c)), (0, 0))  # a control whose additions were never trained
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{l" + "c" * len(MATCHED_BUDGETS) + "|c}", r"\hline",
             r"Model $\backslash$ $TPP_h$ & " + " & ".join(_label(b) for b in MATCHED_BUDGETS) + r" & Other \\", r"\hline"]
    total = 0
    for depth in DEPTHS:
        if depth == HELD_OUT_DEPTHS[0]:
            lines.append(r"\hline")
        row = []
        for b in MATCHED_BUDGETS:
            if (depth, b) in cells:
                a, h = cells[(depth, b)]
                row.append(f"{a}" + (f"\\,+\\,{h}" if h else ""))
                total += 1 + a + h
            else:
                row.append("--")
        other = [m for m in cohort if m.depth == depth and matched_budget(controls()[m.group]) is None]
        total += len(other)
        c, a, h = (sum(m.arm == arm for m in other) for arm in ("control", "ai", "human"))
        row.append(f"{c}: {a}" + (f"\\,+\\,{h}" if h else "") if other else "--")
        lines.append(f"{SIZE_LABEL[depth]}{_dagger(depth)} & " + " & ".join(row) + r" \\")
    assert total == 800, total
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{The training grid. Each cell is one human-only control plus the number of AI-addition runs and, after the plus sign, "
              r"fresh-human-addition runs on the same human corpus. Sizes marked $\dagger$, below the rule, are reserved: every law is scored on them "
              r"and none is fitted on them.}",
              r"\label{tab:runs}", r"\end{table}", ""]
    return write_table(out, "runs_table", "\n".join(lines))


def inventory_table(out: Path) -> Path:
    """All 800 cohort models by size and arm (726 fitted, 74 held out), then the models outside the cohort."""

    cohort = _cohort()

    def counts(ms: list[Model]) -> list[int]:
        return [sum(m.arm == "control" for m in ms), sum(m.arm == "ai" for m in ms), sum(m.arm == "human" for m in ms), len(ms),
                sum(matched_budget(controls()[m.group]) is not None for m in ms), sum(m.seed != PRIMARY_SEED for m in ms)]

    def cells(values: list[int]) -> str:
        return " & ".join(str(v) for v in values)

    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{lcrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Additions} & & & \\", r"\cmidrule(lr){4-5}",
             r"Model & $TPP_h$ & Controls & AI & Human & Total & In \autoref{tab:runs} & Other seeds \\", r"\midrule"]
    for depths, label in ((FITTED_DEPTHS, "Fitted"), (HELD_OUT_DEPTHS, "Held out")):
        block: list[Model] = []
        for depth in depths:
            ms = [m for m in cohort if m.depth == depth]
            block += ms
            tpp = sorted(m.human_tpp for m in ms if m.arm == "control")
            lines.append(f"{SIZE_LABEL[depth]}{_dagger(depth)} & {tpp[0]:.1f}--{tpp[-1]:.1f} & {cells(counts(ms))}" + r" \\")
        lines += [r"\cmidrule(lr){1-8}", f"\\emph{{{label}}} & & {cells(counts(block))}" + r" \\", r"\midrule"]
    lines.append(r"\textbf{All} & & " + " & ".join(f"\\textbf{{{v}}}" for v in counts(cohort)) + r" \\")
    filtering = sum(m.split == "filtering" for m in models().values())
    repetition = sum(m.split == "repetition" for m in models().values())
    lines += [r"\midrule", r"\multicolumn{8}{l}{\emph{Outside the cohort, never fitted}} \\",
              f"\\multicolumn{{5}}{{l}}{{Filtering (\\autoref{{fig:law-ceg}}, left): 22.3\\% AI mix, and AI removed}} & {filtering} & & \\\\",
              f"\\multicolumn{{5}}{{l}}{{Repetition (\\S\\ref{{app:repetition}}): human corpus repeated}} & {repetition} & & \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Every model in the paper. The 800 models of the scaling-law cohort by size and arm: human-only controls, "
              r"AI additions and fresh-human additions, each addition trained on exactly its control's human documents. "
              r"Sizes marked $\dagger$ are held out from every fit. $TPP_h$: range over the controls. \emph{In \autoref{tab:runs}}: runs in the matched-budget grid, "
              r"controls included; the others belong to control groups at per-size budgets from earlier run generations and enter "
              r"every fit and score in the same way. \emph{Other seeds}: runs with a seed other than 1337. "
              f"Below the rule, the {filtering + repetition} filtering and repetition models, which enter no fit.}}",
              r"\label{tab:run-inventory}", r"\end{table}", ""]
    return write_table(out, "run_inventory_table", "\n".join(lines))


def architecture_table(out: Path) -> Path:
    """Shapes of the seven sizes: layers, width, heads, and parameters with and without the input embedding."""

    params = {m.depth: m.n_params for m in _cohort()}
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Layers & Width & Heads & $N$ (with input embedding) & Without \\", r"\midrule"]
    for depth in DEPTHS:
        width, heads = ARCHITECTURE[depth]
        lines.append(f"{SIZE_LABEL[depth]}{_dagger(depth)} & {depth} & {width} & {heads} & {params[depth] / 1e6:.1f}M & {(params[depth] - VOCABULARY * width) / 1e6:.1f}M" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The seven model sizes. All use the nanochat architecture. $N$ in the scaling laws counts the input embedding; "
              r"the last column subtracts it. Reserved sizes ($\dagger$) enter no fit.}",
              r"\label{tab:architecture}", r"\end{table}", ""]
    return write_table(out, "architecture_table", "\n".join(lines))
