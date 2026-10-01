"""Scaling-law tables: the benchmark of twelve laws on the held-out sizes (Table 1 and its appendix companions), our law's
coefficients, the ablation of its terms, and the bootstrap significance of its lead. All read results/laws/."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from paper.results import RESULTS
from paper.style import PALETTE
from paper.tables.layout import write_table

LAWS = RESULTS / "laws"
OURS, JOINT = "ours_logshare_11", "shukor_joint"
TARGET_LABEL = {"c4": "C4", "fw22": "FW22", "fw26_human": "FW26-H", "paloma": "Paloma", "fw26": "FW26", "fw26_ai": "FW26-AI", "cosmopedia": "Cosmo"}
COLUMN_GROUPS = (("Human text", ("c4", "fw22", "fw26_human", "paloma")), ("Mixed", ("fw26",)), ("AI text", ("fw26_ai", "cosmopedia")))
HUMAN = COLUMN_GROUPS[0][1]
MIXED_AND_AI = COLUMN_GROUPS[1][1] + COLUMN_GROUPS[2][1]
LAW_GROUPS = (("General", ("chinchilla_5",)), ("Repetition and paraphrase", ("muennighoff_7", "cd_8", "lovelace_eq8_9")),
              ("Data mixtures", ("atlas_no_repeat", "he_human_share", "hamidieh_first_order", "shukor_additive", "shukor_joint", "sedova_ai_share_coupled")),
              ("AI data", ("jain_token_weighted", OURS)))
CITATION = {"chinchilla_5": "hoffmann2022chinchilla", "muennighoff_7": "muennighoff2023scaling", "cd_8": "qin2026bridging", "lovelace_eq8_9": "lovelace2026prescriptive",
            "atlas_no_repeat": "longpre2026atlas", "he_human_share": "he2025scaling", "hamidieh_first_order": "hamidieh2025domainaware", "shukor_additive": "shukor2025scaling",
            "shukor_joint": "shukor2025scaling", "sedova_ai_share_coupled": "sedova2026scalinglawsmixturepretraining", "jain_token_weighted": "jain2024scaling"}
MUTED_SHADE = ("#C9C5C1", "#F8F7F6")  # mixed and AI-text columns: a quiet grey
COEFFICIENTS = (("E", "$E$"), ("A", "$A$"), ("alpha", r"$\alpha$"), ("B", "$B$"), ("beta", r"$\beta$"), ("eta", r"$\eta$"), ("K", "$K$"),
                ("rho", r"$\rho$"), ("gamma", r"$\gamma$"), ("u", "$u$"), ("v", "$v$"))
BENCHMARK_CAPTION = r"Paired RMSE $\times10^{3}$ on the held-out sizes, lower is better."


def _fits() -> list[dict]:
    fits = json.loads((LAWS / "fits.json").read_text(encoding="utf-8"))
    return [f for f in fits if f["fold"] == "all"]


def _scores(cutoff: str) -> dict[tuple[str, str], dict[str, str]]:
    """(law, target) -> score row over both held-out sizes; cutoff "" (every AI ratio) or "1.0" (r < 1)."""
    with (LAWS / "scores.csv").open(encoding="utf-8") as handle:
        return {(r["law"], r["target"]): r for r in csv.DictReader(handle) if r["fold"] == "all" and r["depth"] == "" and r["cutoff"] == cutoff}


def _law_name(law: str, label: str) -> str:
    """Cite each comparator; 'X et al.' becomes a textual citation so the authors are not named twice."""
    if law == OURS:
        return r"\textbf{Ours}"
    key = CITATION[law]
    return rf"\citet{{{key}}}" if label.endswith("et al.") else f"{label}~\\citep{{{key}}}"


def _shade(value: float, best: float, worst: float, colors: tuple[str, str]) -> str:
    """Darker for lower error, on a log scale between the column's best and worst."""
    t = 0.0 if worst <= best else (math.log(value) - math.log(best)) / (math.log(worst) - math.log(best))
    rgb = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in colors]
    return "".join(f"{round(dark + (light - dark) * t):02X}" for dark, light in zip(*rgb))


def _tabular(targets: tuple[str, ...], metric: str, cutoff: str, *, human: bool, title: str | None, header: bool, groups: bool) -> list[str]:
    scores = _scores(cutoff)
    labels = {f["law"]: (f["label"], f["k"]) for f in _fits()}
    laws = [law for _, members in LAW_GROUPS for law in members]
    value = {(law, t): float(scores[(law, t)][metric]) for law in laws for t in targets}
    best = {t: min(value[(law, t)] for law in laws) for t in targets}
    worst = {t: max(value[(law, t)] for law in laws) for t in targets}
    width = 2 + len(targets)
    lines = [r"\begin{tabular}{l" + "r" * (width - 1) + "}", r"\toprule"]
    if title:
        lines.append(rf"\multicolumn{{{width}}}{{l}}{{\textbf{{{title}}}}} \\")
    if header:
        lines += [rf" & & \multicolumn{{{len(targets)}}}{{c}}{{477M and 973M}} \\", rf"\cmidrule(lr){{3-{width}}}"]
    lines += ["Law & $k$ & " + " & ".join(TARGET_LABEL[t] for t in targets) + r" \\", r"\midrule"]
    for group, members in LAW_GROUPS:
        if groups:
            lines.append(rf"\multicolumn{{{width}}}{{l}}{{\emph{{{group}}}}} \\")
        for law in members:
            label, k = labels[law]
            cells = []
            for t in targets:
                v = value[(law, t)]
                text = f"{1000 * v:.2f}"
                if abs(v - best[t]) < 1e-12:
                    text = rf"\textbf{{{text}}}"
                colors = PALETTE.table_shade if human else MUTED_SHADE
                cells.append(rf"\cellcolor[HTML]{{{_shade(v, best[t], worst[t], colors)}}}{text}")
            lines.append((r"\quad " if groups else "") + _law_name(law, label) + f" & {k} & " + " & ".join(cells) + r" \\")
    return [*lines, r"\bottomrule", r"\end{tabular}"]


def _benchmark(out: Path, stem: str, label: str, caption: str, blocks: list[tuple[str, tuple[str, ...], bool]], *, metric: str = "paired_rmse",
               cutoff: str = "", titles: bool = True, placement: str = "table", header: bool = True, groups: bool = True) -> Path:
    """One or more stacked blocks of target columns. `placement`: "table" (a float), "wrap" (a right-hand wraptable, the conference
    Table 1) or "wide" (Table 1 as a full-width float for the single-column preprint)."""

    compact = placement in ("wrap", "wide")
    env = {"table": ("table", "[t]"), "wide": ("table", "[t]"), "wrap": ("wraptable", r"{r}{0.6\linewidth}")}[placement]
    tabcolsep = {"table": "4pt", "wide": "5pt", "wrap": "3pt"}[placement]
    lines = [rf"\begin{{{env[0]}}}{env[1]}", *([r"\vspace{-10pt}"] if placement == "wrap" else []), r"\centering", r"\small",
             rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}", r"\renewcommand{\arraystretch}{1.05}" if compact else r"\renewcommand{\arraystretch}{1.1}"]
    for i, (title, targets, human) in enumerate(blocks):
        if i:
            lines.append(r"\par\vspace{8pt}")
        body = _tabular(targets, metric, cutoff, human=human, title=title if titles else None, header=header, groups=groups)
        lines += [r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", *body[:-1], body[-1] + "}"]
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}", *([r"\vspace{-5pt}"] if placement == "wrap" else []), rf"\end{{{env[0]}}}", ""]
    return write_table(out, stem, "\n".join(lines))


HUMAN_BLOCK = ("Human-text targets", HUMAN, True)
AI_BLOCK = ("Mixed and AI-text targets", MIXED_AND_AI, False)


def results_table(out: Path) -> Path:
    """Table 1 as the conference version sets it: a wraptable beside the text."""
    return _benchmark(out, "law_results_table", "tab:law-results", BENCHMARK_CAPTION, [HUMAN_BLOCK], titles=False, placement="wrap", header=False, groups=False)


def results_table_wide(out: Path) -> Path:
    """Table 1 as a full-width float, for the single-column preprint."""
    return _benchmark(out, "law_results_table_float", "tab:law-results", BENCHMARK_CAPTION, [HUMAN_BLOCK], titles=False, placement="wide", header=False, groups=False)


def ai_targets_table(out: Path) -> Path:
    return _benchmark(out, "law_results_ai_targets_table", "tab:law-results-ai-targets",
                      r"\autoref{tab:law-results} for the mixed and AI-text targets: paired RMSE $\times10^{3}$ on the held-out sizes, lower is better. "
                      r"On FW26-AI and Cosmo, the exponent of the human-share multiplier of \citet{he2025scaling} sits at its bound of zero, where the law is "
                      r"exactly Chinchilla: the multiplier can only raise the loss as AI text is added, and added AI text lowers the loss on these targets.",
                      [AI_BLOCK], titles=False)


def absolute_table(out: Path) -> Path:
    return _benchmark(out, "law_benchmark_absolute_table", "tab:law-benchmark-absolute",
                      r"Absolute BPB RMSE $\times10^{3}$ on the held-out sizes, lower is better: the error in the predicted loss level itself rather than in "
                      r"the paired change against the human-only control, so the ranking can differ.",
                      [HUMAN_BLOCK, AI_BLOCK], metric="absolute_ai_rmse")


def low_ratio_table(out: Path) -> Path:
    return _benchmark(out, "law_results_lowdose_table", "tab:law-results-lowdose",
                      r"The paired errors of \autoref{tab:law-results} restricted to the held-out runs with $r<1$, the range a web crawl can reach "
                      r"(our 2026 crawl is $r=0.29$; an even split of AI and human text is $r=1$).",
                      [HUMAN_BLOCK], cutoff="1.0", titles=False)


def coefficient_table(out: Path) -> Path:
    """Our law's coefficients, fitted separately on each evaluation target."""

    fits = {f["target"]: f for f in _fits() if f["law"] == OURS}
    columns = [t for _, members in COLUMN_GROUPS for t in members]
    start, header, rules = 2, [""], []
    for title, members in COLUMN_GROUPS:
        text = rf"\textbf{{{title}}}" if title == "Human text" else title
        header.append(rf"\multicolumn{{{len(members)}}}{{c}}{{{text}}}")
        rules.append(rf"\cmidrule(lr){{{start}-{start + len(members) - 1}}}")
        start += len(members)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
             " & ".join(header) + r" \\", "".join(rules), "Coefficient & " + " & ".join(TARGET_LABEL[t] for t in columns) + r" \\", r"\midrule"]
    for key, symbol in COEFFICIENTS:
        lines.append(symbol + " & " + " & ".join(f"{fits[t]['coefficients'][key]:#.4g}" for t in columns) + r" \\")
    runs = fits["c4"]["runs"]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Coefficients of \autoref{{eq:law}} fitted separately on each evaluation target, on the same {runs} models "
              r"(19.9M to 268M) as \autoref{tab:law-results}.}",
              r"\label{tab:law-coefficients}", r"\end{table}", ""]
    return write_table(out, "law_coefficient_table", "\n".join(lines))


ABLATION_BLOCKS = (("The credit AI tokens earn (harm as in our law)", "benefit", r"Exponential window, free $\eta$, $R^\star=K t^\rho$ (ours)"),
                   ("The harm AI tokens do (credit as in our law)", "harm", r"$\gamma t^{u}n^{v}[\log(1+r)-r/(1+r)]$ on the data term (ours)"))


def ablation_table(out: Path) -> Path:
    """Our law on C4 with its credit or its harm swapped for an alternative, one at a time."""

    with (LAWS / "ablation.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.1}",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%", r"\begin{tabular}{lrr}", r"\toprule", r"Form & $k$ & 477M and 973M \\"]
    for title, group, ours_label in ABLATION_BLOCKS:
        members = [r for r in rows if r["group"] == group] + [r for r in rows if r["group"] == "both"]  # our law closes each block
        best = min(f"{1000 * float(r['held_out_paired_rmse']):.2f}" for r in members)
        lines += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{\emph{{{title}}}}} \\"]
        for r in members:
            value = f"{1000 * float(r['held_out_paired_rmse']):.2f}"
            value = rf"\textbf{{{value}}}" if value == best else value
            label = rf"\textbf{{{ours_label}}}" if r["group"] == "both" else r["description"]
            lines.append(rf"\quad {label} & {r['k']} & {value} \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Our law on C4 with the harm or benefit changed one at a time: paired RMSE $\times10^{3}$, lower is better, scored on the held-out sizes.}",
              r"\label{tab:law-ablation}", r"\end{table}", ""]
    return write_table(out, "law_ablation_table", "\n".join(lines))


def _percent(p: float) -> str:
    if p == 1.0:
        return "100"
    return f"{100 * p:.2f}" if p > 0.999 else f"{100 * p:.1f}"


def significance_table(out: Path) -> Path:
    records = json.loads((LAWS / "significance.json").read_text(encoding="utf-8"))
    rows = []
    for target in HUMAN:
        for cut, label in (("all", "all"), ("r<1", "$r<1$")):
            s = next(r for r in records if r["target"] == target and r["cut"] == cut)
            lo, hi = s["interval"][OURS]
            glo, ghi = s["gap_interval"]
            rows.append(" & ".join([TARGET_LABEL[target] if cut == "all" else "", label, f"{1000 * s['point'][OURS]:.2f} [{1000 * lo:.2f}, {1000 * hi:.2f}]",
                                    f"{1000 * s['point'][JOINT]:.2f}", f"{1000 * s['gap']:+.2f} [{1000 * glo:+.2f}, {1000 * ghi:+.2f}]",
                                    _percent(s["p_ours_lower"][JOINT]), _percent(s["p_ours_best"])]) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    all_runs = next(r for r in records if r["target"] == "c4" and r["cut"] == "all")
    low_runs = next(r for r in records if r["target"] == "c4" and r["cut"] == "r<1")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccc}", r"\toprule",
        r" & & & & & \multicolumn{2}{c}{Draws where ours is (\%)} \\", r"\cmidrule(lr){6-7}",
        r"Target & AI runs & Ours & Joint law & Joint $-$ ours & below joint & best of 12 \\", r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}",
        r"\caption{Our law's lead holds over all AI ratios on human text, but not at $r<1$ on C4 and FW22. Paired RMSE $\times10^{3}$ on the held-out runs for our law "
        r"and the joint law of \citet{shukor2025scaling}, with 95\% intervals for ours and for the gap from 10,000 bootstrap draws that resample the "
        rf"{all_runs['runs']} held-out AI runs ({low_runs['runs']} with $r<1$) by their {all_runs['groups']} control groups, every fit held fixed. The last two columns give "
        r"the share of draws in which our law has lower error than the joint law, and than all eleven comparators.}",
        r"\label{tab:law-significance}", r"\end{table}", ""])
    return write_table(out, "law_significance_table", tex)
