"""Model cards (README.md with Hugging Face metadata) for a released model and for a size-level repository."""

from __future__ import annotations

from dataclasses import dataclass, field

from wildai.citation import BIBTEX, LINK
from wildai.hub.catalog import CatalogEntry
from wildai.hub.layout import ModelLocation, ReleaseNaming
from wildai.license import HUB_ID, SUMMARY

TAGS = ("wildai", "nanochat", "scaling-laws", "ai-generated-text", "pretraining-data")

# Held-out sets in the paper's order: (key in losses.csv, label, description).
MAIN_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("c4", "C4", "C4 validation text (human-written, pre-2022)"),
    ("fw22", "FW22", "held-out FineWeb documents crawled before 2022"),
    ("paloma", "Paloma", "macro average over 16 Paloma sources"),
    ("fw26", "FW26", "held-out 2026 FineWeb-style crawl (22.3% AI tokens)"),
    ("fw26_human", "FW26-H", "the human-labeled documents of FW26"),
    ("fw26_ai", "FW26-AI", "the AI-labeled documents of FW26"),
    ("cosmopedia", "Cosmo", "Cosmopedia (AI-generated textbooks and stories)"),
)


@dataclass(frozen=True)
class CardSettings:
    naming: ReleaseNaming = field(default_factory=ReleaseNaming)
    dataset: str = "pangram/WildAI"
    citation: str = BIBTEX


def _tokens(n: float) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def _front_matter(settings: CardSettings) -> str:
    tags = "\n".join(f"- {t}" for t in TAGS)
    return (
        f"---\nlicense: {HUB_ID}\nlibrary_name: transformers\npipeline_tag: text-generation\nlanguage:\n- en\n"
        f"datasets:\n- {settings.dataset}\ntags:\n{tags}\n---\n"
    )


def training_data(entry: CatalogEntry) -> str:
    """One paragraph on what the model was trained on."""
    run, control = entry.run, entry.control
    if run.arm == "control":
        return (
            f"The human-only control of group {run.group}: {_tokens(run.human_tokens)} tokens of human-written web documents. "
            "The other models of the group add AI or fresh human text to exactly these documents, so their loss minus this "
            "model's isolates the effect of the addition."
        )
    if run.arm == "ai" and control is not None:
        return (
            f"The human documents of `{control.name}` plus wild AI-generated web documents: {_tokens(run.human_tokens)} human "
            f"and {_tokens(run.ai_tokens)} AI tokens trained on, an AI-to-human token ratio r = {run.ratio:.3g} (design "
            f"{run.added_ratio:g}), so {entry.ai_share:.1%} of the training tokens are AI-generated."
        )
    if run.arm == "human" and control is not None:
        return (
            f"The human documents of `{control.name}` plus fresh human-written web documents ({run.added_ratio:g} times the "
            f"control's tokens): {_tokens(run.human_tokens)} human tokens, the human-text counterpart of adding AI text. "
            "No AI text."
        )
    if run.arm == "repeat" and control is not None:
        return (
            f"The {_tokens(control.human_tokens)} human tokens of `{control.name}`, repeated to {_tokens(run.total_tokens)} "
            f"training tokens: as many as adding AI text at r = {run.added_ratio:g}. No AI text."
        )
    if run.arm == "natural" and entry.partner is not None:
        return (
            f"A 2026 web mix left as crawled: {_tokens(run.human_tokens)} human and {_tokens(run.ai_tokens)} AI-generated tokens "
            f"({entry.ai_share:.1%} AI). Paired with `{entry.partner.name}`, trained on the same mix without its AI text."
        )
    if run.arm == "filtered" and entry.partner is not None:
        return (
            f"The 2026 web mix of `{entry.partner.name}` with every AI-labeled document removed and not replaced: "
            f"{_tokens(run.human_tokens)} human tokens."
        )
    raise ValueError(f"cannot describe the training data of {run.name}")


SPLIT_ROLES = {
    "fit": "fitting the scaling laws (726 models, 19.9M to 268M)",
    "held_out": "testing the laws' predictions (74 held-out models, 477M and 973M)",
    "filtering": "the filtering experiment (19 pairs)",
    "repetition": "the repetition comparison (6 models)",
}


def _loss_table(entry: CatalogEntry) -> str:
    rows = [f"| {label} | {entry.losses[key]:.4f} | {about} |" for key, label, about in MAIN_TARGETS if key in entry.losses]
    table = "| Set | Bits per byte | Text |\n|---|---|---|\n" + "\n".join(rows)
    sources = sorted(k for k in entry.losses if k.startswith("paloma_"))
    if sources:
        detail = "\n".join(f"| {k.removeprefix('paloma_')} | {entry.losses[k]:.4f} |" for k in sources)
        table += f"\n\n<details><summary>Paloma sources</summary>\n\n| Source | Bits per byte |\n|---|---|\n{detail}\n\n</details>"
    return table


def _usage(location: ModelLocation) -> str:
    args = location.load_arguments()
    return (
        "```python\nimport torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f"tokenizer = AutoTokenizer.from_pretrained({args}, trust_remote_code=True)\n"
        f"model = AutoModelForCausalLM.from_pretrained({args}, trust_remote_code=True, dtype=torch.bfloat16)\n\n"
        'inputs = tokenizer("The history of the printing press", return_tensors="pt")\n'
        "output = model.generate(**inputs, max_new_tokens=40, do_sample=False)\n"
        "print(tokenizer.decode(output[0], skip_special_tokens=True))\n```\n\n"
        "The tokenizer prepends `<|bos|>` to every text, as in training; pass `add_special_tokens=False` to skip it. "
        "The model predicts `<|bos|>` where a document ends, and generation stops there. Weights are stored in bfloat16, "
        "the precision the paper's losses were computed in; `dtype=torch.float32` computes in float32 instead."
    )


def model_card(entry: CatalogEntry, settings: CardSettings) -> str:
    run = entry.run
    location = settings.naming.location(run.name)
    facts = "\n".join(
        (
            "| | |\n|---|---|",
            f"| Parameters (N, input embedding included, value embeddings excluded) | {run.n_params / 1e6:.1f}M |",
            f"| Layers / width | {run.depth} / {64 * run.depth} |",
            f"| Training tokens | {_tokens(run.total_tokens)} ({run.steps:,} optimizer steps) |",
            f"| Human / AI tokens | {_tokens(run.human_tokens)} / {_tokens(run.ai_tokens)} |",
            f"| Group / arm / seed | {run.group} / {run.arm} / {run.seed} |",
            f"| Used in the paper for | {SPLIT_ROLES[run.split]} |",
        )
    )
    return (
        f"{_front_matter(settings)}\n# {run.name}\n\n> {SUMMARY}\n\n"
        f"A {run.size.upper()} decoder-only language model trained from scratch for {LINK}, one of the paper's models "
        f"measuring how wild AI-generated web text in pretraining data changes a model's loss. It is a research artifact for "
        f"studying scaling laws, not an assistant: it has had no instruction tuning or safety training.\n\n"
        f"## Training data\n\n{training_data(entry)}\n\n{facts}\n\n"
        f"## Held-out loss\n\n{_loss_table(entry)}\n\n"
        f"## Usage\n\n{_usage(location)}\n\n"
        "## Architecture\n\n"
        "The nanochat GPT recipe: rotary embeddings, RMSNorm without weights (also on queries and keys), squared-ReLU MLP, "
        "attention windows in a repeating short-short-short-long pattern (512 tokens, then the full 2,048-token context; the "
        "last layer always full), "
        "gated value embeddings on alternating layers, a previous-token smear, per-layer residual scalars, a mid-network "
        "backout, untied input and output embeddings and logits soft-capped at 15. Vocabulary 32,768 tokens, context 2,048. "
        "The code ships with the weights (`trust_remote_code=True`) and matches the training code's forward pass.\n\n"
        f"## License\n\n{SUMMARY} The full text is in `LICENSE`.\n\n## Citation\n\n```bibtex\n{settings.citation}\n```\n"
    )


def index_card(entries: list[CatalogEntry], settings: CardSettings) -> str:
    """The card at the root of the repository: what is inside, how to load one model, and every model by size."""
    entries = sorted(entries, key=lambda e: e.run.n_params)  # stable: catalog order within a size
    location = settings.naming.location(entries[0].run.name)
    sections = []
    for size in dict.fromkeys(e.run.size for e in entries):
        group = [e for e in entries if e.run.size == size]
        rows = "\n".join(
            f"| `{e.run.name}` | {e.run.arm} | {e.run.ratio:.3g} | {_tokens(e.run.human_tokens)} | {_tokens(e.run.total_tokens)} | "
            f"{e.losses.get('c4', float('nan')):.4f} |"
            for e in group
        )
        sections.append(
            f"### {size.upper()}\n\n"
            f"| Model | Arm | r | Human tokens | Training tokens | C4 |\n|---|---|---|---|---|---|\n{rows}"
        )
    return (
        f"{_front_matter(settings)}\n# WildAI models\n\n> {SUMMARY}\n\n"
        f"The language models released with {LINK}, from 19.9M to 973M parameters, trained on controlled "
        "mixtures of human and wild AI-generated web text. Each model is in the subfolder named after it, with its own model "
        f"card; load one by passing that name as `subfolder`, which downloads only that model:\n\n{_usage(location)}\n\n"
        "## Models\n\n"
        "Names are `<size>-<group>-<arm>[-r<ratio>]`. A group `g001`-`g099` is one human-only control (`control`) and every "
        "model trained on exactly its human documents plus added AI text (`ai`), fresh human text (`human`) or its own human "
        "text repeated (`repeat`), with r added tokens per human token. `f01`-`f19` are filtering pairs: a 2026 web mix "
        "(`web-mix`) and the same mix with its AI documents removed (`ai-removed`). In the tables, r is the realized "
        "AI-to-human token ratio and C4 the held-out loss in bits per byte.\n\n"
        + "\n\n".join(sections)
        + f"\n\n## License\n\n{SUMMARY} The full text is in `LICENSE`.\n\n## Citation\n\n```bibtex\n{settings.citation}\n```\n"
    )
