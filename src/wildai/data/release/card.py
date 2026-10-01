"""Write the dataset card (``README.md``) of a release folder, with composition counted from the exported tables.

    python -m wildai.data.release.card --release-dir release

The card's front matter lists every config found in the folder; with ``hub.gated`` it also asks for access-request
fields (``extra_gated_prompt``, ``extra_gated_fields``). Text that depends on decisions about the released data (license,
notes on the extension and labels, contact, citation) comes from ``configs/data/release.yaml``.
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
from string import Template

import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml

from wildai.citation import LINK
from wildai.data.config import default_config, load_config
from wildai.data.release.config import ReleaseConfig
from wildai.license import HUB_ID, SUMMARY, legal_code

POOL_CONFIGS = ("human", "ai", "mixed")
CONFIG_ORDER = ("human", "ai", "mixed", "labels", "monthly_sample", "filter_audit")
CONFIG_ROWS = {
    "pools": "| `human`, `ai`, `mixed` | Pool documents by Pangram label. |",
    "labels": "| `labels` | Every pool document's metadata and labels, without text. |",
    "monthly_sample": "| `monthly_sample` | {monthly}, drawn at random from crawl months January 2021 to August 2026, with labels. |",
    "filter_audit": "| `filter_audit` | 10,000 documents of one 2026 crawl with FineWeb's filter decisions (split `fineweb`). |",
}
GATED_FIELDS = {"Name": "text", "Affiliation": "text", "Country": "country", "Intended use": "text",
                "I agree to use this dataset for research purposes only": "checkbox"}


def configs_in(release_dir: Path) -> list[dict]:
    """Front-matter ``configs`` entries for the tables present in the release folder."""

    entries = []
    for name in CONFIG_ORDER:
        directory = release_dir / name
        if not directory.exists():
            continue
        splits = sorted(p.name for p in directory.iterdir() if p.is_dir())
        files = ([{"split": s, "path": f"{name}/{s}/*.parquet"} for s in splits] if splits
                 else [{"split": "train", "path": f"{name}/*.parquet"}])
        entries.append({"config_name": name, "data_files": files, **({"default": True} if name == "human" else {})})
    return entries


def count(directory: Path, by: str | None = None) -> dict[str, tuple[int, int]]:
    """Documents and GPT-2 tokens of a table, overall (key ``""``) or grouped by a column."""

    table = ds.dataset(directory, format="parquet").to_table(columns=["token_count", *([by] if by else [])])
    if by is None:
        return {"": (len(table), int(pc.sum(table["token_count"]).as_py() or 0))}
    grouped = table.group_by(by).aggregate([("token_count", "count"), ("token_count", "sum")])
    return {row[by]: (row["token_count_count"], row["token_count_sum"]) for row in grouped.to_pylist()}


def composition(release_dir: Path) -> str:
    lines = ["| Config | Source | Documents | GPT-2 tokens |", "|---|---|---:|---:|"]
    for name in POOL_CONFIGS:
        if (release_dir / name).exists():
            for source, (docs, tokens) in sorted(count(release_dir / name, "source").items()):
                lines.append(f"| `{name}` | {source} | {docs:,} | {tokens:,} |")
    for name in ("monthly_sample", "filter_audit"):
        if (release_dir / name).exists():
            docs, tokens = count(release_dir / name)[""]
            lines.append(f"| `{name}` | | {docs:,} | {tokens:,} |")
    return "\n".join(lines)


def render(release_dir: Path, config: ReleaseConfig) -> str:
    card = config.card
    front: dict[str, object] = {
        "license": HUB_ID, "language": ["en"], "pretty_name": "WildAI",
        "task_categories": ["text-generation"], "tags": ["ai-generated-text", "pretraining", "common-crawl"],
        "configs": configs_in(release_dir),
    }
    if config.hub.gated:
        front["extra_gated_prompt"] = card.gated_prompt
        front["extra_gated_fields"] = GATED_FIELDS
    present = {name for name in ("labels", "monthly_sample", "filter_audit") if (release_dir / name).exists()}
    if any((release_dir / name).exists() for name in POOL_CONFIGS):
        present.add("pools")
    monthly = f"{count(release_dir / 'monthly_sample')[''][0]:,} documents" if "monthly_sample" in present else ""
    rows = "\n".join(row.format(monthly=monthly) for name, row in CONFIG_ROWS.items() if name in present)
    companions = {"monthly_sample": "the paper's monthly measurement sample", "filter_audit": "its filter-survival audit"}
    extras = [text for name, text in companions.items() if name in present]
    template = Template(resources.files("wildai.data.release").joinpath("card_template.md").read_text(encoding="utf-8"))
    return template.substitute(
        front_matter=yaml.safe_dump(front, sort_keys=False).strip(),
        composition=composition(release_dir),
        companions=f", and ships with {' and '.join(extras)}" if extras else "",
        measure_note=" Use the `monthly_sample` config to measure the web itself." if "monthly_sample" in present else "",
        config_rows=rows,
        extension_note=f" {card.extension_note}" if card.extension_note else "",
        pangram_note=card.pangram_note,
        windows_note=(" `pangram_windows` holds each scored window's character offsets, label and AI-assistance score."
                      if config.include_pangram_windows else ""),
        contact=card.contact,
        license_summary=SUMMARY,
        license_text=card.license_text,
        citation=card.citation.strip(),
        paper=LINK,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("release.yaml"))
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.release_dir / "README.md"
    path.write_text(render(args.release_dir, load_config(args.config, ReleaseConfig)), encoding="utf-8")
    (args.release_dir / "LICENSE").write_text(legal_code(), encoding="utf-8")
    print(f"wrote {path} and LICENSE")


if __name__ == "__main__":
    main()
