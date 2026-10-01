"""Writing LaTeX tables: the conference style puts a table's caption above its body (figures keep theirs below)."""

from __future__ import annotations

from pathlib import Path

_BODY_STARTS = ("\\begin{tabular}", "\\resizebox")


def caption_above(tex: str) -> str:
    """Move the caption and the label line after it to just before the table body. Idempotent."""

    lines = tex.split("\n")
    start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("\\caption")), None)
    body = next((i for i, line in enumerate(lines) if line.lstrip().startswith(_BODY_STARTS)), None)
    if start is None or body is None or start < body:
        return tex
    end = next(i for i in range(start, len(lines)) if lines[i].lstrip().startswith("\\label{"))
    block = lines[start:end + 1]
    rest = lines[:start] + lines[end + 1:]
    return "\n".join(rest[:body] + block + rest[body:])


def write_table(out: Path, stem: str, tex: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.tex"
    path.write_text(caption_above(tex), encoding="utf-8")
    return path
