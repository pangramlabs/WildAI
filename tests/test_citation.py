"""The README cites the paper with the same BibTeX entry as the Hugging Face cards."""

from pathlib import Path

from wildai.citation import BIBTEX

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_cites_the_paper() -> None:
    assert BIBTEX in README.read_text(encoding="utf-8")
