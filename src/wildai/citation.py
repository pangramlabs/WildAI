"""The paper the WildAI release accompanies: its title, arXiv page and BibTeX entry, for the Hugging Face cards."""

from __future__ import annotations

TITLE = "How Much Is an AI Token Worth? Scaling Laws for Wild AI-Generated Web Text"
ARXIV_ID = "2609.40295"
URL = f"https://arxiv.org/abs/{ARXIV_ID}"
LINK = f"[*{TITLE}*]({URL})"  # the title as a Markdown link to the arXiv page
BIBTEX = (
    "@article{russell2026wildai,\n"
    f"  title   = {{{TITLE}}},\n"
    "  author  = {Russell, Jenna and Glickenhaus, Ben and Thai, Katherine and Wieting, John and Iyyer, Mohit and Spero, Max and Emi, Bradley},\n"
    f"  journal = {{arXiv preprint arXiv:{ARXIV_ID}}},\n"
    "  year    = {2026},\n"
    f"  url     = {{{URL}}}\n"
    "}"
)
