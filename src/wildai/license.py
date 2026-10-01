"""The license of the WildAI release: the code, the models and the dataset are all CC BY-NC-SA 4.0."""

from __future__ import annotations

from pathlib import Path

SPDX = "CC-BY-NC-SA-4.0"
HUB_ID = "cc-by-nc-sa-4.0"  # the Hugging Face license identifier
URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SUMMARY = (
    f"**License: [CC BY-NC-SA 4.0]({URL}).** The WildAI code, models and dataset are released under the Creative Commons "
    "Attribution-NonCommercial-ShareAlike 4.0 International license: you may use and adapt them for non-commercial "
    "purposes, with attribution, and must share adaptations under the same license."
)


def legal_code() -> str:
    """The full license text, written as `LICENSE` into every released repository."""
    return Path(__file__).with_name("LICENSE").read_text(encoding="utf-8")
