"""The release license is stated once and shipped with the package."""

from __future__ import annotations

from pathlib import Path

from wildai.license import HUB_ID, SPDX, legal_code

ROOT = Path(__file__).resolve().parents[1]


def test_repository_license_is_the_packaged_license() -> None:
    assert (ROOT / "LICENSE").read_text(encoding="utf-8") == legal_code()
    assert legal_code().startswith("Attribution-NonCommercial-ShareAlike 4.0 International")
    assert (SPDX, HUB_ID) == ("CC-BY-NC-SA-4.0", "cc-by-nc-sa-4.0")
