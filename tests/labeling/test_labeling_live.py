"""Live checks against real services: skipped unless their credentials or hardware are present.

    PANGRAM_API_KEY=... python -m pytest tests/labeling/test_labeling_live.py -m "not gpu"
    python -m pytest tests/labeling/test_labeling_live.py -m gpu      # needs a GPU and Hugging Face access to the models
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wildai.labeling.config import EditLensSettings, PangramSettings, WebOrganizerSettings
from wildai.labeling.pangram.cache import ShardCache
from wildai.labeling.pangram.client import PangramClient
from wildai.labeling.pangram.labeler import PangramLabeler

TEXTS = {
    "human": "ok so i finally fixed the bike chain lol. took me like 2 hrs bc the tensioner was bent, "
             "had to borrow my neighbor's pliers. rode to the lake after, totally worth it",
    "ai": "In today's fast-paced digital landscape, effective time management is more important than ever. By leveraging "
          "proven strategies such as prioritization, delegation, and mindful scheduling, professionals can unlock their "
          "full potential, foster a healthier work-life balance, and drive meaningful results across every facet of life.",
    "short": "Meeting moved to 3pm Thursday, room 204.",
}


def gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not os.environ.get("PANGRAM_API_KEY"), reason="PANGRAM_API_KEY is not set")
def test_pangram_public_api_labels_three_texts(tmp_path: Path) -> None:
    client = PangramClient.from_env()
    settings = PangramSettings(poll_interval=3.0)
    assert settings.model in client.list_models()
    labels = PangramLabeler(client, settings).label(list(TEXTS), list(TEXTS.values()), ShardCache(tmp_path))
    assert all(lab.pangram_label in ("Human", "Mixed", "AI") and lab.pangram_version for lab in labels)
    assert all(0.0 <= lab.pangram_fraction_ai <= 1.0 for lab in labels)
    print({lab.id: (lab.pangram_label, lab.pangram_version, lab.pangram_fraction_ai) for lab in labels})


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU")
def test_editlens_scores_documents() -> None:
    from wildai.labeling.editlens import EditLensScorer

    labels = EditLensScorer(EditLensSettings()).score(list(TEXTS), list(TEXTS.values()))
    assert all(0 <= lab.editlens_bucket <= 3 and len(lab.editlens_probs) == 4 for lab in labels)
    assert labels[0].editlens_bucket < labels[1].editlens_bucket


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not gpu_available(), reason="needs a CUDA GPU")
def test_weborganizer_labels_documents() -> None:
    from wildai.labeling.weborganizer import WebOrganizer

    labels = WebOrganizer(WebOrganizerSettings()).label(["a"], ["https://example.com/how-to"], [TEXTS["ai"]])
    assert labels[0].topic and labels[0].format and 0.0 < labels[0].topic_score <= 1.0
