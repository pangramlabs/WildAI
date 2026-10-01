"""FineWeb's recipe on a synthetic WARC file (runs only in the pinned collection environment with DataTrove 0.2.0)."""

from __future__ import annotations

import io
import random
from pathlib import Path

import pyarrow as pa
import pytest

datatrove = pytest.importorskip("datatrove")
warcio = pytest.importorskip("warcio")

from tests.data.builders import WordCounter  # noqa: E402  (after the importorskips)
from wildai.data.collect.common_crawl import run_crawl  # noqa: E402  (after the importorskips)
from wildai.data.collect.config import RecipeSettings  # noqa: E402  (after the importorskips)
from wildai.data.commoncrawl import WarcFrame  # noqa: E402  (after the importorskips)
from wildai.data.parquet_io import read_dataset  # noqa: E402  (after the importorskips)

DUMP = "CC-MAIN-2099-01"
WARC = f"crawl-data/{DUMP}/segments/1/warc/test-00000.warc.gz"
TOPICS = ["gardening", "bicycle repair", "bread baking", "local history", "bird watching", "chess openings"]
WORDS = (
    "the", "of", "and", "to", "in", "is", "that", "it", "was", "for", "on", "are", "with", "as", "his", "they", "at",
    "be", "this", "from", "have", "or", "by", "one", "had", "not", "but", "what", "all", "were", "when", "we",
    "there", "can", "an", "your", "which", "their", "said", "if", "do", "will", "each", "about", "how", "up", "out",
    "them", "then", "she", "many", "some", "so", "these", "would", "other", "into", "has", "more", "her", "two",
    "like", "him", "see", "time", "could", "no", "make", "than", "first", "been", "its", "who", "now", "people", "my",
    "made", "over", "did", "down", "only", "way", "find", "use", "may", "water", "long", "little", "very", "after",
    "words", "called", "just", "where", "most", "know", "get", "through", "back", "much", "before", "go", "good",
    "new", "write", "our", "used", "me", "man", "too", "any", "day", "same", "right", "look", "think", "also",
    "around", "another", "came", "come", "work", "three", "word", "must", "because", "does", "part", "even", "place",
    "well", "such", "here", "take", "why", "things", "help", "put", "years", "different", "away", "again", "off",
    "went", "old", "number", "great", "tell", "men", "say", "small", "every", "found", "still", "between", "name",
    "should", "home", "big", "give", "air", "line", "set", "own", "under", "read", "last", "never", "us", "left",
    "end", "along", "while", "might", "next", "sound", "below", "saw", "something", "thought", "both", "few", "those",
    "always", "looked", "show", "large", "often", "together", "asked", "house", "world", "going", "want", "school",
    "important", "until", "form", "food", "keep", "children", "feet", "land", "side", "without", "boy", "once",
    "animals", "life", "enough", "took", "sometimes", "four", "head", "above", "kind", "began", "almost", "live",
    "page", "got", "earth", "need", "far", "hand", "high", "year", "mother", "light", "parts", "country", "father",
    "let", "night", "following", "picture", "being", "study", "second", "eyes", "soon", "times", "story", "boys",
    "since", "white", "days", "ever", "paper", "hard", "near", "sentence", "better", "best", "across", "during",
    "today", "others", "however", "sure", "means", "knew", "try", "told", "young", "miles", "sun", "ways", "thing",
    "whole", "hear", "example", "heard", "several", "change", "answer", "room", "sea", "against", "top", "turned",
    "learn", "point", "city", "play", "toward", "five", "using", "himself", "usually", "money", "seen", "car",
    "morning",
)


def sentences(topic: str, count: int = 12) -> list[str]:
    """Distinct English-looking sentences with almost no repeated 5-grams, within or across documents."""

    rng = random.Random(topic)
    return [" ".join(rng.choices(WORDS, k=14)).capitalize() + f" {topic}." for _ in range(count)]


def article(topic: str, contact: str = "") -> str:
    body = "".join(f"<p>{s}</p>" for s in sentences(topic))
    note = f"<p>If you have questions about {topic}, you can write to us{contact} and we will answer soon.</p>"
    return f"<html><head><title>{topic}</title></head><body><article><h1>{topic}</h1>{body}{note}</article></body></html>"


def write_warc(path: Path, pages: list[tuple[str, str]]) -> None:
    from warcio.statusandheaders import StatusAndHeaders
    from warcio.warcwriter import WARCWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = WARCWriter(handle, gzip=True)
        for url, html in pages:
            headers = StatusAndHeaders("200 OK", [("Content-Type", "text/html; charset=utf-8")], protocol="HTTP/1.1")
            record = writer.create_warc_record(url, "response", payload=io.BytesIO(html.encode()), http_headers=headers,
                                               warc_headers_dict={"WARC-Identified-Payload-Type": "text/html"})
            writer.write_record(record)


@pytest.fixture
def frame(tmp_path: Path) -> WarcFrame:
    pages = [(f"https://example.org/{t.replace(' ', '-')}", article(t)) for t in TOPICS]
    pages.append(("https://mirror.example.net/gardening", article("gardening")))  # a duplicate page
    pages.append(("https://example.org/contact", article("pottery", contact=" at pottery.club@example.com")))
    pages.append(("https://example.de/garten", "<html><body><p>" + "Der Garten ist schön und groß. " * 40 + "</p></body></html>"))
    pages.append(("https://example.org/short", "<html><body><p>Too short.</p></body></html>"))
    write_warc(tmp_path / "work" / "batches" / "00000" / "raw" / Path(WARC).name, pages)  # already downloaded
    return WarcFrame(dump=DUMP, seed=0, crawl_files=1, paths=[WARC])


def test_recipe_filters_deduplicates_and_anonymizes(frame: WarcFrame, tmp_path: Path) -> None:
    settings = RecipeSettings(batch_files=1, min_chars=0, extraction_timeout=10.0)  # 0.1 s is load-sensitive
    documents = run_crawl(frame, settings, tmp_path / "work", tmp_path / "out", workers=1, counter=WordCounter())
    table = read_dataset(tmp_path / "out")
    urls = table["url"].to_pylist()
    assert documents == len(table) == 7  # six topics (one copy of the duplicate) plus the contact page
    assert "https://example.de/garten" not in urls and "https://example.org/short" not in urls
    assert sum("gardening" in u for u in urls) == 1
    contact = table.filter(pa.compute.equal(table["url"], "https://example.org/contact"))["text"][0].as_py()
    assert "pottery.club@example.com" not in contact and "@example." in contact
    assert set(table["source"].to_pylist()) == {"common_crawl"} and set(table["warc_path"].to_pylist()) == {WARC}
    assert all(table["pii_anonymized"].to_pylist()) and all(s > 0.65 for s in table["language_score"].to_pylist())
    assert run_crawl(frame, settings, tmp_path / "work", tmp_path / "out", workers=1) == documents  # resumes as done


def test_filter_audit_scores_stages_in_order() -> None:
    from wildai.data.measure.filter_audit import score_stages

    sample = pa.table({"id": ["en", "de", "short"], "url": ["u"] * 3,
                       "text": ["\n".join(sentences("gardening")), "Der Garten ist schön und groß. " * 40, "Too short."]})
    stages = score_stages(sample).to_pylist()
    assert stages[0]["english"] and stages[0]["fineweb_quality"] and stages[0]["removed_at"] is None
    assert stages[1]["removed_at"] == "language" and not any(stages[1][k] for k in ("english", "c4", "fineweb_quality"))
    assert stages[2]["removed_at"] is not None and not stages[2]["fineweb_quality"]
