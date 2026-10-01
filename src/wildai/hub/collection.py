"""The Hugging Face collection that groups the release: the paper, the dataset and one repository per model size."""

from __future__ import annotations

from typing import Literal, Protocol

from wildai.citation import ARXIV_ID

TITLE = "WildAI"
DESCRIPTION = "Language models trained on controlled mixtures of human and wild AI-generated web text, and the WildAI dataset."


class CollectionApi(Protocol):
    """The part of ``huggingface_hub.HfApi`` the collection uses."""

    def create_collection(self, title: str, *, namespace: str, description: str, private: bool, exists_ok: bool) -> object: ...

    def add_collection_item(self, collection_slug: str, item_id: str, item_type: Literal["model", "dataset", "paper"], *,
                            exists_ok: bool) -> object: ...


def add_to_collection(api: CollectionApi, namespace: str, repo_id: str, repo_type: Literal["model", "dataset"],
                      *, title: str = TITLE, private: bool = True) -> str:
    """Add ``repo_id`` to the namespace's collection named ``title``, creating it with the paper as its first item.

    Returns the collection's slug. A new collection is private when ``private`` is set; an existing one keeps its
    visibility. Adding an item that is already in the collection does nothing.
    """
    collection = api.create_collection(title, namespace=namespace, description=DESCRIPTION, private=private, exists_ok=True)
    slug: str = collection.slug  # type: ignore[attr-defined]
    api.add_collection_item(slug, ARXIV_ID, "paper", exists_ok=True)
    api.add_collection_item(slug, repo_id, repo_type, exists_ok=True)
    return slug
